import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from models import Transformer


class DecayBiasedNewsCrossAttention(nn.Module):
    """Horizon-level cross-attention over a fixed news memory.

    Compared with the existing decay-pooling adapter, this module keeps the
    selected headlines as individual memory tokens. Each forecast horizon forms
    a query, attends to the news memory, and receives a learned temporal decay
    bias in the attention logits.
    """

    def __init__(self, configs):
        super().__init__()
        self.seq_len = int(configs.seq_len)
        self.pred_len = int(configs.pred_len)
        self.freq = str(getattr(configs, 'freq', 'd'))
        self.enc_in = int(getattr(configs, 'enc_in', getattr(configs, 'c_out', 1)))
        self.d_model = int(getattr(configs, 'd_model', 64))
        self.n_heads = int(getattr(configs, 'n_heads', 4))
        if self.d_model % self.n_heads != 0:
            raise ValueError(f'd_model={self.d_model} must be divisible by n_heads={self.n_heads}')
        self.head_dim = self.d_model // self.n_heads

        self.news_emb_dim = int(getattr(configs, 'news_emb_dim', 32))
        self.news_decay_hidden_dim = int(getattr(configs, 'news_decay_hidden_dim', 64))
        self.disable_time_decay = bool(getattr(configs, 'disable_time_decay', False))
        self.use_fixed_time_decay = bool(getattr(configs, 'news_cross_use_fixed_decay', False))
        self.fixed_decay_alpha = float(getattr(configs, 'news_cross_fixed_decay_alpha', 0.05))
        self.gap_scale = float(getattr(configs, 'news_cross_gap_scale', 1.0))
        self.gap_scale = max(self.gap_scale, 1e-6)

        self.price_state_proj = nn.Linear(self.seq_len, self.d_model)
        self.news_proj = nn.Linear(self.news_emb_dim, self.d_model)
        self.horizon_proj = nn.Sequential(
            nn.Linear(1, self.d_model),
            nn.GELU(),
            nn.Linear(self.d_model, self.d_model),
        )
        self.query_proj = nn.Linear(self.d_model * 2, self.d_model)
        self.key_proj = nn.Linear(self.d_model, self.d_model)
        self.value_proj = nn.Linear(self.d_model, self.d_model)
        self.out_proj = nn.Linear(self.d_model, self.d_model)
        self.attn_dropout = nn.Dropout(getattr(configs, 'dropout', 0.0))

        self.decay_mlp = nn.Sequential(
            nn.Linear(self.d_model + 1, self.news_decay_hidden_dim),
            nn.GELU(),
            nn.Linear(self.news_decay_hidden_dim, self.n_heads),
        )
        # Start with a mild temporal penalty so semantic attention can still
        # learn before the decay branch sharpens.
        nn.init.constant_(self.decay_mlp[-1].bias, -3.0)

        self.fusion = nn.Sequential(
            nn.Linear(self.d_model * 2, self.d_model),
            nn.GELU(),
            nn.Dropout(getattr(configs, 'dropout', 0.0)),
        )
        self.news_output = nn.Linear(self.d_model, 1)
        self.latest_debug_stats = None

    def _step_in_days(self):
        freq = self.freq.lower()
        base_steps = {
            's': 1.0 / 86400.0,
            't': 1.0 / 1440.0,
            'min': 1.0 / 1440.0,
            'h': 1.0 / 24.0,
            'd': 1.0,
            'b': 1.0,
            'w': 7.0,
            'm': 30.0,
            'a': 365.0,
            'y': 365.0,
        }
        return base_steps.get(freq, 1.0)

    def _horizon_offsets(self, device, dtype):
        offsets = torch.arange(1, self.pred_len + 1, device=device, dtype=dtype)
        return offsets * self._step_in_days()

    def _price_tokens(self, x_enc_norm):
        return self.price_state_proj(x_enc_norm.permute(0, 2, 1))

    def _split_heads(self, x):
        batch, length, _ = x.shape
        return x.view(batch, length, self.n_heads, self.head_dim).transpose(1, 2)

    def _debug_stats(self, news_mask, news_time_gaps, alpha, attn, temporal_bias):
        with torch.no_grad():
            active = news_mask.sum(dim=-1).float() if news_mask is not None else None
            valid = news_mask > 0 if news_mask is not None else torch.ones_like(news_time_gaps, dtype=torch.bool)
            valid_gaps = news_time_gaps[valid]
            stats = {
                'disable_time_decay': bool(self.disable_time_decay),
                'use_fixed_time_decay': bool(self.use_fixed_time_decay),
                'active_news_mean': active.mean().item() if active is not None else None,
                'time_gap_mean': valid_gaps.float().mean().item() if valid_gaps.numel() else None,
                'time_gap_std': valid_gaps.float().std(unbiased=False).item() if valid_gaps.numel() else None,
                'time_gap_min': valid_gaps.float().min().item() if valid_gaps.numel() else None,
                'time_gap_max': valid_gaps.float().max().item() if valid_gaps.numel() else None,
                'weight_mean': attn.mean().item(),
                'weight_std': attn.std(unbiased=False).item(),
                'weight_min': attn.min().item(),
                'weight_max': attn.max().item(),
                'gate_mean': temporal_bias.mean().item() if temporal_bias is not None else None,
                'gate_std': temporal_bias.std(unbiased=False).item() if temporal_bias is not None else None,
            }
            if alpha is not None:
                stats.update({
                    'alpha_mean': alpha.mean().item(),
                    'alpha_std': alpha.std(unbiased=False).item(),
                    'alpha_min': alpha.min().item(),
                    'alpha_max': alpha.max().item(),
                })
            self.latest_debug_stats = stats

    def _cross_attend_news(self, price_tokens, news_embeddings, news_time_gaps, news_mask):
        if news_embeddings is None:
            return None

        batch = news_embeddings.shape[0]
        news_h = self.news_proj(news_embeddings)
        market_state = price_tokens.mean(dim=1)
        horizon_offsets = self._horizon_offsets(news_time_gaps.device, news_time_gaps.dtype)
        horizon_features = self.horizon_proj(horizon_offsets.view(self.pred_len, 1))
        horizon_features = horizon_features.unsqueeze(0).expand(batch, -1, -1)

        market_features = market_state.unsqueeze(1).expand(-1, self.pred_len, -1)
        queries = self.query_proj(torch.cat([market_features, horizon_features], dim=-1))
        keys = self.key_proj(news_h)
        values = self.value_proj(news_h)

        q = self._split_heads(queries)
        k = self._split_heads(keys)
        v = self._split_heads(values)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(float(self.head_dim))

        alpha = None
        temporal_bias = None
        if not self.disable_time_decay:
            effective_gaps = news_time_gaps.unsqueeze(1) + horizon_offsets.view(1, self.pred_len, 1)
            scaled_gaps = effective_gaps / self.gap_scale
            if self.use_fixed_time_decay:
                alpha = torch.full(
                    (batch, self.n_heads, self.pred_len, news_h.shape[1]),
                    self.fixed_decay_alpha,
                    device=news_h.device,
                    dtype=news_h.dtype,
                )
            else:
                expanded_news = news_h.unsqueeze(1).expand(-1, self.pred_len, -1, -1)
                expanded_horizon = horizon_offsets.view(1, self.pred_len, 1, 1).expand(
                    batch, -1, news_h.shape[1], -1
                )
                decay_input = torch.cat([expanded_news, expanded_horizon], dim=-1)
                alpha = F.softplus(self.decay_mlp(decay_input)).permute(0, 3, 1, 2) + 1e-6
            temporal_bias = -(alpha * scaled_gaps.unsqueeze(1))
            scores = scores + temporal_bias

        if news_mask is not None:
            scores = scores.masked_fill(news_mask.unsqueeze(1).unsqueeze(1) <= 0, -1e9)

        attn = torch.softmax(scores, dim=-1)
        if news_mask is not None:
            attn = attn * news_mask.unsqueeze(1).unsqueeze(1)
            attn = attn / attn.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        attn = self.attn_dropout(attn)

        context = torch.matmul(attn, v).transpose(1, 2).contiguous()
        context = context.view(batch, self.pred_len, self.d_model)
        context = self.out_proj(context)
        self._debug_stats(news_mask, news_time_gaps, alpha, attn, temporal_bias)
        return context

    def forward(
        self,
        base_out,
        x_enc_norm,
        news_embeddings=None,
        news_time_gaps=None,
        news_mask=None,
        news_novelty=None,
        news_duration_probs=None,
        news_sentiment=None,
    ):
        if news_embeddings is None:
            return base_out

        _, _, n_vars = base_out.shape
        price_tokens = self._price_tokens(x_enc_norm)[:, :n_vars, :]
        news_ctx = self._cross_attend_news(price_tokens, news_embeddings, news_time_gaps, news_mask)
        if news_ctx is None:
            return base_out

        expanded_price = price_tokens.unsqueeze(1).expand(-1, self.pred_len, -1, -1)
        expanded_news = news_ctx.unsqueeze(2).expand(-1, -1, n_vars, -1)
        fused = self.fusion(torch.cat([expanded_price, expanded_news], dim=-1))
        news_delta = self.news_output(fused).squeeze(-1)
        return base_out + news_delta


class Model(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.task_name = configs.task_name
        self.pred_len = configs.pred_len
        self.backbone = Transformer.Model(configs)
        self.news_adapter = DecayBiasedNewsCrossAttention(configs)
        self.latest_debug_stats = None

    def _normalize(self, x_enc):
        means = x_enc.mean(1, keepdim=True).detach()
        x_norm = x_enc - means
        stdev = torch.sqrt(torch.var(x_norm, dim=1, keepdim=True, unbiased=False) + 1e-5)
        return x_norm / stdev

    def forward(
        self,
        x_enc,
        x_mark_enc,
        x_dec,
        x_mark_dec,
        mask=None,
        news_embeddings=None,
        news_time_gaps=None,
        news_mask=None,
        news_novelty=None,
        news_duration_probs=None,
        news_sentiment=None,
    ):
        if self.task_name not in ['long_term_forecast', 'short_term_forecast']:
            return self.backbone(x_enc, x_mark_enc, x_dec, x_mark_dec, mask=mask)

        base_out = self.backbone(x_enc, x_mark_enc, x_dec, x_mark_dec, mask=mask)
        output = self.news_adapter(
            base_out,
            self._normalize(x_enc),
            news_embeddings=news_embeddings,
            news_time_gaps=news_time_gaps,
            news_mask=news_mask,
            news_novelty=news_novelty,
            news_duration_probs=news_duration_probs,
            news_sentiment=news_sentiment,
        )
        self.latest_debug_stats = self.news_adapter.latest_debug_stats
        return output
