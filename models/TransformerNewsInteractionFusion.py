import math

import torch
import torch.nn as nn

from models import Transformer


class NewsInteractionFusionAdapter(nn.Module):
    """FININ-style news interaction baseline adapted to forecasting.

    The original FININ task is next-day market trend classification. This
    adapter keeps the comparable idea, namely news-news interaction followed by
    market/news influence modeling, but uses the local no-future-news tensors
    and outputs a multi-step residual forecast.
    """

    def __init__(self, configs):
        super().__init__()
        self.seq_len = int(configs.seq_len)
        self.pred_len = int(configs.pred_len)
        self.enc_in = int(getattr(configs, 'enc_in', getattr(configs, 'c_out', 1)))
        self.d_model = int(getattr(configs, 'd_model', 64))
        self.n_heads = int(getattr(configs, 'n_heads', 4))
        if self.d_model % self.n_heads != 0:
            raise ValueError(f'd_model={self.d_model} must be divisible by n_heads={self.n_heads}')
        self.head_dim = self.d_model // self.n_heads

        self.news_emb_dim = int(getattr(configs, 'news_emb_dim', 32))
        dropout = float(getattr(configs, 'dropout', 0.0))

        self.price_state_proj = nn.Linear(self.seq_len, self.d_model)
        self.news_proj = nn.Linear(self.news_emb_dim, self.d_model)

        self.news_q = nn.Linear(self.d_model, self.d_model)
        self.news_k = nn.Linear(self.d_model, self.d_model)
        self.news_v = nn.Linear(self.d_model, self.d_model)
        self.news_out = nn.Linear(self.d_model, self.d_model)
        self.news_norm = nn.LayerNorm(self.d_model)

        self.horizon_proj = nn.Sequential(
            nn.Linear(1, self.d_model),
            nn.GELU(),
            nn.Linear(self.d_model, self.d_model),
        )
        self.query_proj = nn.Linear(self.d_model * 2, self.d_model)
        self.cross_k = nn.Linear(self.d_model, self.d_model)
        self.cross_v = nn.Linear(self.d_model, self.d_model)
        self.cross_out = nn.Linear(self.d_model, self.d_model)

        self.attn_dropout = nn.Dropout(dropout)
        self.fusion = nn.Sequential(
            nn.Linear(self.d_model * 2, self.d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.news_output = nn.Linear(self.d_model, 1)
        self.latest_debug_stats = None

    def _split_heads(self, x):
        batch, length, _ = x.shape
        return x.view(batch, length, self.n_heads, self.head_dim).transpose(1, 2)

    def _price_tokens(self, x_enc_norm):
        return self.price_state_proj(x_enc_norm.permute(0, 2, 1))

    def _masked_attention(self, q, k, v, key_mask=None):
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(float(self.head_dim))
        if key_mask is not None:
            scores = scores.masked_fill(key_mask.unsqueeze(1).unsqueeze(1) <= 0, -1e9)
        attn = torch.softmax(scores, dim=-1)
        if key_mask is not None:
            attn = attn * key_mask.unsqueeze(1).unsqueeze(1)
            attn = attn / attn.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        attn = self.attn_dropout(attn)
        context = torch.matmul(attn, v).transpose(1, 2).contiguous()
        return context, attn

    def _interact_news(self, news_embeddings, news_mask):
        news_h = self.news_proj(news_embeddings)
        q = self._split_heads(self.news_q(news_h))
        k = self._split_heads(self.news_k(news_h))
        v = self._split_heads(self.news_v(news_h))
        context, self_attn = self._masked_attention(q, k, v, news_mask)
        context = context.view(news_h.shape[0], news_h.shape[1], self.d_model)
        context = self.news_out(context)
        interacted = self.news_norm(news_h + context)
        if news_mask is not None:
            interacted = interacted * news_mask.unsqueeze(-1)
        return interacted, self_attn

    def _cross_attend_news(self, price_tokens, interacted_news, news_mask):
        batch = interacted_news.shape[0]
        market_state = price_tokens.mean(dim=1)
        horizon_offsets = torch.arange(
            1, self.pred_len + 1, device=interacted_news.device, dtype=interacted_news.dtype
        )
        horizon_features = self.horizon_proj(horizon_offsets.view(self.pred_len, 1))
        horizon_features = horizon_features.unsqueeze(0).expand(batch, -1, -1)
        market_features = market_state.unsqueeze(1).expand(-1, self.pred_len, -1)
        queries = self.query_proj(torch.cat([market_features, horizon_features], dim=-1))

        q = self._split_heads(queries)
        k = self._split_heads(self.cross_k(interacted_news))
        v = self._split_heads(self.cross_v(interacted_news))
        context, cross_attn = self._masked_attention(q, k, v, news_mask)
        context = context.view(batch, self.pred_len, self.d_model)
        return self.cross_out(context), cross_attn

    def _debug_stats(self, news_mask, self_attn, cross_attn):
        with torch.no_grad():
            active = news_mask.sum(dim=-1).float() if news_mask is not None else None
            self.latest_debug_stats = {
                'active_news_mean': active.mean().item() if active is not None else None,
                'self_attn_mean': self_attn.mean().item(),
                'self_attn_std': self_attn.std(unbiased=False).item(),
                'cross_attn_mean': cross_attn.mean().item(),
                'cross_attn_std': cross_attn.std(unbiased=False).item(),
            }

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
        if news_mask is not None and torch.all(news_mask <= 0):
            return base_out

        _, _, n_vars = base_out.shape
        price_tokens = self._price_tokens(x_enc_norm)[:, :n_vars, :]
        interacted_news, self_attn = self._interact_news(news_embeddings, news_mask)
        news_ctx, cross_attn = self._cross_attend_news(price_tokens, interacted_news, news_mask)
        self._debug_stats(news_mask, self_attn, cross_attn)

        expanded_price = price_tokens.unsqueeze(1).expand(-1, self.pred_len, -1, -1)
        expanded_news = news_ctx.unsqueeze(2).expand(-1, -1, n_vars, -1)
        fused = self.fusion(torch.cat([expanded_price, expanded_news], dim=-1))
        news_delta = self.news_output(fused).squeeze(-1)
        return base_out + news_delta


class Model(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.task_name = configs.task_name
        self.backbone = Transformer.Model(configs)
        self.news_adapter = NewsInteractionFusionAdapter(configs)
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
