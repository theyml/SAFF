"""
Incremental Phase-1 news-aware model.

This file intentionally leaves the original `models/iTransformer.py` unchanged.
The new model reuses the same price-side iTransformer stack and adds a small,
text-conditioned time-decay aggregation branch for precomputed news embeddings.
"""

import re

import torch
import torch.nn as nn
import torch.nn.functional as F

from layers.Transformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import DataEmbedding_inverted


class Model(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.output_attention = configs.output_attention
        self.freq = str(getattr(configs, 'freq', 'd'))
        # Phase-1 addition relative to `models/iTransformer.py`. #++
        # Extra config for the news-decay branch. #++
        self.use_novelty = bool(getattr(configs, 'use_novelty', False))  #++
        self.disable_time_decay = bool(getattr(configs, 'disable_time_decay', False))  #++
        self.use_market_state_decay = bool(getattr(configs, 'use_market_state_decay', False))  #++
        self.use_news_selector = bool(getattr(configs, 'use_news_selector', False))  #++
        self.use_channel_specific_news = bool(getattr(configs, 'use_channel_specific_news', False))  #++
        self.use_novelty_persistence = bool(getattr(configs, 'use_novelty_persistence', False))  #++
        self.use_duration_persistence = bool(getattr(configs, 'use_duration_persistence', False))  #++
        self.duration_confidence_threshold = float(getattr(configs, 'duration_confidence_threshold', 0.55))  #++
        self.duration_margin_threshold = float(getattr(configs, 'duration_margin_threshold', 0.15))  #++
        self.debug_news_stats = bool(getattr(configs, 'debug_news_stats', False))  #++
        self.news_emb_dim = int(getattr(configs, 'news_emb_dim', 32))  #++
        self.news_decay_hidden_dim = int(getattr(configs, 'news_decay_hidden_dim', 64))  #++
        self.latest_debug_stats = None  #++

        # Price-side iTransformer components, kept aligned with the original implementation.
        self.enc_embedding = DataEmbedding_inverted(
            configs.seq_len, configs.d_model, configs.embed, configs.freq, configs.dropout
        )
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, configs.factor,
                                      attention_dropout=configs.dropout,
                                      output_attention=configs.output_attention),
                        configs.d_model, configs.n_heads
                    ),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation,
                )
                for _ in range(configs.e_layers)
            ],
            norm_layer=nn.LayerNorm(configs.d_model),
        )

        self.projection = nn.Linear(configs.d_model, configs.pred_len, bias=True)

        # Phase-1 addition relative to `models/iTransformer.py`. #++
        # News branch for precomputed news embeddings. #++
        self.news_proj = nn.Linear(self.news_emb_dim, configs.d_model)  #++
        self.decay_state_dim = configs.d_model if (self.use_market_state_decay or self.use_channel_specific_news) else 0  #++
        self.selector_state_dim = configs.d_model if (self.use_news_selector or self.use_channel_specific_news) else 0  #++
        decay_in_dim = configs.d_model + self.decay_state_dim + (1 if self.use_novelty else 0) + 1  #++
        self.decay_mlp = nn.Sequential(  #++
            nn.Linear(decay_in_dim, self.news_decay_hidden_dim),
            nn.GELU(),
            nn.Linear(self.news_decay_hidden_dim, 1),
        )
        selector_in_dim = configs.d_model + self.selector_state_dim
        if self.use_news_selector and selector_in_dim > configs.d_model:  #++
            self.selector_mlp = nn.Sequential(  #++
                nn.Linear(selector_in_dim, self.news_decay_hidden_dim),
                nn.GELU(),
                nn.Linear(self.news_decay_hidden_dim, 1),
            )
        else:
            self.selector_mlp = None
        if self.use_novelty and self.use_novelty_persistence:  #++
            self.persistence_mlp = nn.Sequential(  #++
                nn.Linear(decay_in_dim, self.news_decay_hidden_dim),
                nn.GELU(),
                nn.Linear(self.news_decay_hidden_dim, 1),
            )
        else:
            self.persistence_mlp = None
        if self.use_duration_persistence:
            self.duration_head = nn.Sequential(
                nn.Linear(configs.d_model, self.news_decay_hidden_dim),
                nn.GELU(),
                nn.Linear(self.news_decay_hidden_dim, 3),
            )
            duration_values = self._parse_duration_persistence_values(
                getattr(configs, 'duration_persistence_values', '0.75,1.25,1.0')
            )
            self.register_buffer('duration_persistence_values', duration_values)
        else:
            self.duration_head = None

        self.fusion = nn.Sequential(  #++
            nn.Linear(configs.d_model * 2, configs.d_model),
            nn.GELU(),
            nn.Dropout(configs.dropout),
        )
        self.news_output = nn.Linear(configs.d_model, 1, bias=True)  #++

    @staticmethod
    def _parse_duration_persistence_values(values):
        if isinstance(values, str):
            parsed = [float(v.strip()) for v in values.split(',') if v.strip()]
        else:
            parsed = [float(v) for v in values]
        if len(parsed) != 3:
            raise ValueError(
                'duration_persistence_values must contain exactly 3 values: short,long,unsure'
            )
        return torch.tensor(parsed, dtype=torch.float32)

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
        if freq in base_steps:
            return base_steps[freq]

        match = re.fullmatch(r'(\d+)([a-zA-Z]+)', freq)
        if match:
            mult = float(match.group(1))
            unit = match.group(2).lower()
            if unit in base_steps:
                return mult * base_steps[unit]

        return 1.0

    def _horizon_offsets(self, device, dtype):
        step_in_days = self._step_in_days()
        offsets = torch.arange(1, self.pred_len + 1, device=device, dtype=dtype)
        return offsets * step_in_days

    @staticmethod
    def _safe_item(value):
        if value is None:
            return None
        return float(value.detach().cpu().item())

    def _masked_stats(self, name, tensor, mask=None):
        if tensor is None:
            return {}

        values = tensor.detach()
        if mask is not None:
            mask = mask.detach().to(dtype=torch.bool, device=values.device)
            while mask.dim() < values.dim():
                mask = mask.unsqueeze(1)
            mask = torch.broadcast_to(mask, values.shape)
            values = values[mask]
        else:
            values = values.reshape(-1)

        if values.numel() == 0:
            return {
                f'{name}_count': 0,
                f'{name}_mean': None,
                f'{name}_std': None,
                f'{name}_min': None,
                f'{name}_max': None,
            }

        values = values.float()
        return {
            f'{name}_count': int(values.numel()),
            f'{name}_mean': self._safe_item(values.mean()),
            f'{name}_std': self._safe_item(values.std(unbiased=False)),
            f'{name}_min': self._safe_item(values.min()),
            f'{name}_max': self._safe_item(values.max()),
        }

    def _capture_debug_stats(
        self,
        news_mask,
        news_time_gaps,
        alpha,
        gates,
        weights,
        news_novelty=None,
        selector_gate=None,
        persistence_bonus=None,
        duration_assign=None,
    ):
        if not self.debug_news_stats:
            self.latest_debug_stats = None
            return

        base_mask = None if news_mask is None else news_mask > 0
        stats = {
            'use_market_state_decay': bool(self.use_market_state_decay),
            'disable_time_decay': bool(self.disable_time_decay),
            'use_news_selector': bool(self.use_news_selector),
            'use_channel_specific_news': bool(self.use_channel_specific_news),
            'use_novelty': bool(self.use_novelty),
            'use_novelty_persistence': bool(self.use_novelty_persistence),
            'use_duration_persistence': bool(self.use_duration_persistence),
            'active_news_mean': self._safe_item(news_mask.sum(dim=-1).float().mean()) if news_mask is not None else None,
        }
        stats.update(self._masked_stats('time_gap', news_time_gaps, base_mask))
        stats.update(self._masked_stats('alpha', alpha, base_mask))
        stats.update(self._masked_stats('gate', gates, base_mask))
        stats.update(self._masked_stats('weight', weights, base_mask))

        if selector_gate is not None:
            stats.update(self._masked_stats('selector_gate', selector_gate, base_mask))

        if news_novelty is not None and self.use_novelty:
            novelty_values = news_novelty.squeeze(-1) if news_novelty.dim() == news_time_gaps.dim() + 1 else news_novelty
            stats.update(self._masked_stats('novelty', novelty_values, base_mask))

        if persistence_bonus is not None:
            stats.update(self._masked_stats('persistence_bonus', persistence_bonus, base_mask))

        if duration_assign is not None:
            duration_assign = duration_assign.detach().float()
            reduce_dims = tuple(range(duration_assign.dim() - 1))
            if news_mask is not None:
                duration_mask = (news_mask > 0).detach().float().unsqueeze(-1)
                denom = duration_mask.sum(dim=reduce_dims).clamp_min(1.0)
                duration_mean = (duration_assign * duration_mask).sum(dim=reduce_dims) / denom
            else:
                duration_mean = duration_assign.mean(dim=reduce_dims)
            labels = ('short', 'long', 'unsure')
            for idx, label in enumerate(labels):
                stats[f'duration_{label}_share'] = self._safe_item(duration_mean[idx])

        self.latest_debug_stats = stats

    def _prepare_state_tensor(self, price_tokens, news_h, state_dim):
        if price_tokens is None or state_dim == 0:
            return None

        if self.use_channel_specific_news:
            channel_state = price_tokens.unsqueeze(1).unsqueeze(3).expand(
                -1, self.pred_len, -1, news_h.shape[1], -1
            )
            return channel_state

        market_state = price_tokens.mean(dim=1)
        market_state = market_state.unsqueeze(1).unsqueeze(2).expand(
            -1, self.pred_len, news_h.shape[1], -1
        )
        return market_state

    def _apply_selector(self, logits, news_h, state_tensor, news_mask):
        if self.selector_mlp is None or state_tensor is None:
            return logits, None

        if self.use_channel_specific_news:
            expanded_news = news_h.unsqueeze(1).unsqueeze(1).expand(
                -1, self.pred_len, state_tensor.shape[2], -1, -1
            )
            selector_input = torch.cat([expanded_news, state_tensor], dim=-1)
        else:
            expanded_news = news_h.unsqueeze(1).expand(-1, self.pred_len, -1, -1)
            selector_input = torch.cat([expanded_news, state_tensor], dim=-1)

        selector_gate = torch.sigmoid(self.selector_mlp(selector_input)).squeeze(-1)
        if news_mask is not None:
            selector_mask = news_mask.unsqueeze(1)
            if self.use_channel_specific_news:
                selector_mask = selector_mask.unsqueeze(1)
            selector_gate = selector_gate * selector_mask
        return logits + torch.log(selector_gate + 1e-6), selector_gate

    def _apply_novelty_persistence(self, alpha, logits, decay_input, news_novelty):
        if self.persistence_mlp is None or news_novelty is None:
            return alpha, logits, None

        persistence_gate = torch.sigmoid(self.persistence_mlp(decay_input)).squeeze(-1)
        if self.use_channel_specific_news:
            novelty_term = news_novelty.squeeze(-1).unsqueeze(1).unsqueeze(1).expand_as(alpha)
        else:
            novelty_term = news_novelty.squeeze(-1).unsqueeze(1).expand_as(alpha)
        persistence_bonus = persistence_gate * novelty_term
        alpha = alpha / (1.0 + persistence_bonus.clamp_min(0.0))
        logits = logits + torch.log1p(persistence_bonus.clamp_min(0.0))
        return alpha, logits, persistence_bonus

    def _apply_duration_persistence(self, alpha, news_h, news_duration_probs=None):
        if not self.use_duration_persistence:
            return alpha, None

        if news_duration_probs is not None:
            duration_prob = news_duration_probs.to(device=news_h.device, dtype=news_h.dtype).clamp_min(0.0)
            duration_sum = duration_prob.sum(dim=-1, keepdim=True)
            duration_default = torch.zeros_like(duration_prob)
            duration_default[..., 2] = 1.0
            duration_prob = torch.where(
                duration_sum > 1e-6,
                duration_prob / duration_sum.clamp_min(1e-6),
                duration_default,
            )
            duration_assign = duration_prob
        elif self.duration_head is not None:
            duration_logits = self.duration_head(news_h)
            duration_prob = torch.softmax(duration_logits, dim=-1)
        else:
            return alpha, None

        short_long_prob = duration_prob[..., :2]
        best_short_long, best_short_long_idx = short_long_prob.max(dim=-1, keepdim=True)
        short_long_margin = (short_long_prob[..., 0] - short_long_prob[..., 1]).abs().unsqueeze(-1)
        confident_mask = (
            (best_short_long >= self.duration_confidence_threshold)
            & (short_long_margin >= self.duration_margin_threshold)
        )

        duration_hard = torch.zeros_like(duration_prob)
        duration_hard[..., 2] = 1.0
        short_long_hard = torch.zeros_like(duration_prob)
        short_long_hard.scatter_(-1, best_short_long_idx, 1.0)
        duration_hard = torch.where(confident_mask.expand_as(duration_hard), short_long_hard, duration_hard)

        if news_duration_probs is not None:
            duration_assign = duration_hard
        else:
            # Forward uses confident hard routing, backward keeps the soft gradients.
            duration_assign = duration_hard - duration_prob.detach() + duration_prob

        persistence = (duration_assign * self.duration_persistence_values.view(1, 1, 3)).sum(dim=-1)
        if self.use_channel_specific_news:
            persistence = persistence.unsqueeze(1).unsqueeze(1).expand_as(alpha)
        else:
            persistence = persistence.unsqueeze(1).expand_as(alpha)
        return alpha / persistence.clamp_min(1e-4), duration_assign

    def _aggregate_news(
        self,
        price_tokens,
        news_embeddings,
        news_time_gaps,
        news_mask,
        news_novelty=None,
        news_duration_probs=None,
        news_sentiment=None,
    ):
        # 2. #计算decay：先把新闻向量映射到 d_model，再根据新闻内容和 horizon
        # 预测 alpha_{i,h}。 #++
        # 然后按 exp(-alpha_{i,h} * delta_t_h) 得到逐 horizon 的新闻权重，并聚合成
        # [B, pred_len, d_model] 的 horizon-specific news contexts。 #++
        if news_embeddings is None:
            return None

        news_h = self.news_proj(news_embeddings)  #++
        horizon_offsets = self._horizon_offsets(news_time_gaps.device, news_time_gaps.dtype)  #++
        decay_state_tensor = self._prepare_state_tensor(price_tokens, news_h, self.decay_state_dim)  #++
        selector_state_tensor = self._prepare_state_tensor(price_tokens, news_h, self.selector_state_dim)  #++

        if self.use_channel_specific_news:
            n_vars = price_tokens.shape[1]
            expanded_news_h = news_h.unsqueeze(1).unsqueeze(1).expand(-1, self.pred_len, n_vars, -1, -1)  #++
            horizon_feature = horizon_offsets.view(1, self.pred_len, 1, 1, 1).expand(
                news_h.shape[0], -1, n_vars, news_h.shape[1], -1
            )  #++
            decay_parts = [expanded_news_h]
            if decay_state_tensor is not None:
                decay_parts.append(decay_state_tensor)
            if self.use_novelty:
                if news_novelty is None:
                    news_novelty = torch.zeros_like(news_time_gaps).unsqueeze(-1)  #++
                expanded_novelty = news_novelty.unsqueeze(1).unsqueeze(1).expand(-1, self.pred_len, n_vars, -1, -1)  #++
                decay_parts.append(expanded_novelty)
            decay_parts.append(horizon_feature)
            decay_input = torch.cat(decay_parts, dim=-1)  #++

            alpha = F.softplus(self.decay_mlp(decay_input)).squeeze(-1) + 1e-6  #++
            alpha, duration_assign = self._apply_duration_persistence(alpha, news_h, news_duration_probs=news_duration_probs)  #++
            effective_gaps = news_time_gaps.unsqueeze(1).unsqueeze(1) + horizon_offsets.view(1, self.pred_len, 1, 1)  #++
            if self.disable_time_decay:
                logits = torch.zeros_like(effective_gaps)  #++
            else:
                logits = -(alpha * effective_gaps)  #++
            logits, selector_gate = self._apply_selector(logits, news_h, selector_state_tensor, news_mask)  #++
            alpha, logits, persistence_bonus = self._apply_novelty_persistence(alpha, logits, decay_input, news_novelty)  #++

            if news_mask is not None:
                logits = logits.masked_fill(news_mask.unsqueeze(1).unsqueeze(1) <= 0, -1e9)  #++

            gates = torch.sigmoid(logits)  #++
            if news_mask is not None:
                gates = gates * news_mask.unsqueeze(1).unsqueeze(1)  #++
            weights = gates / gates.sum(dim=-1, keepdim=True).clamp_min(1e-6)  #++
            news_ctx = torch.einsum('bhvn,bnd->bhvd', weights, news_h)  #++
            self._capture_debug_stats(
                news_mask=news_mask,
                news_time_gaps=news_time_gaps,
                alpha=alpha,
                gates=gates,
                weights=weights,
                news_novelty=news_novelty,
                selector_gate=selector_gate,
                persistence_bonus=persistence_bonus,
                duration_assign=duration_assign,
            )
        else:
            horizon_feature = horizon_offsets.view(1, self.pred_len, 1, 1).expand(
                news_h.shape[0], -1, news_h.shape[1], -1
            )  #++
            expanded_news_h = news_h.unsqueeze(1).expand(-1, self.pred_len, -1, -1)  #++
            decay_parts = [expanded_news_h]
            if decay_state_tensor is not None:
                decay_parts.append(decay_state_tensor)
            if self.use_novelty:
                if news_novelty is None:
                    news_novelty = torch.zeros_like(news_time_gaps).unsqueeze(-1)  #++
                expanded_novelty = news_novelty.unsqueeze(1).expand(-1, self.pred_len, -1, -1)  #++
                decay_parts.append(expanded_novelty)
            decay_parts.append(horizon_feature)
            decay_input = torch.cat(decay_parts, dim=-1)  #++

            alpha = F.softplus(self.decay_mlp(decay_input)).squeeze(-1) + 1e-6  #++
            alpha, duration_assign = self._apply_duration_persistence(alpha, news_h, news_duration_probs=news_duration_probs)  #++
            effective_gaps = news_time_gaps.unsqueeze(1) + horizon_offsets.view(1, self.pred_len, 1)  #++
            if self.disable_time_decay:
                logits = torch.zeros_like(effective_gaps)  #++
            else:
                logits = -(alpha * effective_gaps)  #++
            logits, selector_gate = self._apply_selector(logits, news_h, selector_state_tensor, news_mask)  #++
            alpha, logits, persistence_bonus = self._apply_novelty_persistence(alpha, logits, decay_input, news_novelty)  #++

            if news_mask is not None:
                logits = logits.masked_fill(news_mask.unsqueeze(1) <= 0, -1e9)  #++

            gates = torch.sigmoid(logits)  #++
            if news_mask is not None:
                gates = gates * news_mask.unsqueeze(1)  #++
            weights = gates / gates.sum(dim=-1, keepdim=True).clamp_min(1e-6)  #++
            news_ctx = torch.einsum('bhn,bnd->bhd', weights, news_h)  #++
            self._capture_debug_stats(
                news_mask=news_mask,
                news_time_gaps=news_time_gaps,
                alpha=alpha,
                gates=gates,
                weights=weights,
                news_novelty=news_novelty,
                selector_gate=selector_gate,
                persistence_bonus=persistence_bonus,
                duration_assign=duration_assign,
            )

        return news_ctx

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec,
                 news_embeddings=None, news_time_gaps=None, news_mask=None, news_novelty=None,
                 news_duration_probs=None, news_sentiment=None):
        # The normalization / price encoder / output projection path below is copied
        # from the original iTransformer forecast path and kept behavior-compatible.
        if self.debug_news_stats:
            self.latest_debug_stats = None

        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc = x_enc / stdev

        _, _, n_vars = x_enc.shape

        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)

        price_tokens = enc_out[:, :n_vars, :]
        base_out = self.projection(price_tokens).permute(0, 2, 1)[:, :, :n_vars]

        # 1. #融合时间序列和新闻：先算逐 horizon 的 news contexts，再产出
        # horizon-specific residual forecast。 #++
        news_ctx = self._aggregate_news(
            price_tokens,
            news_embeddings,
            news_time_gaps,
            news_mask,
            news_novelty,
            news_duration_probs,
            news_sentiment,
        )  #++
        if news_ctx is not None:
            expanded_price = price_tokens.unsqueeze(1).expand(-1, self.pred_len, -1, -1)  #++
            if self.use_channel_specific_news:
                expanded_news = news_ctx
            else:
                expanded_news = news_ctx.unsqueeze(2).expand(-1, -1, n_vars, -1)  #++
            fused = self.fusion(torch.cat([expanded_price, expanded_news], dim=-1))  #++
            news_delta = self.news_output(fused).squeeze(-1)  #++
            if self.disable_time_decay:
                # "No decay" ablation means news only adjusts the immediate next step.
                horizon_mask = torch.zeros(self.pred_len, device=news_delta.device, dtype=news_delta.dtype)
                horizon_mask[0] = 1.0
                news_delta = news_delta * horizon_mask.view(1, self.pred_len, 1)
            dec_out = base_out + news_delta
        else:
            dec_out = base_out

        dec_out = dec_out * stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1)
        dec_out = dec_out + means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1)
        return dec_out

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None,
                news_embeddings=None, news_time_gaps=None, news_mask=None, news_novelty=None,
                news_duration_probs=None, news_sentiment=None):
        # Forward keeps the original forecast contract, but accepts extra news tensors. #++
        if self.task_name in ['long_term_forecast', 'short_term_forecast']:
            dec_out = self.forecast(
                x_enc, x_mark_enc, x_dec, x_mark_dec,
                news_embeddings=news_embeddings,  #++
                news_time_gaps=news_time_gaps,  #++
                news_mask=news_mask,  #++
                news_novelty=news_novelty,  #++
                news_duration_probs=news_duration_probs,  #++
                news_sentiment=news_sentiment,  #++
            )
            return dec_out[:, -self.pred_len:, :]
        return None
