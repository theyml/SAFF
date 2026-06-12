import re

import torch
import torch.nn as nn
import torch.nn.functional as F


class DecayAwareResidualAdapter(nn.Module):
    """Backbone-agnostic decay-aware residual news adapter.

    The wrapped price model produces a normal forecast. This module consumes the
    same input history plus aligned news tensors and adds a horizon-specific news
    residual. It intentionally keeps the adapter small so it can be attached to
    linear, patch, convolutional, or Transformer backbones without changing them.
    """

    def __init__(self, configs):
        super().__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.freq = str(getattr(configs, 'freq', 'd'))
        self.enc_in = int(getattr(configs, 'enc_in', getattr(configs, 'c_out', 1)))
        self.d_model = int(getattr(configs, 'd_model', 512))
        self.news_emb_dim = int(getattr(configs, 'news_emb_dim', 32))
        self.news_decay_hidden_dim = int(getattr(configs, 'news_decay_hidden_dim', 64))
        self.news_residual_scale = float(getattr(configs, 'news_residual_scale', 1.0))
        self.disable_time_decay = bool(getattr(configs, 'disable_time_decay', False))
        self.use_news_selector = bool(getattr(configs, 'use_news_selector', False))
        self.use_market_state_decay = bool(getattr(configs, 'use_market_state_decay', False))
        self.use_novelty = bool(getattr(configs, 'use_novelty', False))
        self.use_sentiment = bool(getattr(configs, 'use_sentiment', False))
        self.use_signed_impact = bool(getattr(configs, 'use_signed_impact', False))
        self.use_signed_decay_kernel = bool(getattr(configs, 'use_signed_decay_kernel', False))

        self.price_state_proj = nn.Linear(self.seq_len, self.d_model)
        self.news_proj = nn.Linear(self.news_emb_dim, self.d_model)

        decay_in_dim = self.d_model + 1
        if self.use_market_state_decay:
            decay_in_dim += self.d_model
        if self.use_novelty:
            decay_in_dim += 1
        if self.use_sentiment:
            decay_in_dim += 1
        self.decay_mlp = nn.Sequential(
            nn.Linear(decay_in_dim, self.news_decay_hidden_dim),
            nn.GELU(),
            nn.Linear(self.news_decay_hidden_dim, 1),
        )

        if self.use_news_selector:
            self.selector_mlp = nn.Sequential(
                nn.Linear(self.d_model * 2, self.news_decay_hidden_dim),
                nn.GELU(),
                nn.Linear(self.news_decay_hidden_dim, 1),
            )
        else:
            self.selector_mlp = None

        if self.use_signed_impact or self.use_signed_decay_kernel:
            signed_in_dim = self.d_model + 1
            if self.use_sentiment:
                signed_in_dim += 1
            self.signed_impact_mlp = nn.Sequential(
                nn.Linear(signed_in_dim, self.news_decay_hidden_dim),
                nn.GELU(),
                nn.Linear(self.news_decay_hidden_dim, 1),
            )
        else:
            self.signed_impact_mlp = None

        fusion_in_dim = self.d_model * 2 + (1 if self.use_sentiment else 0)
        self.fusion = nn.Sequential(
            nn.Linear(fusion_in_dim, self.d_model),
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
        offsets = torch.arange(1, self.pred_len + 1, device=device, dtype=dtype)
        return offsets * self._step_in_days()

    def _price_tokens(self, x_enc):
        # x_enc is already normalized by the wrapper: [B, seq_len, n_vars].
        return self.price_state_proj(x_enc.permute(0, 2, 1))

    def _aggregate_news(self, price_tokens, news_embeddings, news_time_gaps, news_mask, news_novelty=None, news_sentiment=None):
        if news_embeddings is None:
            return None, None

        news_h = self.news_proj(news_embeddings)
        market_state = price_tokens.mean(dim=1)
        horizon_offsets = self._horizon_offsets(news_time_gaps.device, news_time_gaps.dtype)

        expanded_news = news_h.unsqueeze(1).expand(-1, self.pred_len, -1, -1)
        horizon_feature = horizon_offsets.view(1, self.pred_len, 1, 1).expand(
            news_h.shape[0], -1, news_h.shape[1], -1
        )
        decay_parts = [expanded_news]
        if self.use_market_state_decay:
            state = market_state.unsqueeze(1).unsqueeze(2).expand(-1, self.pred_len, news_h.shape[1], -1)
            decay_parts.append(state)
        if self.use_novelty:
            if news_novelty is None:
                news_novelty = torch.zeros_like(news_time_gaps).unsqueeze(-1)
            decay_parts.append(news_novelty.unsqueeze(1).expand(-1, self.pred_len, -1, -1))
        if self.use_sentiment:
            if news_sentiment is None:
                news_sentiment = torch.zeros_like(news_time_gaps).unsqueeze(-1)
            decay_parts.append(news_sentiment.unsqueeze(1).expand(-1, self.pred_len, -1, -1))
        decay_parts.append(horizon_feature)

        decay_input = torch.cat(decay_parts, dim=-1)
        alpha = F.softplus(self.decay_mlp(decay_input)).squeeze(-1) + 1e-6
        effective_gaps = news_time_gaps.unsqueeze(1) + horizon_offsets.view(1, self.pred_len, 1)
        logits = torch.zeros_like(effective_gaps) if self.disable_time_decay else -(alpha * effective_gaps)

        selector_gate = None
        if self.selector_mlp is not None:
            selector_state = market_state.unsqueeze(1).unsqueeze(2).expand(-1, self.pred_len, news_h.shape[1], -1)
            selector_gate = torch.sigmoid(
                self.selector_mlp(torch.cat([expanded_news, selector_state], dim=-1))
            ).squeeze(-1)
            logits = logits + torch.log(selector_gate + 1e-6)

        if news_mask is not None:
            logits = logits.masked_fill(news_mask.unsqueeze(1) <= 0, -1e9)

        gates = torch.sigmoid(logits)
        if news_mask is not None:
            gates = gates * news_mask.unsqueeze(1)

        signed_gate = None
        if self.signed_impact_mlp is not None:
            signed_parts = [expanded_news, horizon_feature]
            if self.use_sentiment:
                if news_sentiment is None:
                    news_sentiment = torch.zeros_like(news_time_gaps).unsqueeze(-1)
                signed_parts.append(news_sentiment.unsqueeze(1).expand(-1, self.pred_len, -1, -1))
            signed_input = torch.cat(signed_parts, dim=-1)
            signed_gate = torch.tanh(self.signed_impact_mlp(signed_input))

        if self.use_signed_decay_kernel:
            signed_kernel = gates.unsqueeze(-1)
            if signed_gate is not None:
                signed_kernel = signed_kernel * signed_gate
            active_count = news_mask.sum(dim=-1).clamp_min(1.0) if news_mask is not None else torch.full(
                (news_h.shape[0],), news_h.shape[1], device=news_h.device, dtype=news_h.dtype
            )
            scale = torch.sqrt(active_count).view(-1, 1, 1).clamp_min(1.0)
            news_ctx = torch.einsum('bhnd,bnd->bhd', signed_kernel, news_h) / scale
        else:
            weights = gates / gates.sum(dim=-1, keepdim=True).clamp_min(1e-6)
            weighted_news = news_h
            if self.use_signed_impact:
                if signed_gate is None:
                    signed_gate = torch.ones_like(expanded_news[..., :1])
                weighted_news = expanded_news * signed_gate
                news_ctx = torch.einsum('bhn,bhnd->bhd', weights, weighted_news)
            else:
                news_ctx = torch.einsum('bhn,bnd->bhd', weights, weighted_news)
        if self.use_signed_decay_kernel:
            weights_for_sentiment = None
        else:
            weights_for_sentiment = weights
        sentiment_ctx = None
        if self.use_sentiment:
            if news_sentiment is None:
                news_sentiment = torch.zeros_like(news_time_gaps).unsqueeze(-1)
            if weights_for_sentiment is None:
                sentiment_ctx = torch.einsum('bhnd,bnc->bhc', signed_kernel, news_sentiment) / scale
            else:
                sentiment_ctx = torch.einsum('bhn,bnc->bhc', weights_for_sentiment, news_sentiment)
        return news_ctx, sentiment_ctx

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
        news_ctx, sentiment_ctx = self._aggregate_news(
            price_tokens,
            news_embeddings,
            news_time_gaps,
            news_mask,
            news_novelty,
            news_sentiment,
        )
        if news_ctx is None:
            return base_out

        expanded_price = price_tokens.unsqueeze(1).expand(-1, self.pred_len, -1, -1)
        expanded_news = news_ctx.unsqueeze(2).expand(-1, -1, n_vars, -1)
        fusion_parts = [expanded_price, expanded_news]
        if self.use_sentiment:
            if sentiment_ctx is None:
                sentiment_ctx = torch.zeros(
                    news_ctx.shape[0], news_ctx.shape[1], 1,
                    device=news_ctx.device, dtype=news_ctx.dtype,
                )
            expanded_sentiment = sentiment_ctx.unsqueeze(2).expand(-1, -1, n_vars, -1)
            fusion_parts.append(expanded_sentiment)
        fused = self.fusion(torch.cat(fusion_parts, dim=-1))
        news_delta = self.news_output(fused).squeeze(-1)

        if self.disable_time_decay:
            horizon_mask = torch.zeros(self.pred_len, device=news_delta.device, dtype=news_delta.dtype)
            horizon_mask[0] = 1.0
            news_delta = news_delta * horizon_mask.view(1, self.pred_len, 1)

        return base_out + self.news_residual_scale * news_delta
