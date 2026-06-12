"""
News-aware dataset additions for Phase 1.

Design constraints:
- keep the original Dataset_Custom untouched
- add an incremental dataset that reuses the same price/K-line slicing logic
- return extra news tensors only when `--use_news` is enabled through `decay_run.py`

Phase 1 assumes news embeddings are already available as numeric columns
in a news CSV. If no embedding columns are found, the dataset returns
zero vectors so the pipeline remains runnable.
"""

import os
import re
import numpy as np
import pandas as pd

from data_provider.data_loader import Dataset_Custom


class Dataset_FinNews(Dataset_Custom):
    def __init__(self, args, flag='train', time_col='Date'):
        self.news_path = getattr(args, 'news_path', None)
        self.news_time_col = getattr(args, 'news_time_col', 'Date')
        self.news_id_col = getattr(args, 'news_id_col', None)
        self.news_embed_prefix = getattr(args, 'news_embed_prefix', 'news_emb_')
        self.news_emb_dim = int(getattr(args, 'news_emb_dim', 32))
        self.news_max_items = int(getattr(args, 'news_max_items', 32))
        self.news_lookback_days = int(getattr(args, 'news_lookback_days', 30))
        self.news_selection_mode = getattr(args, 'news_selection_mode', 'latest')
        self.news_rank_col = getattr(args, 'news_rank_col', None)
        self.news_rank_recency_weight = float(getattr(args, 'news_rank_recency_weight', 0.15))
        self.news_rank_fill_value = float(getattr(args, 'news_rank_fill_value', 0.0))
        # For daily K-line data, align news to the market sample date instead of
        # comparing raw timestamps like 00:00 vs 22:00 on the same calendar day.
        self.news_align_mode = getattr(args, 'news_align_mode', None) or ('date' if args.freq == 'd' else 'timestamp')
        self.use_novelty = bool(getattr(args, 'use_novelty', False))
        self.use_sentiment = bool(getattr(args, 'use_sentiment', False))
        self.use_duration_persistence = bool(getattr(args, 'use_duration_persistence', False))
        self.news_novelty_col = getattr(args, 'news_novelty_col', None)
        self.news_sentiment_col = getattr(args, 'news_sentiment_col', None)
        self.news_sentiment_title_col = getattr(args, 'news_sentiment_title_col', 'Article_title')
        self.news_duration_prefix = getattr(args, 'news_duration_prefix', 'duration_')
        self.news_duration_short_col = getattr(args, 'news_duration_short_col', None) or f'{self.news_duration_prefix}short_prob'
        self.news_duration_long_col = getattr(args, 'news_duration_long_col', None) or f'{self.news_duration_prefix}long_prob'
        self.news_duration_unsure_col = getattr(args, 'news_duration_unsure_col', None) or f'{self.news_duration_prefix}unsure_prob'
        self.news_df = None
        self.news_time_ns = None
        self.news_embeddings = None
        self.news_rank_values = None
        self.news_novelty = None
        self.news_sentiment = None
        self.news_duration = None
        super().__init__(args, flag=flag, time_col=time_col)
        self._load_news_table()

    @staticmethod
    def _lexicon_sentiment(text):
        tokens = set(re.findall(r"[a-z]+", str(text).lower()))
        positive = (
            'beat', 'beats', 'beating', 'raise', 'raises', 'raised', 'upgrade', 'upgraded',
            'outperform', 'bullish', 'gain', 'gains', 'surge', 'surges', 'jump', 'jumps',
            'rally', 'rallies', 'growth', 'profit', 'profits', 'record', 'strong',
            'positive', 'buy', 'tops', 'top', 'higher', 'boost', 'boosts', 'expands',
        )
        negative = (
            'miss', 'misses', 'missed', 'cut', 'cuts', 'downgrade', 'downgraded',
            'underperform', 'bearish', 'loss', 'losses', 'fall', 'falls', 'drop',
            'drops', 'plunge', 'plunges', 'slump', 'slumps', 'weak', 'negative',
            'sell', 'lower', 'lawsuit', 'probe', 'investigation', 'warning',
            'warns', 'risk', 'risks', 'decline', 'declines', 'layoff', 'layoffs',
        )
        pos = sum(1 for word in positive if word in tokens)
        neg = sum(1 for word in negative if word in tokens)
        if pos == 0 and neg == 0:
            return 0.0
        return float((pos - neg) / max(pos + neg, 1))

    def _resolve_news_path(self):
        if not self.news_path:
            return None
        if os.path.isabs(self.news_path):
            return self.news_path
        return os.path.join(self.root_path, self.news_path)

    def _load_news_table(self):
        path = self._resolve_news_path()
        if not path or not os.path.exists(path):
            return

        news_df = pd.read_csv(path)
        if self.news_time_col not in news_df.columns:
            raise ValueError(f'news_time_col={self.news_time_col!r} not found in {path}')

        news_df[self.news_time_col] = pd.to_datetime(news_df[self.news_time_col], errors='coerce')
        news_df = news_df.dropna(subset=[self.news_time_col]).sort_values(self.news_time_col).reset_index(drop=True)
        if self.news_align_mode == 'date':
            news_df['_align_time'] = news_df[self.news_time_col].dt.normalize()
        else:
            news_df['_align_time'] = news_df[self.news_time_col]

        embed_cols = [c for c in news_df.columns if c.startswith(self.news_embed_prefix)]
        if embed_cols:
            raw_emb = news_df[embed_cols].apply(pd.to_numeric, errors='coerce').fillna(0.0).to_numpy(dtype=np.float32)
        else:
            raw_emb = np.zeros((len(news_df), 0), dtype=np.float32)

        # Keep the model-facing embedding width fixed so the model can be built
        # before the dataset is instantiated.
        emb = np.zeros((len(news_df), self.news_emb_dim), dtype=np.float32)
        if raw_emb.shape[1] > 0:
            copy_dim = min(raw_emb.shape[1], self.news_emb_dim)
            emb[:, :copy_dim] = raw_emb[:, :copy_dim]

        novelty = np.zeros((len(news_df), 1), dtype=np.float32)
        if self.use_novelty and self.news_novelty_col and self.news_novelty_col in news_df.columns:
            novelty[:, 0] = pd.to_numeric(news_df[self.news_novelty_col], errors='coerce').fillna(0.0).to_numpy(dtype=np.float32)

        sentiment = np.zeros((len(news_df), 1), dtype=np.float32)
        if self.use_sentiment:
            if self.news_sentiment_col and self.news_sentiment_col in news_df.columns:
                sentiment[:, 0] = (
                    pd.to_numeric(news_df[self.news_sentiment_col], errors='coerce')
                    .fillna(0.0)
                    .clip(-1.0, 1.0)
                    .to_numpy(dtype=np.float32)
                )
            elif self.news_sentiment_title_col in news_df.columns:
                sentiment[:, 0] = news_df[self.news_sentiment_title_col].map(self._lexicon_sentiment).to_numpy(dtype=np.float32)

        duration = np.zeros((len(news_df), 3), dtype=np.float32)
        duration[:, 2] = 1.0
        duration_cols = [
            self.news_duration_short_col,
            self.news_duration_long_col,
            self.news_duration_unsure_col,
        ]
        if all(col in news_df.columns for col in duration_cols):
            raw_duration = (
                news_df[duration_cols]
                .apply(pd.to_numeric, errors='coerce')
                .fillna(0.0)
                .to_numpy(dtype=np.float32)
            )
            duration = np.zeros_like(raw_duration, dtype=np.float32)
            duration[:, 2] = 1.0
            row_sum = raw_duration.sum(axis=1, keepdims=True)
            valid = row_sum.squeeze(-1) > 1e-8
            duration[valid] = raw_duration[valid] / row_sum[valid]

        self.news_df = news_df
        self.news_time_ns = news_df['_align_time'].astype('int64').to_numpy()
        self.news_embeddings = emb
        if self.news_rank_col and self.news_rank_col in news_df.columns:
            self.news_rank_values = (
                pd.to_numeric(news_df[self.news_rank_col], errors='coerce')
                .fillna(self.news_rank_fill_value)
                .to_numpy(dtype=np.float32)
            )
        else:
            self.news_rank_values = None
        self.news_novelty = novelty
        self.news_sentiment = sentiment
        self.news_duration = duration

    def _rank_candidate_news(self, idx, anchor_ns):
        if len(idx) <= self.news_max_items:
            return idx
        if self.news_selection_mode == 'latest':
            return idx[-self.news_max_items:]
        if self.news_rank_values is None:
            return idx[-self.news_max_items:]

        scores = self.news_rank_values[idx].astype(np.float32, copy=True)
        if self.news_selection_mode == 'quality_recency':
            age_days = ((anchor_ns - self.news_time_ns[idx]) / (24 * 3600 * 1e9)).astype(np.float32)
            recency = 1.0 - np.clip(age_days / max(float(self.news_lookback_days), 1.0), 0.0, 1.0)
            scores = scores + self.news_rank_recency_weight * recency
        elif self.news_selection_mode != 'quality':
            return idx[-self.news_max_items:]

        top_pos = np.argpartition(scores, -self.news_max_items)[-self.news_max_items:]
        selected = idx[top_pos]
        return np.sort(selected)

    def _select_news_for_anchor(self, anchor_time):
        # If no news file is configured, return padded zero tensors so the
        # rest of the training stack does not need special-case logic.
        if self.news_df is None or len(self.news_df) == 0:
            return (
                np.zeros((self.news_max_items, self.news_emb_dim), dtype=np.float32),
                np.zeros((self.news_max_items,), dtype=np.float32),
                np.zeros((self.news_max_items,), dtype=np.float32),
                np.zeros((self.news_max_items, 1), dtype=np.float32),
                np.zeros((self.news_max_items, 3), dtype=np.float32),
                np.zeros((self.news_max_items, 1), dtype=np.float32),
            )

        anchor_time = pd.Timestamp(anchor_time)
        if self.news_align_mode == 'date':
            anchor_time = anchor_time.normalize()
        anchor_ns = anchor_time.value
        low_ns = (anchor_time - pd.Timedelta(days=self.news_lookback_days)).value

        left = np.searchsorted(self.news_time_ns, low_ns, side='left')
        right = np.searchsorted(self.news_time_ns, anchor_ns, side='right')

        emb_out = np.zeros((self.news_max_items, self.news_emb_dim), dtype=np.float32)
        gap_out = np.zeros((self.news_max_items,), dtype=np.float32)
        mask_out = np.zeros((self.news_max_items,), dtype=np.float32)
        novelty_out = np.zeros((self.news_max_items, 1), dtype=np.float32)
        duration_out = np.zeros((self.news_max_items, 3), dtype=np.float32)
        sentiment_out = np.zeros((self.news_max_items, 1), dtype=np.float32)

        if right <= left:
            return emb_out, gap_out, mask_out, novelty_out, duration_out, sentiment_out

        idx = np.arange(left, right)
        idx = self._rank_candidate_news(idx, anchor_ns)

        used = len(idx)
        emb_out[:used] = self.news_embeddings[idx]
        gap_out[:used] = ((anchor_ns - self.news_time_ns[idx]) / (24 * 3600 * 1e9)).astype(np.float32)
        mask_out[:used] = 1.0
        novelty_out[:used] = self.news_novelty[idx]
        duration_out[:used] = self.news_duration[idx]
        sentiment_out[:used] = self.news_sentiment[idx]

        return emb_out, gap_out, mask_out, novelty_out, duration_out, sentiment_out

    def __getitem__(self, index):
        seq_x, seq_y, seq_x_mark, seq_y_mark = super().__getitem__(index)
        anchor_time = self.index.loc[index, self.time_col]
        news_emb, news_gap, news_mask, news_novelty, news_duration, news_sentiment = self._select_news_for_anchor(anchor_time)
        return seq_x, seq_y, seq_x_mark, seq_y_mark, news_emb, news_gap, news_mask, news_novelty, news_duration, news_sentiment
