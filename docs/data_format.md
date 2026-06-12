# Data Format

SAFF uses two aligned inputs:

1. A price time-series CSV under `data/`, passed with `--data_path`.
2. An optional news table, passed with `--news_path` and enabled by `--use_news`.

## Price CSV

The default loader expects a `Date` column and numeric feature columns. NASDAQ-style price files with dollar signs are accepted because the loader strips `$` before conversion.

Example columns:

```text
Date,Close/Last,Volume,Open,High,Low
```

For multivariate forecasting, use `--features M --n_features <number_of_price_columns>`.

## News CSV

The news-aware loader expects one timestamp column and precomputed numeric embedding columns.

Required columns:

```text
Date,news_emb_0,news_emb_1,...,news_emb_N
```

Useful optional columns:

```text
Article_title,Stock_symbol,zs_rel_weight,sentiment_score,
duration_short_prob,duration_long_prob,duration_unsure_prob
```

Important arguments:

```bash
--use_news
--news_path news_schema_example.csv
--news_time_col Date
--news_embed_prefix news_emb_
--news_emb_dim 4
--news_max_items 32
--news_lookback_days 30
```

`news_path` can be either absolute or relative to `--root_path`. Full experiment embeddings are intentionally not committed because they are large generated artifacts.
