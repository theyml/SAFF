# SAFF

Code for **SAFF: Situation-Aware Financial Forecasting with Decay-Aware Semantic News Fusion**.

SAFF is a research codebase for situation-aware financial forecasting with semantic news fusion. It extends a financial time-series forecasting stack with news-aware datasets, decay-aware residual adapters, cross-backbone experiments, and Transformer news-fusion ablations.

The core idea is to keep a normal price forecaster as the backbone, then add a small learned temporal-decay module that turns recent headline embeddings into a horizon-specific news residual.

Architecture figure: [`docs/figures/decay_aware_module_structure.pdf`](docs/figures/decay_aware_module_structure.pdf)

## What Is Included

- News-aware data loading in `data_provider/news_data_loader.py`.
- Decay-aware model wrappers for DLinear, PatchTST, TimesNet, TimeMixer, Transformer, and iTransformer.
- A reusable backbone-agnostic adapter in `models/NewsDecayAdapter.py`.
- Transformer cross-attention and interaction-fusion variants.
- Experiment entry points:
  - `run.py` for price-only baselines.
  - `decay_run.py` for news-aware decay experiments.
  - `cross_attention_run.py` for Transformer news cross-attention experiments.
- Main training and ablation scripts under `scripts/`.
- Small public price-series examples under `data/`.
- A synthetic news schema example at `data/news_schema_example.csv`.

Large generated assets such as full news tables, FinBERT embeddings, logs, checkpoints, prediction dumps, and paper build artifacts are not committed.

## Install

Create an environment with Python 3.10+ and install PyTorch for your CUDA/CPU setup first. Then install the project dependencies:

```bash
pip install -r requirements.txt
```

For embedding-generation scripts, Hugging Face model downloads may be needed unless the models are already cached.

## Quick Checks

Run a small price-only dry run:

```bash
python run.py \
  --model DLinear \
  --root_path ./data \
  --data_path Apple.csv \
  --features M \
  --n_features 5 \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 24 \
  --train_epochs 1 \
  --batch_size 16 \
  --num_workers 0 \
  --dry_run \
  --result_path results_demo
```

Run a small news-aware dry run with the synthetic news schema:

```bash
python decay_run.py \
  --model DLinearNewsDecay \
  --use_news \
  --root_path ./data \
  --data_path Apple.csv \
  --news_path news_schema_example.csv \
  --news_time_col Date \
  --news_embed_prefix news_emb_ \
  --news_emb_dim 4 \
  --features M \
  --n_features 5 \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 24 \
  --train_epochs 1 \
  --batch_size 16 \
  --num_workers 0 \
  --dry_run \
  --result_path results_demo
```

## Full Experiments

The scripts under `scripts/` encode the main experiment families:

- `run_backbone_news_decay_ablation.sh`: price/plain/decay comparisons across backbones.
- `run_transformer_news_memory_ablation.sh`: news-memory size ablations.
- `run_transformer_news_cross_attention_ablation.sh`: Transformer long-news cross-attention variants.
- `run_transformer_news_cleaning_ablation.sh`: latest versus quality-ranked news selection.
- `run_transformer_news_interaction_fusion_baseline.sh`: interaction-fusion news baseline.
- `build_news_embeddings.py`: offline embedding construction from news text.

Full runs require aligned price CSVs and precomputed news embedding CSVs using the schema described in `docs/data_format.md`.

## Acknowledgments

This repository builds on the Financial-Time-Series forecasting codebase and time-series model implementations inspired by Time-Series-Library style baselines. The SAFF-specific additions are the news-aware loader, decay-aware residual fusion modules, and financial-news experiment scripts.

## License

Apache License 2.0. See `LICENSE`.
