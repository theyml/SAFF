#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-$(pwd)}"
DATA_DIR="$ROOT/data"
RESULT_DIR="${RESULT_DIR:-$ROOT/results_transformer_news_cross_attention_ablation}"
LOG_DIR="${LOG_DIR:-$ROOT/logs_transformer_news_cross_attention_ablation}"

PYTHON_BIN="${PYTHON_BIN:-python}"
CUDA_DEVICES="${CUDA_DEVICES:-0,1,2,3,4,5,6,7}"
MAX_JOBS="${MAX_JOBS:-6}"
VARIANTS="${VARIANTS:-pool_decay,cross_nodecay,cross_decay}"
NEWS_MAX_ITEMS="${NEWS_MAX_ITEMS:-128}"
NEWS_LOOKBACK_DAYS="${NEWS_LOOKBACK_DAYS:-60}"
ITRS="${ITRS:-3}"
TRAIN_EPOCHS="${TRAIN_EPOCHS:-10}"
PATIENCE="${PATIENCE:-3}"
NEWS_SELECTION_MODE="${NEWS_SELECTION_MODE:-latest}"
NEWS_RANK_COL="${NEWS_RANK_COL:-}"
NEWS_RANK_RECENCY_WEIGHT="${NEWS_RANK_RECENCY_WEIGHT:-0.15}"
NEWS_CROSS_GAP_SCALE="${NEWS_CROSS_GAP_SCALE:-1.0}"
NEWS_CROSS_FIXED_ALPHA="${NEWS_CROSS_FIXED_ALPHA:-0.05}"

mkdir -p "$RESULT_DIR" "$LOG_DIR"
cd "$ROOT"

# Transformer-only long-news ablation.
# Variants:
# - pool_decay: existing TransformerNewsDecay pooling adapter with learned decay
# - cross_nodecay: news cross-attention without temporal decay bias
# - cross_decay: news cross-attention with learned temporal decay bias
# - cross_fixed: optional news cross-attention with fixed temporal decay bias
#
# Format: ticker seq_len label_len pred_len
ROWS=(
  "INTC 96 48 24"
  "TSLA 96 48 24"
  "SBUX 96 48 24"
  "GOOGL_PRECOVID 96 48 24"
  "AAPL 72 36 10"
  "NFLX_PRECOVID 72 36 10"
)

first_existing() {
  local file
  for file in "$@"; do
    if [[ -f "$DATA_DIR/$file" ]]; then
      echo "$file"
      return 0
    fi
  done
  return 1
}

price_file_for_ticker() {
  case "$1" in
    GOOGL_PRECOVID)
      first_existing GOOGL_pre_covid_news_aligned.csv
      ;;
    NFLX_PRECOVID)
      first_existing NFLX_pre_covid_news_aligned.csv
      ;;
    *)
      first_existing "${1}_news_aligned.csv" "$(echo "$1" | tr '[:upper:]' '[:lower:]')_news_aligned.csv"
      ;;
  esac
}

news_file_for_ticker() {
  case "$1" in
    GOOGL_PRECOVID)
      first_existing GOOGL_title_finbert_emb.csv googl_title_finbert_emb.csv
      ;;
    NFLX_PRECOVID)
      first_existing NFLX_title_finbert_emb_precovid.csv NFLX_title_finbert_emb.csv nflx_title_finbert_emb.csv
      ;;
    AAPL)
      first_existing AAPL_title_finbert_emb.csv apple_title_finbert_emb.csv aapl_title_finbert_emb.csv
      ;;
    *)
      first_existing "${1}_title_finbert_emb.csv" "$(echo "$1" | tr '[:upper:]' '[:lower:]')_title_finbert_emb.csv"
      ;;
  esac
}

IFS=',' read -r -a GPUS <<< "$CUDA_DEVICES"
IFS=',' read -r -a VARIANT_LIST <<< "$VARIANTS"
JOB_INDEX=0

run_one() {
  local ticker="$1"
  local sl="$2"
  local ll="$3"
  local pl="$4"
  local variant="$5"

  local price_file
  local news_file
  if ! price_file="$(price_file_for_ticker "$ticker")"; then
    echo "Missing price file for $ticker under $DATA_DIR" >&2
    exit 1
  fi
  if ! news_file="$(news_file_for_ticker "$ticker")"; then
    echo "Missing news file for $ticker under $DATA_DIR" >&2
    exit 1
  fi

  local model="TransformerCrossAttentionNewsDecay"
  local variant_args=()
  case "$variant" in
    pool_decay)
      model="TransformerNewsDecay"
      ;;
    cross_nodecay)
      variant_args=(--disable_time_decay)
      ;;
    cross_decay)
      ;;
    cross_fixed)
      variant_args=(--news_cross_use_fixed_decay --news_cross_fixed_decay_alpha "$NEWS_CROSS_FIXED_ALPHA")
      ;;
    *)
      echo "Unknown variant: $variant" >&2
      exit 1
      ;;
  esac

  local rank_args=()
  if [[ -n "$NEWS_RANK_COL" ]]; then
    rank_args=(--news_rank_col "$NEWS_RANK_COL")
  fi

  local stem="${price_file%.csv}"
  echo
  echo "===== $variant | $ticker | $model | sl=$sl pl=$pl | lookback=$NEWS_LOOKBACK_DAYS k=$NEWS_MAX_ITEMS | CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset} ====="

  "$PYTHON_BIN" cross_attention_run.py \
    --model "$model" \
    --root_path "$DATA_DIR" \
    --data_path "$price_file" \
    --features M \
    --n_features 5 \
    --seq_len "$sl" \
    --label_len "$ll" \
    --pred_len "$pl" \
    --scaler_mode global \
    --use_news \
    --news_path "$DATA_DIR/$news_file" \
    --news_time_col Date \
    --news_embed_prefix news_emb_ \
    --news_emb_dim 768 \
    --news_max_items "$NEWS_MAX_ITEMS" \
    --news_lookback_days "$NEWS_LOOKBACK_DAYS" \
    --news_selection_mode "$NEWS_SELECTION_MODE" \
    "${rank_args[@]}" \
    --news_rank_recency_weight "$NEWS_RANK_RECENCY_WEIGHT" \
    --news_cross_gap_scale "$NEWS_CROSS_GAP_SCALE" \
    "${variant_args[@]}" \
    --batch_size 32 \
    --learning_rate 1e-3 \
    --train_epochs "$TRAIN_EPOCHS" \
    --patience "$PATIENCE" \
    --itrs "$ITRS" \
    --num_workers 4 \
    --disable_progress \
    --result_path "$RESULT_DIR" \
    --des "${variant}_lookback${NEWS_LOOKBACK_DAYS}_k${NEWS_MAX_ITEMS}_g" \
    | tee "$LOG_DIR/${stem}_sl${sl}_pl${pl}_${variant}_lookback${NEWS_LOOKBACK_DAYS}_k${NEWS_MAX_ITEMS}.log"
}

launch() {
  while [[ "$(jobs -rp | wc -l)" -ge "$MAX_JOBS" ]]; do
    wait -n
  done

  local gpu="${GPUS[$((JOB_INDEX % ${#GPUS[@]}))]}"
  JOB_INDEX=$((JOB_INDEX + 1))
  CUDA_VISIBLE_DEVICES="$gpu" run_one "$@" &
}

for row in "${ROWS[@]}"; do
  read -r ticker sl ll pl <<< "$row"
  for variant in "${VARIANT_LIST[@]}"; do
    variant="$(echo "$variant" | xargs)"
    [[ -z "$variant" ]] && continue
    launch "$ticker" "$sl" "$ll" "$pl" "$variant"
  done
done

wait

echo
echo "Done. Results: $RESULT_DIR"
echo "Logs: $LOG_DIR"
