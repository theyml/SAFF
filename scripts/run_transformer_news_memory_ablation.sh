#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/data/lym/Financial-Time-Series}"
DATA_DIR="$ROOT/data"
RESULT_DIR="${RESULT_DIR:-$ROOT/results_transformer_news_memory_ablation}"
LOG_DIR="${LOG_DIR:-$ROOT/logs_transformer_news_memory_ablation}"

PYTHON_BIN="${PYTHON_BIN:-python}"
CUDA_DEVICES="${CUDA_DEVICES:-0,1,2,3,4,5,6,7}"
MAX_JOBS="${MAX_JOBS:-4}"
LOOKBACK_DAYS_LIST="${LOOKBACK_DAYS_LIST:-60}"
NEWS_MAX_ITEMS_LIST="${NEWS_MAX_ITEMS_LIST:-16,32,64,128}"
ITRS="${ITRS:-3}"
TRAIN_EPOCHS="${TRAIN_EPOCHS:-10}"
PATIENCE="${PATIENCE:-3}"
NEWS_SELECTION_MODE="${NEWS_SELECTION_MODE:-latest}"
NEWS_RANK_COL="${NEWS_RANK_COL:-}"
NEWS_RANK_RECENCY_WEIGHT="${NEWS_RANK_RECENCY_WEIGHT:-0.15}"

mkdir -p "$RESULT_DIR" "$LOG_DIR"
cd "$ROOT"

# Effective news-memory sweep.
# A calendar lookback sweep with news_max_items=32 saturates quickly on dense
# tickers. This sweep fixes a long candidate window and varies the actual
# number of latest headlines admitted into the model.
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
IFS=',' read -r -a LOOKBACKS <<< "$LOOKBACK_DAYS_LIST"
IFS=',' read -r -a MAX_ITEMS_VALUES <<< "$NEWS_MAX_ITEMS_LIST"
JOB_INDEX=0

run_one() {
  local ticker="$1"
  local sl="$2"
  local ll="$3"
  local pl="$4"
  local lookback="$5"
  local max_items="$6"
  local mode="$7"

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

  local decay_flag=()
  if [[ "$mode" == "plain" ]]; then
    decay_flag=(--disable_time_decay)
  fi

  local rank_args=()
  if [[ -n "$NEWS_RANK_COL" ]]; then
    rank_args=(--news_rank_col "$NEWS_RANK_COL")
  fi

  local stem="${price_file%.csv}"
  echo
  echo "===== lookback=$lookback k=$max_items $mode | $ticker | TransformerNewsDecay | sl=$sl pl=$pl | CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset} ====="

  "$PYTHON_BIN" decay_run.py \
    --model TransformerNewsDecay \
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
    --news_max_items "$max_items" \
    --news_lookback_days "$lookback" \
    --news_selection_mode "$NEWS_SELECTION_MODE" \
    "${rank_args[@]}" \
    --news_rank_recency_weight "$NEWS_RANK_RECENCY_WEIGHT" \
    "${decay_flag[@]}" \
    --batch_size 32 \
    --learning_rate 1e-3 \
    --train_epochs "$TRAIN_EPOCHS" \
    --patience "$PATIENCE" \
    --itrs "$ITRS" \
    --num_workers 4 \
    --disable_progress \
    --result_path "$RESULT_DIR" \
    --des "TransformerNewsDecay_${mode}_lookback${lookback}_k${max_items}_g" \
    | tee "$LOG_DIR/${stem}_sl${sl}_pl${pl}_lookback${lookback}_k${max_items}_${mode}.log"
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
  for lookback in "${LOOKBACKS[@]}"; do
    lookback="$(echo "$lookback" | xargs)"
    [[ -z "$lookback" ]] && continue
    for max_items in "${MAX_ITEMS_VALUES[@]}"; do
      max_items="$(echo "$max_items" | xargs)"
      [[ -z "$max_items" ]] && continue
      launch "$ticker" "$sl" "$ll" "$pl" "$lookback" "$max_items" plain
      launch "$ticker" "$sl" "$ll" "$pl" "$lookback" "$max_items" decay
    done
  done
done

wait

echo
echo "Done. Results: $RESULT_DIR"
echo "Logs: $LOG_DIR"
