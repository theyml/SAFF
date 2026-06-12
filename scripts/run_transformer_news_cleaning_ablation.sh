#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-$(pwd)}"
DATA_DIR="$ROOT/data"
RESULT_DIR="${RESULT_DIR:-$ROOT/results_backbone_news_cleaning_ablation}"
LOG_DIR="${LOG_DIR:-$ROOT/logs_backbone_news_cleaning_ablation}"

PYTHON_BIN="${PYTHON_BIN:-python}"
CUDA_DEVICES="${CUDA_DEVICES:-0,1,2,3,4,5,6,7}"
MAX_JOBS="${MAX_JOBS:-8}"
BACKBONES="${BACKBONES:-Transformer}"
ITRS="${ITRS:-3}"
TRAIN_EPOCHS="${TRAIN_EPOCHS:-10}"
PATIENCE="${PATIENCE:-3}"
NEWS_RANK_RECENCY_WEIGHT="${NEWS_RANK_RECENCY_WEIGHT:-0.15}"

mkdir -p "$RESULT_DIR" "$LOG_DIR"
cd "$ROOT"

# Selected stock-horizon rows where latest32 decay already provides useful evidence.
# This run replaces latest32 with quality_recency on scored title files and runs
# only plain/decay arms, so the old price-only baselines remain reusable.
ROWS=(
  "INTC 96 48 24"
  "TSLA 96 48 24"
  "SBUX 96 48 24"
  "MU 96 48 24"
  "ORCL 96 48 24"
  "ADP 72 36 10"
  "ADP 96 48 24"
  "AMD 72 36 10"
  "AAPL 72 36 10"
  "GOOGL_PRECOVID 96 48 24"
  "INTC 72 36 10"
  "BKNG 96 48 24"
  "AMD 96 48 24"
  "NFLX_PRECOVID 72 36 10"
  "QCOM 96 48 24"
  "COST 96 48 24"
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

news_model_name() {
  case "$1" in
    DLinear) echo "DLinearNewsDecay" ;;
    PatchTST) echo "PatchTSTNewsDecay" ;;
    TimesNet) echo "TimesNetNewsDecay" ;;
    TimeMixer) echo "TimeMixerNewsDecay" ;;
    SegRNN) echo "SegRNNNewsDecay" ;;
    Transformer) echo "TransformerNewsDecay" ;;
    *) echo "Unsupported backbone: $1" >&2; exit 1 ;;
  esac
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

scored_news_file_for_ticker() {
  case "$1" in
    GOOGL_PRECOVID)
      first_existing \
        GOOGL_title_finbert_emb_zs_relevance.csv \
        GOOGL_title_finbert_emb_zs060_relevance.csv
      ;;
    NFLX_PRECOVID)
      first_existing \
        NFLX_title_finbert_emb_precovid_zs_relevance.csv \
        NFLX_title_finbert_emb_zs_relevance.csv
      ;;
    *)
      first_existing \
        "${1}_title_finbert_emb_zs_relevance.csv" \
        "$(echo "$1" | tr '[:upper:]' '[:lower:]')_title_finbert_emb_zs_relevance.csv"
      ;;
  esac
}

extra_backbone_args() {
  case "$1" in
    TimeMixer) echo "--down_sampling_method avg --down_sampling_layers 1 --down_sampling_window 2" ;;
    SegRNN) echo "--seg_len 1" ;;
    *) echo "" ;;
  esac
}

IFS=',' read -r -a GPUS <<< "$CUDA_DEVICES"
IFS=',' read -r -a MODEL_NAMES <<< "$BACKBONES"
JOB_INDEX=0

run_one() {
  local ticker="$1"
  local sl="$2"
  local ll="$3"
  local pl="$4"
  local backbone="$5"
  local mode="$6"

  local price_file
  local news_file
  if ! price_file="$(price_file_for_ticker "$ticker")"; then
    echo "Missing price file for $ticker under $DATA_DIR" >&2
    exit 1
  fi
  if ! news_file="$(scored_news_file_for_ticker "$ticker")"; then
    echo "Missing scored news file for $ticker under $DATA_DIR" >&2
    echo "Expected *_title_finbert_emb_zs_relevance.csv with zs_rel_weight." >&2
    exit 1
  fi

  local news_model
  news_model="$(news_model_name "$backbone")"

  local decay_flag=()
  if [[ "$mode" == "plain" ]]; then
    decay_flag=(--disable_time_decay)
  fi

  local backbone_args=()
  read -r -a backbone_args <<< "$(extra_backbone_args "$backbone")"

  local stem="${price_file%.csv}"
  echo
  echo "===== quality_recency $mode | $ticker | $backbone | sl=$sl pl=$pl | news=$news_file | CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset} ====="

  "$PYTHON_BIN" decay_run.py \
    --model "$news_model" \
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
    --news_max_items 32 \
    --news_lookback_days 30 \
    --news_selection_mode quality_recency \
    --news_rank_col zs_rel_weight \
    --news_rank_recency_weight "$NEWS_RANK_RECENCY_WEIGHT" \
    "${decay_flag[@]}" \
    --batch_size 32 \
    --learning_rate 1e-3 \
    --train_epochs "$TRAIN_EPOCHS" \
    --patience "$PATIENCE" \
    --itrs "$ITRS" \
    --num_workers 4 \
    --disable_progress \
    "${backbone_args[@]}" \
    --result_path "$RESULT_DIR" \
    --des "${news_model}_${mode}_quality_recency_g" \
    | tee "$LOG_DIR/${stem}_sl${sl}_pl${pl}_${news_model}_${mode}_quality_recency.log"
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
  for backbone in "${MODEL_NAMES[@]}"; do
    backbone="$(echo "$backbone" | xargs)"
    [[ -z "$backbone" ]] && continue
    launch "$ticker" "$sl" "$ll" "$pl" "$backbone" plain
    launch "$ticker" "$sl" "$ll" "$pl" "$backbone" decay
  done
done

wait

echo
echo "Done. Results: $RESULT_DIR"
echo "Logs: $LOG_DIR"
