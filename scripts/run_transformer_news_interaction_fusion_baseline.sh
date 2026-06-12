#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-$(pwd)}"
DATA_DIR="${DATA_DIR:-$ROOT/data}"
RESULT_DIR="${RESULT_DIR:-$ROOT/results_transformer_news_interaction_fusion}"
LOG_DIR="${LOG_DIR:-$ROOT/logs_transformer_news_interaction_fusion}"
PYTHON_BIN="${PYTHON_BIN:-python}"

CUDA_DEVICES="${CUDA_DEVICES:-0}"
ONLY_TICKERS="${ONLY_TICKERS:-INTC,SBUX,MU,AMD,CRM,QCOM,DIS}"
SEQ_LEN="${SEQ_LEN:-96}"
LABEL_LEN="${LABEL_LEN:-48}"
PRED_LEN="${PRED_LEN:-24}"
ITRS="${ITRS:-3}"
TRAIN_EPOCHS="${TRAIN_EPOCHS:-10}"
PATIENCE="${PATIENCE:-3}"
BATCH_SIZE="${BATCH_SIZE:-32}"
NEWS_MAX_ITEMS="${NEWS_MAX_ITEMS:-32}"
NEWS_LOOKBACK_DAYS="${NEWS_LOOKBACK_DAYS:-30}"
D_MODEL="${D_MODEL:-64}"
N_HEADS="${N_HEADS:-4}"
D_FF="${D_FF:-128}"
E_LAYERS="${E_LAYERS:-2}"
D_LAYERS="${D_LAYERS:-1}"
LEARNING_RATE="${LEARNING_RATE:-0.001}"
SKIP_MISSING_TICKERS="${SKIP_MISSING_TICKERS:-1}"
DRY_RUN="${DRY_RUN:-0}"

mkdir -p "$RESULT_DIR" "$LOG_DIR"

IFS=',' read -ra GPU_IDS <<< "$CUDA_DEVICES"
GPU_COUNT="${#GPU_IDS[@]}"
NEXT_GPU_IDX=0

first_existing() {
  local name
  for name in "$@"; do
    if [[ -f "$DATA_DIR/$name" ]]; then
      printf '%s\n' "$name"
      return 0
    fi
  done
  return 1
}

files_for_ticker() {
  local ticker="$1"
  case "$ticker" in
    GOOGL_PRECOVID)
      echo "$(first_existing GOOGL_pre_covid_news_aligned.csv)|$(first_existing GOOGL_title_finbert_emb.csv)"
      ;;
    NFLX_PRECOVID)
      echo "$(first_existing NFLX_pre_covid_news_aligned.csv)|$(first_existing NFLX_title_finbert_emb.csv)"
      ;;
    *)
      echo "$(first_existing "${ticker}_news_aligned.csv")|$(first_existing "${ticker}_title_finbert_emb.csv")"
      ;;
  esac
}

run_one() {
  local ticker="$1"
  local price_file="$2"
  local news_file="$3"
  local gpu="$4"
  local stem="${price_file%.csv}"
  local dry_args=()
  if [[ "$DRY_RUN" == "1" ]]; then
    dry_args=(--dry_run)
  fi

  echo "===== FININ-style interaction fusion | $ticker | ${SEQ_LEN}/${PRED_LEN} | gpu=$gpu ====="
  CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON_BIN" decay_run.py \
    --model TransformerNewsInteractionFusion \
    --root_path "$DATA_DIR" \
    --data_path "$price_file" \
    --features M \
    --n_features 5 \
    --target OT \
    --seq_len "$SEQ_LEN" \
    --label_len "$LABEL_LEN" \
    --pred_len "$PRED_LEN" \
    --d_model "$D_MODEL" \
    --n_heads "$N_HEADS" \
    --e_layers "$E_LAYERS" \
    --d_layers "$D_LAYERS" \
    --d_ff "$D_FF" \
    --batch_size "$BATCH_SIZE" \
    --train_epochs "$TRAIN_EPOCHS" \
    --patience "$PATIENCE" \
    --learning_rate "$LEARNING_RATE" \
    --itrs "$ITRS" \
    --result_path "$RESULT_DIR" \
    --use_news \
    --news_path "$DATA_DIR/$news_file" \
    --news_time_col Date \
    --news_embed_prefix news_emb_ \
    --news_emb_dim 768 \
    --news_max_items "$NEWS_MAX_ITEMS" \
    --news_lookback_days "$NEWS_LOOKBACK_DAYS" \
    --scaler_mode global \
    --des TransformerNewsInteractionFusion_finin_style_g \
    "${dry_args[@]}" \
    2>&1 | tee "$LOG_DIR/${stem}_sl${SEQ_LEN}_pl${PRED_LEN}_TransformerNewsInteractionFusion_finin_style.log"
}

launch() {
  local ticker="$1"
  local file_pair price_file news_file gpu
  file_pair="$(files_for_ticker "$ticker")"
  price_file="${file_pair%%|*}"
  news_file="${file_pair##*|}"

  if [[ -z "$price_file" || -z "$news_file" ]]; then
    if [[ "$SKIP_MISSING_TICKERS" == "1" ]]; then
      echo "Skipping $ticker because required files are missing: price='$price_file' news='$news_file'"
      return 0
    fi
    echo "Missing required files for $ticker: price='$price_file' news='$news_file'" >&2
    return 1
  fi

  gpu="${GPU_IDS[$NEXT_GPU_IDX]}"
  NEXT_GPU_IDX=$(( (NEXT_GPU_IDX + 1) % GPU_COUNT ))
  run_one "$ticker" "$price_file" "$news_file" "$gpu" &
}

IFS=',' read -ra TICKERS <<< "$ONLY_TICKERS"
for raw_ticker in "${TICKERS[@]}"; do
  ticker="$(echo "$raw_ticker" | tr '[:lower:]' '[:upper:]' | xargs)"
  [[ -z "$ticker" ]] && continue
  launch "$ticker"
done

wait
echo "All FININ-style interaction fusion jobs finished."
