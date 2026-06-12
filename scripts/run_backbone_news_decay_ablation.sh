#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-$(pwd)}"
DATA_DIR="$ROOT/data"
RESULT_DIR="$ROOT/results_backbone_news_decay_ablation"
LOG_DIR="$ROOT/logs_backbone_news_decay_ablation"
PYTHON_BIN="${PYTHON_BIN:-python}"
ONLY_TICKERS="${ONLY_TICKERS:-AAPL,NVDA,TSLA}"
BACKBONES="${BACKBONES:-DLinear,PatchTST,TimesNet}"
CUDA_DEVICES="${CUDA_DEVICES:-0,1,2,3,4,5,6,7}"
RUN_BASELINE="${RUN_BASELINE:-1}"
RUN_PLAIN="${RUN_PLAIN:-1}"
RUN_DECAY="${RUN_DECAY:-1}"
NEWS_SELECTION_MODE="${NEWS_SELECTION_MODE:-latest}"
NEWS_RANK_COL="${NEWS_RANK_COL:-}"
NEWS_RANK_RECENCY_WEIGHT="${NEWS_RANK_RECENCY_WEIGHT:-0.15}"
BATCH_SIZE="${BATCH_SIZE:-32}"
TRAIN_EPOCHS="${TRAIN_EPOCHS:-10}"
PATIENCE="${PATIENCE:-3}"
ITRS="${ITRS:-3}"
NUM_WORKERS="${NUM_WORKERS:-4}"
DRY_RUN="${DRY_RUN:-0}"
SKIP_MISSING_TICKERS="${SKIP_MISSING_TICKERS:-0}"

mkdir -p "$RESULT_DIR" "$LOG_DIR"
cd "$ROOT"

SETTINGS=(
  "72 36 10"
  "96 48 24"
)

if [[ -n "${ONLY_SETTINGS:-}" ]]; then
  SETTINGS=()
  IFS=',' read -r -a SETTING_ITEMS <<< "$ONLY_SETTINGS"
  for setting_item in "${SETTING_ITEMS[@]}"; do
    setting_item="$(echo "$setting_item" | tr ':' ' ' | xargs)"
    SETTINGS+=("$setting_item")
  done
fi

IFS=',' read -r -a TICKERS <<< "$ONLY_TICKERS"
IFS=',' read -r -a MODEL_NAMES <<< "$BACKBONES"
IFS=',' read -r -a GPU_IDS <<< "$CUDA_DEVICES"

if [[ "${#GPU_IDS[@]}" -eq 0 ]]; then
  echo "CUDA_DEVICES is empty." >&2
  exit 1
fi

for i in "${!GPU_IDS[@]}"; do
  GPU_IDS[$i]="$(echo "${GPU_IDS[$i]}" | xargs)"
done

NUM_GPUS="${#GPU_IDS[@]}"
JOB_INDEX=0
declare -a SLOT_PIDS=()
declare -a SLOT_LABELS=()

wait_for_slot() {
  local slot="$1"
  local pid="${SLOT_PIDS[$slot]:-}"
  local label="${SLOT_LABELS[$slot]:-}"
  if [[ -n "$pid" ]]; then
    echo
    echo "===== Wait slot $slot | pid=$pid | $label ====="
    if ! wait "$pid"; then
      echo "Job failed on slot $slot: $label" >&2
      exit 1
    fi
    SLOT_PIDS[$slot]=""
    SLOT_LABELS[$slot]=""
  fi
}

launch_on_next_gpu() {
  local label="$1"
  shift
  local slot=$((JOB_INDEX % NUM_GPUS))
  local gpu="${GPU_IDS[$slot]}"
  wait_for_slot "$slot"
  echo
  echo "===== Launch GPU $gpu | $label ====="
  (
    export CUDA_VISIBLE_DEVICES="$gpu"
    "$@"
  ) &
  SLOT_PIDS[$slot]="$!"
  SLOT_LABELS[$slot]="$label"
  JOB_INDEX=$((JOB_INDEX + 1))
}

wait_for_all_jobs() {
  local slot
  for slot in "${!GPU_IDS[@]}"; do
    wait_for_slot "$slot"
  done
}

first_existing() {
  local file
  for file in "$@"; do
    if [[ -f "$DATA_DIR/$file" ]]; then
      echo "$file"
      return 0
    fi
  done
  echo "Missing all candidate files: $*" >&2
  return 1
}

ticker_files() {
  local ticker="$1"
  case "$ticker" in
    AAPL)
      echo "$(first_existing AAPL_news_aligned.csv)|$(first_existing AAPL_title_finbert_emb.csv)"
      ;;
    NVDA)
      echo "$(first_existing NVDA_news_aligned.csv)|$(first_existing NVDA_title_finbert_emb.csv)"
      ;;
    TSLA)
      echo "$(first_existing TSLA_news_aligned.csv)|$(first_existing TSLA_title_finbert_emb.csv)"
      ;;
    GOOGL)
      echo "$(first_existing GOOGL_news_aligned.csv)|$(first_existing GOOGL_title_finbert_emb.csv)"
      ;;
    GOOGL_PRECOVID)
      echo "$(first_existing GOOGL_pre_covid_news_aligned.csv)|$(first_existing GOOGL_title_finbert_emb.csv)"
      ;;
    NFLX)
      echo "$(first_existing NFLX_news_aligned.csv)|$(first_existing NFLX_title_finbert_emb.csv)"
      ;;
    NFLX_PRECOVID)
      echo "$(first_existing NFLX_pre_covid_news_aligned.csv)|$(first_existing NFLX_title_finbert_emb_precovid.csv NFLX_title_finbert_emb.csv)"
      ;;
    ORCL)
      echo "$(first_existing ORCL_news_aligned.csv)|$(first_existing ORCL_title_finbert_emb.csv)"
      ;;
    AMZN)
      echo "$(first_existing AMZN_news_aligned.csv)|$(first_existing AMZN_title_finbert_emb.csv)"
      ;;
    PANW)
      echo "$(first_existing PANW_news_aligned.csv)|$(first_existing PANW_title_finbert_emb.csv)"
      ;;
    DIS)
      echo "$(first_existing DIS_news_aligned.csv)|$(first_existing DIS_title_finbert_emb.csv)"
      ;;
    AMD)
      echo "$(first_existing AMD_news_aligned.csv)|$(first_existing AMD_title_finbert_emb.csv)"
      ;;
    MSFT)
      echo "$(first_existing MSFT_news_aligned.csv)|$(first_existing MSFT_title_finbert_emb.csv)"
      ;;
    ADBE)
      echo "$(first_existing ADBE_news_aligned.csv adbe_news_aligned.csv)|$(first_existing ADBE_title_finbert_emb.csv adbe_title_finbert_emb.csv)"
      ;;
    CSCO)
      echo "$(first_existing CSCO_news_aligned.csv csco_news_aligned.csv)|$(first_existing CSCO_title_finbert_emb.csv csco_title_finbert_emb.csv)"
      ;;
    INTC)
      echo "$(first_existing INTC_news_aligned.csv intc_news_aligned.csv)|$(first_existing INTC_title_finbert_emb.csv intc_title_finbert_emb.csv)"
      ;;
    QCOM)
      echo "$(first_existing QCOM_news_aligned.csv qcom_news_aligned.csv)|$(first_existing QCOM_title_finbert_emb.csv qcom_title_finbert_emb.csv)"
      ;;
    ADP)
      echo "$(first_existing ADP_news_aligned.csv)|$(first_existing ADP_title_finbert_emb.csv)"
      ;;
    AMAT)
      echo "$(first_existing AMAT_news_aligned.csv)|$(first_existing AMAT_title_finbert_emb.csv)"
      ;;
    BKNG)
      echo "$(first_existing BKNG_news_aligned.csv)|$(first_existing BKNG_title_finbert_emb.csv)"
      ;;
    COST)
      echo "$(first_existing COST_news_aligned.csv)|$(first_existing COST_title_finbert_emb.csv)"
      ;;
    MRVL)
      echo "$(first_existing MRVL_news_aligned.csv)|$(first_existing MRVL_title_finbert_emb.csv)"
      ;;
    MU)
      echo "$(first_existing MU_news_aligned.csv)|$(first_existing MU_title_finbert_emb.csv)"
      ;;
    SBUX)
      echo "$(first_existing SBUX_news_aligned.csv)|$(first_existing SBUX_title_finbert_emb.csv)"
      ;;
    ASML)
      echo "$(first_existing ASML_news_aligned.csv)|$(first_existing ASML_title_finbert_emb.csv)"
      ;;
    CRWD)
      echo "$(first_existing CRWD_news_aligned.csv)|$(first_existing CRWD_title_finbert_emb.csv)"
      ;;
    TSM)
      echo "$(first_existing TSM_news_aligned.csv)|$(first_existing TSM_title_finbert_emb.csv)"
      ;;
    AVGO)
      echo "$(first_existing AVGO_news_aligned.csv)|$(first_existing AVGO_title_finbert_emb.csv)"
      ;;
    CRM)
      echo "$(first_existing CRM_news_aligned.csv)|$(first_existing CRM_title_finbert_emb.csv)"
      ;;
    CVX)
      echo "$(first_existing CVX_news_aligned.csv)|$(first_existing CVX_title_finbert_emb.csv)"
      ;;
    NOW)
      echo "$(first_existing NOW_news_aligned.csv)|$(first_existing NOW_title_finbert_emb.csv)"
      ;;
    SHOP)
      echo "$(first_existing SHOP_news_aligned.csv)|$(first_existing SHOP_title_finbert_emb.csv)"
      ;;
    UBER)
      echo "$(first_existing UBER_news_aligned.csv)|$(first_existing UBER_title_finbert_emb.csv)"
      ;;
    *)
      echo "Unknown ticker mapping: $ticker" >&2
      return 1
      ;;
  esac
}

news_model_name() {
  local backbone="$1"
  case "$backbone" in
    DLinear) echo "DLinearNewsDecay" ;;
    PatchTST) echo "PatchTSTNewsDecay" ;;
    TimesNet) echo "TimesNetNewsDecay" ;;
    TimeMixer) echo "TimeMixerNewsDecay" ;;
    SegRNN) echo "SegRNNNewsDecay" ;;
    Transformer) echo "TransformerNewsDecay" ;;
    *)
      echo "Unsupported backbone for news adapter: $backbone" >&2
      return 1
      ;;
  esac
}

run_price_only() {
  local ticker="$1"
  local backbone="$2"
  local price_file="$3"
  local stem="$4"
  local sl="$5"
  local ll="$6"
  local pl="$7"

  local dry_run_args=()
  local backbone_args=()
  if [[ "$DRY_RUN" == "1" ]]; then
    dry_run_args=(--dry_run)
  fi
  case "$backbone" in
    TimeMixer)
      backbone_args=(--down_sampling_method avg --down_sampling_layers 1 --down_sampling_window 2)
      ;;
    SegRNN)
      backbone_args=(--seg_len 1)
      ;;
  esac
  local -a optional_args=()
  if [[ ${#dry_run_args[@]} -gt 0 ]]; then
    optional_args+=("${dry_run_args[@]}")
  fi
  if [[ ${#backbone_args[@]} -gt 0 ]]; then
    optional_args+=("${backbone_args[@]}")
  fi

  echo
  echo "===== Price-only | $ticker | $backbone | sl=$sl pl=$pl | CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset} ====="
  "$PYTHON_BIN" run.py \
    --model "$backbone" \
    --root_path "$DATA_DIR" \
    --data_path "$price_file" \
    --features M \
    --n_features 5 \
    --seq_len "$sl" \
    --label_len "$ll" \
    --pred_len "$pl" \
    --scaler_mode global \
    --batch_size "$BATCH_SIZE" \
    --learning_rate 1e-3 \
    --train_epochs "$TRAIN_EPOCHS" \
    --patience "$PATIENCE" \
    --itrs "$ITRS" \
    --num_workers "$NUM_WORKERS" \
    --disable_progress \
    "${optional_args[@]}" \
    --result_path "$RESULT_DIR" \
    --des "${backbone}_price_only_g" | tee "$LOG_DIR/${stem}_sl${sl}_pl${pl}_${backbone}_price_only.log"
}

run_news_adapter() {
  local ticker="$1"
  local backbone="$2"
  local news_model="$3"
  local price_file="$4"
  local news_file="$5"
  local stem="$6"
  local sl="$7"
  local ll="$8"
  local pl="$9"
  local mode="${10}"
  local extra_decay_flag=()
  local rank_args=()
  local dry_run_args=()
  local backbone_args=()

  if [[ "$mode" == "plain" ]]; then
    extra_decay_flag=(--disable_time_decay)
  fi
  if [[ -n "$NEWS_RANK_COL" ]]; then
    rank_args=(--news_rank_col "$NEWS_RANK_COL")
  fi
  if [[ "$DRY_RUN" == "1" ]]; then
    dry_run_args=(--dry_run)
  fi
  case "$backbone" in
    TimeMixer)
      backbone_args=(--down_sampling_method avg --down_sampling_layers 1 --down_sampling_window 2)
      ;;
    SegRNN)
      backbone_args=(--seg_len 1)
      ;;
  esac
  local -a optional_args=()
  if [[ ${#rank_args[@]} -gt 0 ]]; then
    optional_args+=("${rank_args[@]}")
  fi
  if [[ ${#extra_decay_flag[@]} -gt 0 ]]; then
    optional_args+=("${extra_decay_flag[@]}")
  fi
  if [[ ${#dry_run_args[@]} -gt 0 ]]; then
    optional_args+=("${dry_run_args[@]}")
  fi
  if [[ ${#backbone_args[@]} -gt 0 ]]; then
    optional_args+=("${backbone_args[@]}")
  fi

  echo
  echo "===== News $mode | $ticker | $news_model | sl=$sl pl=$pl | CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset} ====="
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
    --news_selection_mode "$NEWS_SELECTION_MODE" \
    --news_rank_recency_weight "$NEWS_RANK_RECENCY_WEIGHT" \
    --batch_size "$BATCH_SIZE" \
    --learning_rate 1e-3 \
    --train_epochs "$TRAIN_EPOCHS" \
    --patience "$PATIENCE" \
    --itrs "$ITRS" \
    --num_workers "$NUM_WORKERS" \
    --disable_progress \
    "${optional_args[@]}" \
    --result_path "$RESULT_DIR" \
    --des "${news_model}_${mode}_g" | tee "$LOG_DIR/${stem}_sl${sl}_pl${pl}_${news_model}_${mode}.log"
}

for ticker in "${TICKERS[@]}"; do
  ticker="$(echo "$ticker" | tr '[:lower:]' '[:upper:]' | xargs)"
  [[ -z "$ticker" ]] && continue

  IFS='|' read -r PRICE_FILE NEWS_FILE <<< "$(ticker_files "$ticker")"
  PRICE_STEM="${PRICE_FILE%.csv}"

  if [[ ! -f "$DATA_DIR/$PRICE_FILE" ]]; then
    echo "Missing price file: $DATA_DIR/$PRICE_FILE" >&2
    if [[ "$SKIP_MISSING_TICKERS" == "1" ]]; then
      echo "Skipping ticker $ticker because price file is missing." >&2
      continue
    fi
    exit 1
  fi
  if [[ ! -f "$DATA_DIR/$NEWS_FILE" ]]; then
    echo "Missing news file: $DATA_DIR/$NEWS_FILE" >&2
    if [[ "$SKIP_MISSING_TICKERS" == "1" ]]; then
      echo "Skipping ticker $ticker because news file is missing." >&2
      continue
    fi
    exit 1
  fi

  for setting in "${SETTINGS[@]}"; do
    read -r SL LL PL <<< "$setting"
    for backbone in "${MODEL_NAMES[@]}"; do
      backbone="$(echo "$backbone" | xargs)"
      [[ -z "$backbone" ]] && continue
      NEWS_MODEL="$(news_model_name "$backbone")"

      if [[ "$RUN_BASELINE" == "1" ]]; then
        launch_on_next_gpu \
          "$ticker $backbone price-only sl=$SL pl=$PL" \
          run_price_only "$ticker" "$backbone" "$PRICE_FILE" "$PRICE_STEM" "$SL" "$LL" "$PL"
      fi

      if [[ "$RUN_PLAIN" == "1" ]]; then
        launch_on_next_gpu \
          "$ticker $NEWS_MODEL plain sl=$SL pl=$PL" \
          run_news_adapter "$ticker" "$backbone" "$NEWS_MODEL" "$PRICE_FILE" "$NEWS_FILE" "$PRICE_STEM" "$SL" "$LL" "$PL" plain
      fi

      if [[ "$RUN_DECAY" == "1" ]]; then
        launch_on_next_gpu \
          "$ticker $NEWS_MODEL decay sl=$SL pl=$PL" \
          run_news_adapter "$ticker" "$backbone" "$NEWS_MODEL" "$PRICE_FILE" "$NEWS_FILE" "$PRICE_STEM" "$SL" "$LL" "$PL" decay
      fi
    done
  done
done

wait_for_all_jobs

echo
echo "Done. Results: $RESULT_DIR"
echo "Logs: $LOG_DIR"
