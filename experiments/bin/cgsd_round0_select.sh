#!/usr/bin/env bash
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/cgsd_env.sh"

ROUND=0
BUDGET="${BUDGET:-250}"
TEACHER_BETA="${TEACHER_BETA:-1}"
N_CALIBRATION="${N_CALIBRATION:-200}"
N_FINAL_CALIBRATION="${N_FINAL_CALIBRATION:-0}"
SELECTION_METHOD="${SELECTION_METHOD:-crc-error-mass}"
ACCEPT_STRATEGY="${ACCEPT_STRATEGY:-random}"
DEFER_STRATEGY="${DEFER_STRATEGY:-random}"
SELECTION_BUFFER_MULTIPLIER="${SELECTION_BUFFER_MULTIPLIER:-1}"
TEACHER_CONFIDENCE_FILTER="${TEACHER_CONFIDENCE_FILTER:-0}"

EXTRA_SELECT_ARGS=()
if [[ "$TEACHER_CONFIDENCE_FILTER" == "1" || "$TEACHER_CONFIDENCE_FILTER" == "true" ]]; then
  EXTRA_SELECT_ARGS+=(--teacher_confidence_filter)
fi

python scripts/cgsd_prepare.py \
  --data_path "$DATA" \
  --embeddings_path "$EMB" \
  --embedding_dim "$DIM" \
  --output_dir "$OUT" \
  --n_calibration "$N_CALIBRATION" \
  --n_final_calibration "$N_FINAL_CALIBRATION" \
  --seed "$SEED" \
  --cache_policy "$CACHE_POLICY"

python scripts/cgsd_predict_vllm_openai.py \
  --output_dir "$OUT" \
  --round_index "$ROUND" \
  --model_path "$MODEL" \
  --data_path "$DATA" \
  --split_ids_path "$OUT/cgsd_split_ids.json" \
  --cache_policy "$CACHE_POLICY" \
  "${EXTRA_PREDICT_ARGS[@]}"

python scripts/cgsd_calibrate.py \
  --output_dir "$OUT" \
  --round_index "$ROUND" \
  --temperature "$TEMP" \
  --alpha "$ALPHA" \
  --embeddings_path "$EMB" \
  --cache_policy "$CACHE_POLICY"

python scripts/cgsd_select.py \
  --output_dir "$OUT" \
  --round_index "$ROUND" \
  --split_ids_path "$OUT/cgsd_split_ids.json" \
  --embeddings_path "$EMB" \
  --embedding_dim "$DIM" \
  --budget "$BUDGET" \
  --teacher_beta "$TEACHER_BETA" \
  --selection_method "$SELECTION_METHOD" \
  --accept_strategy "$ACCEPT_STRATEGY" \
  --defer_strategy "$DEFER_STRATEGY" \
  --selection_buffer_multiplier "$SELECTION_BUFFER_MULTIPLIER" \
  --cache_policy "$CACHE_POLICY" \
  "${EXTRA_SELECT_ARGS[@]}"

printf 'round0 select complete: %s method=%s accept=%s defer=%s\n' "$OUT" "$SELECTION_METHOD" "$ACCEPT_STRATEGY" "$DEFER_STRATEGY"
printf 'check: %s\n' "$OUT/round_0/round_summary.json"
printf 'selected rows: %s\n' "$OUT/cgsd_train_rows.jsonl"
