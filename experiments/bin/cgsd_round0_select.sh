#!/usr/bin/env bash
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/cgsd_env.sh"

ROUND=0
BUDGET="${BUDGET:-250}"
DELTA="${DELTA:-0.1}"
TEACHER_BETA="${TEACHER_BETA:-1}"
N_CALIBRATION="${N_CALIBRATION:-200}"
N_FINAL_CALIBRATION="${N_FINAL_CALIBRATION:-0}"
EASY_ANCHOR_RATIO="${EASY_ANCHOR_RATIO:-}"
ANCHOR_COUNT="${ANCHOR_COUNT:-}"
BAND_RATIOS="${BAND_RATIOS:-}"
SELECT_EXTRA_ARGS=()
if [[ -n "$EASY_ANCHOR_RATIO" ]]; then
  SELECT_EXTRA_ARGS+=(--easy_anchor_ratio "$EASY_ANCHOR_RATIO")
fi
if [[ -n "$ANCHOR_COUNT" ]]; then
  SELECT_EXTRA_ARGS+=(--anchor_count "$ANCHOR_COUNT")
fi
if [[ -n "$BAND_RATIOS" ]]; then
  SELECT_EXTRA_ARGS+=(--band_ratios "$BAND_RATIOS")
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

python scripts/cgsd_predict.py \
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
  --cache_policy "$CACHE_POLICY"

python scripts/cgsd_select.py \
  --output_dir "$OUT" \
  --round_index "$ROUND" \
  --split_ids_path "$OUT/cgsd_split_ids.json" \
  --embeddings_path "$EMB" \
  --embedding_dim "$DIM" \
  --budget "$BUDGET" \
  --delta "$DELTA" \
  --teacher_beta "$TEACHER_BETA" \
  --cache_policy "$CACHE_POLICY" \
  "${SELECT_EXTRA_ARGS[@]}"

printf 'round0 select complete: %s\n' "$OUT"
printf 'check: %s\n' "$OUT/round_0/round_summary.json"
printf 'selected rows: %s\n' "$OUT/cgsd_train_rows.jsonl"
