#!/usr/bin/env bash
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/cgsd_env.sh"

STRATEGY="${STRATEGY:?set STRATEGY to random, uncertainty, k-center, or defer-random}"
ROUND="${ROUND:-0}"
BUDGET="${BUDGET:-500}"
TEACHER_BETA="${TEACHER_BETA:-1}"
SOURCE_OUT="${SOURCE_OUT:-$OUT}"

mkdir -p "$OUT"
if [[ "$SOURCE_OUT" != "$OUT" ]]; then
  cp "$SOURCE_OUT/cgsd_split_ids.json" "$OUT/cgsd_split_ids.json"
fi

BASELINE_EXTRA_ARGS=(
  --pool_crc_predictions_path "$SOURCE_OUT/round_$ROUND/pool_crc_predictions.jsonl"
)
if [[ "$STRATEGY" == "k-center" ]]; then
  BASELINE_EXTRA_ARGS+=(--embeddings_path "$EMB" --embedding_dim "$DIM")
fi

python scripts/cgsd_make_baseline_rows.py \
  --output_dir "$OUT" \
  --round_index "$ROUND" \
  --strategy "$STRATEGY" \
  --budget "$BUDGET" \
  --split_ids_path "$SOURCE_OUT/cgsd_split_ids.json" \
  --pool_student_predictions_path "$SOURCE_OUT/round_$ROUND/pool_student_predictions.jsonl" \
  --teacher_beta "$TEACHER_BETA" \
  --seed "$SEED" \
  --cache_policy "$CACHE_POLICY" \
  "${BASELINE_EXTRA_ARGS[@]}"

printf 'baseline rows complete: strategy=%s out=%s\n' "$STRATEGY" "$OUT"
printf 'train rows: %s\n' "$OUT/cgsd_train_rows.jsonl"
