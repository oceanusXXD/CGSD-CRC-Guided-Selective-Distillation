#!/usr/bin/env bash
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/cgsd_env.sh"

ROUND="${ROUND:?set ROUND to the prediction round to select from, e.g. ROUND=1}"
BUDGET="${BUDGET:?set BUDGET for this selection round}"
TEACHER_BETA="${TEACHER_BETA:-1}"
SELECTION_METHOD="${SELECTION_METHOD:-crc-error-mass}"
ACCEPT_STRATEGY="${ACCEPT_STRATEGY:-random}"
DEFER_STRATEGY="${DEFER_STRATEGY:-random}"
SELECTION_BUFFER_MULTIPLIER="${SELECTION_BUFFER_MULTIPLIER:-1}"
TEACHER_CONFIDENCE_FILTER="${TEACHER_CONFIDENCE_FILTER:-0}"

EXTRA_SELECT_ARGS=()
if [[ "$TEACHER_CONFIDENCE_FILTER" == "1" || "$TEACHER_CONFIDENCE_FILTER" == "true" ]]; then
  EXTRA_SELECT_ARGS+=(--teacher_confidence_filter)
fi

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

printf 'selection complete: round=%s method=%s budget=%s accept=%s defer=%s out=%s\n' "$ROUND" "$SELECTION_METHOD" "$BUDGET" "$ACCEPT_STRATEGY" "$DEFER_STRATEGY" "$OUT"
