#!/usr/bin/env bash
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/cgsd_env.sh"

ROUND="${ROUND:?set ROUND to the prediction round to select from, e.g. ROUND=1}"
BUDGET="${BUDGET:?set BUDGET for this selection round}"
DELTA="${DELTA:-0.1}"
TEACHER_BETA="${TEACHER_BETA:-1}"
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

printf 'selection complete: round=%s budget=%s out=%s\n' "$ROUND" "$BUDGET" "$OUT"
