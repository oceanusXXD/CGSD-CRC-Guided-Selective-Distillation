#!/usr/bin/env bash
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/cgsd_env.sh"

ROUND="${ROUND:?set ROUND to the prediction round to select from, e.g. ROUND=1}"
BUDGET="${BUDGET:?set BUDGET for this selection round}"
TEACHER_BETA="${TEACHER_BETA:-1}"

python scripts/cgsd_select.py \
  --output_dir "$OUT" \
  --round_index "$ROUND" \
  --split_ids_path "$OUT/cgsd_split_ids.json" \
  --embeddings_path "$EMB" \
  --embedding_dim "$DIM" \
  --budget "$BUDGET" \
  --teacher_beta "$TEACHER_BETA" \
  --cache_policy "$CACHE_POLICY"

printf 'selection complete: round=%s budget=%s out=%s\n' "$ROUND" "$BUDGET" "$OUT"
