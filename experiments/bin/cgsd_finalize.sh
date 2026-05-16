#!/usr/bin/env bash
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/cgsd_env.sh"

ROUND="${ROUND:?set ROUND to the final model round, e.g. ROUND=3}"

python scripts/cgsd_finalize.py \
  --output_dir "$OUT" \
  --round_index "$ROUND" \
  --cache_policy "$CACHE_POLICY"

printf 'finalized: round=%s summary=%s\n' "$ROUND" "$OUT/cgsd_summary.json"
