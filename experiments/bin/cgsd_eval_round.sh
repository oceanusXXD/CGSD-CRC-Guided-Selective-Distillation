#!/usr/bin/env bash
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/cgsd_env.sh"

ROUND="${ROUND:?set ROUND to the model round to evaluate, e.g. ROUND=1}"
PREV_ROUND="${PREV_ROUND:-$((ROUND - 1))}"

python scripts/cgsd_predict_vllm_openai.py \
  --output_dir "$OUT" \
  --round_index "$ROUND" \
  --model_path "$MODEL" \
  --data_path "$DATA" \
  --split_ids_path "$OUT/cgsd_split_ids.json" \
  --checkpoint_dir "$OUT/round_$ROUND/model" \
  --cache_policy "$CACHE_POLICY" \
  "${EXTRA_PREDICT_ARGS[@]}"

python scripts/cgsd_calibrate.py \
  --output_dir "$OUT" \
  --round_index "$ROUND" \
  --temperature "$TEMP" \
  --alpha "$ALPHA" \
  --embeddings_path "$EMB" \
  --previous_round_summary_path "$OUT/round_$PREV_ROUND/round_summary.json" \
  --previous_selection_summary_path "$OUT/round_$PREV_ROUND/selection_summary.json" \
  --train_rows_path "$OUT/cgsd_train_rows.jsonl" \
  --cache_policy "$CACHE_POLICY"

printf 'evaluation complete: round=%s summary=%s\n' "$ROUND" "$OUT/round_$ROUND/round_summary.json"
