#!/usr/bin/env bash
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/cgsd_env.sh"

ROUND="${ROUND:?set ROUND to the model round to train, e.g. ROUND=1}"
LORA_R="${LORA_R:-1}"
LORA_TARGET_MODULES="${LORA_TARGET_MODULES:-qv}"
LORA_LAYER_SCOPE="${LORA_LAYER_SCOPE:-all}"
EPOCHS="${EPOCHS:-3}"
LR="${LR:-2e-4}"
BATCH_SIZE="${BATCH_SIZE:-1}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-16}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-4}"
MAX_LENGTH="${MAX_LENGTH:-512}"
INIT_ADAPTER_PATH="${INIT_ADAPTER_PATH:-}"

EXTRA_TRAIN_ARGS=()
if [[ -n "$INIT_ADAPTER_PATH" ]]; then
  EXTRA_TRAIN_ARGS+=(--init_adapter_path "$INIT_ADAPTER_PATH")
fi

python scripts/cgsd_train_round.py \
  --output_dir "$OUT" \
  --round_index "$ROUND" \
  --model_path "$MODEL" \
  --data_path "$DATA" \
  --split_ids_path "$OUT/cgsd_split_ids.json" \
  --train_rows_path "$OUT/cgsd_train_rows.jsonl" \
  --lora_r "$LORA_R" \
  --lora_target_modules "$LORA_TARGET_MODULES" \
  --lora_layer_scope "$LORA_LAYER_SCOPE" \
  --epochs "$EPOCHS" \
  --lr "$LR" \
  --batch_size "$BATCH_SIZE" \
  --gradient_accumulation_steps "$GRADIENT_ACCUMULATION_STEPS" \
  --eval_batch_size "$EVAL_BATCH_SIZE" \
  --max_length "$MAX_LENGTH" \
  --cache_policy "$CACHE_POLICY" \
  "${EXTRA_TRAIN_ARGS[@]}"

printf 'training complete: round=%s checkpoint=%s\n' "$ROUND" "$OUT/round_$ROUND/model"
