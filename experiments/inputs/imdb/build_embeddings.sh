#!/usr/bin/env bash
set -euo pipefail

ROOT="/teamspace/studios/this_studio/LLM_layer_test"
cd "$ROOT"

MODEL_PATH="${MODEL_PATH:-/teamspace/studios/this_studio/model/qwen3-4b-embedding}"
BACKEND="${BACKEND:-vllm}"
REQUEST_BATCH_SIZE="${REQUEST_BATCH_SIZE:-256}"
FLUSH_ROWS="${FLUSH_ROWS:-1024}"
MAX_LENGTH="${MAX_LENGTH:-4096}"
TORCH_DTYPE="${TORCH_DTYPE:-bfloat16}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.92}"
QIDS="${QIDS:-1 2}"
OVERWRITE="${OVERWRITE:-0}"

overwrite_args=()
if [[ "${OVERWRITE}" == "1" ]]; then
  overwrite_args+=(--overwrite)
fi

for qid in ${QIDS}; do
  input_dir="experiments/inputs/imdb/query_id_${qid}"
  python scripts/cgsd_build_embeddings.py \
    --data_path "${input_dir}/data.jsonl" \
    --output_path "${input_dir}/embeddings.npy" \
    --ids_path "${input_dir}/embeddings.ids.jsonl" \
    --meta_path "${input_dir}/embeddings.meta.json" \
    --model_path "${MODEL_PATH}" \
    --backend "${BACKEND}" \
    --request_batch_size "${REQUEST_BATCH_SIZE}" \
    --flush_rows "${FLUSH_ROWS}" \
    --max_length "${MAX_LENGTH}" \
    --torch_dtype "${TORCH_DTYPE}" \
    --tensor_parallel_size "${TENSOR_PARALLEL_SIZE}" \
    --gpu_memory_utilization "${GPU_MEMORY_UTILIZATION}" \
    --query_field query \
    --document_field document \
    --id_field id \
    --mode document \
    "${overwrite_args[@]}"
done
