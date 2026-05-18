#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXPERIMENTS_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
PROJECT_ROOT="$(cd "$EXPERIMENTS_DIR/.." && pwd)"

DATASET="${DATASET:-lrobench}"
RUN_NAME="${RUN_NAME:-exp1_seed${SEED:-1}}"
MODEL="${MODEL:-model/qwen3-0.6b}"
SEED="${SEED:-1}"
DIM="${DIM:-2560}"
ALPHA="${ALPHA:-0.1}"
TEMP="${TEMP:-15}"
CACHE_POLICY="${CACHE_POLICY:-reuse}"

INPUT_DIR="${INPUT_DIR:-$EXPERIMENTS_DIR/inputs/$DATASET}"
RUN_ROOT="${RUN_ROOT:-$EXPERIMENTS_DIR/runs/$DATASET}"
OUT="${OUT:-$RUN_ROOT/$RUN_NAME}"
DATA="${DATA:-$INPUT_DIR/data.jsonl}"
EMB="${EMB:-$INPUT_DIR/embeddings.npy}"
TEACHER="${TEACHER:-}"

EXTRA_PREDICT_ARGS=()
VLLM_BASE_URL="${VLLM_BASE_URL:-http://127.0.0.1:18021/v1}"
VLLM_PARALLEL_REQUESTS="${VLLM_PARALLEL_REQUESTS:-1024}"
VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-40960}"
VLLM_MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS:-4096}"
VLLM_MAX_NUM_BATCHED_TOKENS="${VLLM_MAX_NUM_BATCHED_TOKENS:-524288}"
VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.98}"
VLLM_TOP_LOGPROBS="${VLLM_TOP_LOGPROBS:-20}"
VLLM_START_SERVER="${VLLM_START_SERVER:-1}"
if [[ "$VLLM_START_SERVER" == "1" ]]; then
  EXTRA_PREDICT_ARGS+=(--start_server)
fi
EXTRA_PREDICT_ARGS+=(
  --base_url "$VLLM_BASE_URL"
  --parallel_requests "$VLLM_PARALLEL_REQUESTS"
  --max_model_len "$VLLM_MAX_MODEL_LEN"
  --max_num_seqs "$VLLM_MAX_NUM_SEQS"
  --max_num_batched_tokens "$VLLM_MAX_NUM_BATCHED_TOKENS"
  --gpu_memory_utilization "$VLLM_GPU_MEMORY_UTILIZATION"
  --top_logprobs "$VLLM_TOP_LOGPROBS"
  --temperature 0
  --max_tokens 1
)
if [[ -n "$TEACHER" ]]; then
  EXTRA_PREDICT_ARGS+=(--teacher_labels_path "$TEACHER")
fi

mkdir -p "$OUT"
cd "$PROJECT_ROOT"
