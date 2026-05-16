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
ALPHA="${ALPHA:-0.07}"
TEMP="${TEMP:-15}"
CACHE_POLICY="${CACHE_POLICY:-reuse}"

INPUT_DIR="${INPUT_DIR:-$EXPERIMENTS_DIR/inputs/$DATASET}"
RUN_ROOT="${RUN_ROOT:-$EXPERIMENTS_DIR/runs/$DATASET}"
OUT="${OUT:-$RUN_ROOT/$RUN_NAME}"
DATA="${DATA:-$INPUT_DIR/data.jsonl}"
EMB="${EMB:-$INPUT_DIR/embeddings.npy}"
TEACHER="${TEACHER:-}"

EXTRA_PREDICT_ARGS=()
if [[ -n "$TEACHER" ]]; then
  EXTRA_PREDICT_ARGS+=(--teacher_labels_path "$TEACHER")
fi

mkdir -p "$OUT"
cd "$PROJECT_ROOT"
