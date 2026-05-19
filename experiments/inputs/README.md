# Experiment Inputs

This directory is the durable input surface for experiments. It should contain
source data, embeddings, reusable split definitions, and round0 caches that are
expensive or important to reproduce. It should not contain LoRA adapters,
checkpoints, transient run logs, or per-run prediction dumps.

All task data must use `id/query/document/groundtruth` JSONL rows.

## Current Layout

```text
lrobench/
  data.jsonl
  embeddings.npy
  embeddings.ids.jsonl
  embeddings.meta.json

fever/
  data.jsonl
  embeddings.npy
  embeddings.ids.jsonl
  embeddings.meta.json
  cgsd_split_ids.json
  round_0/                         # FEVER 0.6B round0 baseline cache
  qwen17b_alpha010_t1_seed1/
    round_0/                       # FEVER 1.7B, T=1 round0 CRC cache
  qwen17b_alpha010_t15_seed1/
    round_0/                       # FEVER 1.7B, T=15 round0 prediction/CRC cache
  recalibrated_T1/
  recalibrated_T10/
  balanced_lora_subsets_seed1/     # reusable FEVER subset definitions
```

`embeddings.npy` must have a matching `embeddings.ids.jsonl` sidecar and must
cover every `id` in `data.jsonl` when running `scripts/cgsd_prepare.py`.

## Retention Policy

- Keep FEVER original 0.6B and 1.7B round0 data/caches.
- Keep source `data.jsonl`, embeddings, metadata, and reusable split/subset files.
- Keep result summaries in `experiments/runs/` as CSV/Markdown.
- Remove generated adapters, checkpoints, JSONL prediction dumps, and old one-off
  generated subsets unless they are round0 source caches.

Current status:

- `lrobench` is ready for CGSD with `DIM=2560`.
- `fever` has source JSONL, embeddings, and preserved 0.6B/1.7B round0 caches.
