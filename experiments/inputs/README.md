# Experiment Inputs

This directory is the durable input surface for MIAS/DCMS experiments. It should
contain source data, reusable split definitions, compact benchmark smoke inputs,
and round-0 caches that are expensive or important to reproduce. It should not
contain LoRA adapters, checkpoints, transient run logs, or per-run prediction
dumps.

Task data must use stable ids and task-appropriate selector-safe fields. Legacy
binary selective-distillation rows use `id/query/document/groundtruth` JSONL.
Current multiclass and preference benchmark rows are prepared through
`scripts/benchmark_pipeline.py`, `scripts/prepare_multiclass_splits.py`, or
`scripts/prepare_preference_pool.py`.

## Current Layout

```text
benchmarks/
  ag_news/
    train.jsonl
    test.jsonl
    dataset_summary.json
    qwen3_0.6b_*.jsonl
    qwen3_0.6b_*.summary.json
  dbpedia_14/
    train_5000_per_class.jsonl
    test.jsonl
    dataset_summary.json
    qwen3_0.6b_*.jsonl
    qwen3_0.6b_*.summary.json
  helpsteer2_preference/
    train.jsonl
    val.jsonl
    dataset_summary.json
    qwen3_0.6b_*.jsonl
    qwen3_0.6b_*.summary.json
preference/
  helpsteer2_preference/
    active_pool.jsonl
    oracle_store.json
    pool_summary.json
    split_manifest.json
    split_summary.json
    swap_manifest.json
  smoke_shift_gate.json
```

If a legacy binary input directory is restored locally for re-audit,
`embeddings.npy` must have a matching `embeddings.ids.jsonl` sidecar and must
cover every `id` in `data.jsonl` when running `scripts/prepare.py`.

## Retention Policy

- Keep compact benchmark smoke inputs and summaries needed by repository tests.
- Keep source JSONL, metadata, and reusable split/subset files when they are
  small enough to review.
- Keep result summaries in `experiments/runs/` or `experiments/reports/` as
  CSV/Markdown.
- Remove generated adapters, checkpoints, large JSONL prediction dumps, and
  old one-off generated subsets unless they are required round-0 source caches.

## Current Status

- Current tracked inputs are under `benchmarks/` for AG News, DBPedia-14, and
  HelpSteer2-Preference smoke workflows.
- Current preference fixed-pool artifacts under `preference/helpsteer2_preference/`
  are generated from `benchmarks/helpsteer2_preference/train.jsonl` with seed
  `20260712`; they cover selector-safe active rows, oracle labels, A/B swaps,
  and a fixed seed/active/heldout/test split manifest.
- Large FEVER, IMDb, and other binary-task input trees have been removed from
  the active tree. Their compact evidence is retained in
  `experiments/reports/binary_legacy/` and related report files.
