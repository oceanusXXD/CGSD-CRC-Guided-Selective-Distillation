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
binary/
  codebase_q2/
    source_train.jsonl
    source_test.jsonl
    source_manifest.json
    protocol_manifest.json
  twitter_hate_q1/
    source_train.jsonl
    source_test.jsonl
    source_manifest.json
    protocol_manifest.json
  imdb/
    train.jsonl
    test.jsonl
    source_manifest.json
  paws_labeled_final/
    train.jsonl
    validation.jsonl
    test.jsonl
    source_manifest.json
  tweeteval_hate/
    train.jsonl
    validation.jsonl
    test.jsonl
    source_manifest.json
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
- Native-label binary inputs are generated locally under `binary/` by
  `scripts/download_binary_benchmarks.py`. Each dataset directory contains the
  official split boundaries in the repository's `id/query/document/groundtruth`
  schema and a pinned-source manifest with output hashes and label counts.
- Local single-query JSONL sources can be made into separate, non-legacy binary
  benchmarks with `scripts/prepare_query_binary_benchmark.py`. The importer
  removes prior parsed-answer fields, deduplicates exact documents before
  splitting, and records a document-disjoint fixed holdout. These datasets do
  not recreate historical aggregate results or use an official source split.
- Current preference fixed-pool artifacts under `preference/helpsteer2_preference/`
  are generated from `benchmarks/helpsteer2_preference/train.jsonl` with seed
  `20260712`; they cover selector-safe active rows, oracle labels, A/B swaps,
  and a fixed seed/active/heldout/test split manifest.
- MIAS preference selection consumes separate frozen `h(prompt,response_A)` and
  `h(prompt,response_B)` matrices. Package them with
  `scripts/prepare_mias_features.py`; the selector then forms the antisymmetric
  difference and never reads active-pool oracle labels. The seed feature file
  must cover all initially revealed pairs, including ties: MIAS learns A/B
  direction from non-ties and an order-invariant non-tie gate from all revealed
  seed labels. Candidate feature rows must also carry exact response
  completion-token counts for cost normalization.
- Large FEVER, IMDb, and other binary-task input trees have been removed from
  the active tree. Their original sample-level records are unavailable and must
  be recovered before a historical setting is used again.
