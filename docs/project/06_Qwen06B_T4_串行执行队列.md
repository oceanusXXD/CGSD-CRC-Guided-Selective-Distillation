# Qwen3-0.6B T4 Serial Execution Queue

> Status: planned. This document defines task boundaries and execution order;
> it is not an experimental result report.

## 1. Global Rules

Only one GPU-intensive stage may run at a time. Before starting a stage, check
that the previous stage has written its required report and that the GPU is not
occupied by another user process. Do not use a process-local memory fraction as
a substitute for serial execution: a T4 has no MIG partitioning and concurrent
model jobs contend for compute even when memory is available.

The shared base model may be read from `/home/ubuntu/models/qwen3-0.6b`, but a
trained adapter, selected ids, oracle store, score cache, run matrix, or report
must never be reused across task roots unless the run manifest explicitly names
it as an immutable input.

Every completed run records the config snapshot, source-data SHA256, git state,
hardware, training tokens, selected ids, metrics, and failure reason. One run
means one dataset, method, budget, and training seed.

## 2. Isolated Layout

Generated artifacts are kept under these mutually exclusive roots:

```text
experiments/runs/binary_qwen06b_t4_v1/<dataset>/
experiments/reports/binary_qwen06b_t4_v1/<dataset>/
experiments/runs/multiclass_qwen06b_t4_v1/<dataset>/
experiments/reports/multiclass_qwen06b_t4_v1/<dataset>/
experiments/runs/dpo_qwen06b_t4_v1/helpsteer2_preference/
experiments/reports/dpo_qwen06b_t4_v1/helpsteer2_preference/
```

Do not write new main-study artifacts to an existing `current/`, `cpu_pilot/`,
or smoke-test directory. Copy the exact configuration into the task root before
the first selection or training run.

All Qwen3-0.6B LoRA/DPO runs use the T4-compatible recipe already validated by
the initial HelpSteer2 adapter: `float16`, `fp16`, batch size 1, gradient
accumulation 8, gradient checkpointing, LoRA rank 8, LoRA alpha 16, and maximum
length 2048. Any change creates a new task version rather than mutating `v1`.

## 3. Serial Queue

### S0: Shared Readiness, No Training

1. Wait for GPU availability and record `nvidia-smi` output in each task root.
2. Verify native binary source manifests with
   `scripts/download_binary_benchmarks.py`; do not create labels or change
   source revisions.
3. Freeze the five training seeds, task-specific budgets, model revision, and
   evaluation inputs in a config snapshot for each task root.
4. Run unit tests and the code-cleanliness test before any GPU stage.

**Exit artifact:** one immutable config snapshot and readiness report per task.

### S1: Binary Classification

Datasets run in this order: IMDb, PAWS `labeled_final`, TweetEval `hate`.
Inputs remain in their official source splits under `experiments/inputs/binary/`.
The final test split must never enter selection or hyperparameter decisions.

For each dataset, finish the full sequence before moving to the next dataset:

1. Materialize a selector-safe training/active-pool split from the native
   training split and record the split manifest.
2. Score the active pool without reading `groundtruth`.
3. Materialize Random, Entropy, and Margin selections at the frozen budget.
4. Train and evaluate the five training seeds with isolated adapters.
5. Produce selection, budget, and run-level audits.

**Exit artifact:** `binary_run_records.jsonl`, selection audit, fairness report,
and a per-dataset decision report. Do not start S2 while any binary run is
unreported or failed without a recorded reason.

### S2: Multiclass Classification

Datasets run in this order: AG News, then TREC. Use separate fixed splits and
separate adapters for each dataset. Methods are Random, Entropy, BADGE, GALAXY,
Entropy+DCMS, and BADGE+DCMS.

For each dataset: score once, cache only that dataset's logits/representations,
run selection, train/evaluate the frozen seeds, then complete the
class-intercept and DCMS audits before proceeding. The AG News result must be
audited before TREC begins; TREC is a replication, not a continuation of the
AG News run directory.

**Exit artifact:** a dataset-specific run record collection, class-intercept
response report, and paired method comparison.

### S3: DPO

Use only the HelpSteer2 fixed split and its isolated DPO root. Rebuild a new
run matrix and execution manifest from the Qwen3-0.6B T4 config; never modify
or reuse the CPU-pilot manifest. First complete Random and ActiveDPO+DCMS for
the frozen initial seeds, audit them, then execute the six-method matrix:
Random, Reward Margin, APL, ActiveDPO, APL+DCMS, and ActiveDPO+DCMS.

Every stage follows the existing manifest order: selection, reveal, training,
evaluation, and run-summary. The next stage starts only after the prior report
exists and the preflight remains valid.

**Exit artifact:** execution-status report, validated DPO run pack, paired
metric comparison, and preference claim audit.

### S4: Cross-Task Packaging, No New Training

Aggregate only completed run records. Keep binary, multiclass, and DPO tables
separate; no pooled average across task types is valid. Record zero results and
failed runs rather than deleting them. A paper claim may compare tasks only in
the discussion after each task's own metrics and protocol have passed audit.

## 4. Code-Cleanliness Gate

Before each new serial stage, run:

```bash
python3 -m unittest tests.test_binary_benchmark_data tests.test_codebase_cleanliness
```

The binary source downloader is a public entrypoint. Its output manifests,
not transient console logs, are the source-of-truth for data revision and
label provenance.
