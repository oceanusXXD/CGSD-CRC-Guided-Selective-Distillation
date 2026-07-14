# MIAS/DCMS Research Repository

This repository is organized around the MIAS/DCMS AAAI mainline:

- MIAS: Model-Induced Acquisition Shift.
- DCMS: Distribution-Constrained Model-Guided Selection.

The active research direction is documented in `docs/project/01_最终任务_算法与AAAI论文路径_MIAS_DCMS.md`. The development and acceptance path is documented in `docs/project/02_开发与实验执行文档_MIAS_DCMS.md` and `docs/project/03_开发与实验验收清单_MIAS_DCMS.md`.

Previous CRC/PCSS-centered binary-task drafts are archived under `docs/archive/emnlp_legacy/`. Binary-task tables and figures are retained only as legacy evidence under `experiments/reports/binary_legacy/`; they are not the active package or experiment tree.

## Repository Layout

- `mias_dcms/`: active reusable Python package for the MIAS/DCMS mainline.
- `scripts/`: active command-line entrypoints.
- `tests/`: regression and cleanliness tests for the active codebase.
- `configs/`: frozen protocol and experiment configuration files.
- `docs/project/`: MIAS/DCMS task, execution, status, and acceptance documents.
- `docs/paper/`: active paper draft materials.
- `docs/archive/`: superseded paper drafts and narratives.
- `experiments/inputs/`: durable input surfaces such as source data, split ids, embeddings, and reusable caches.
- `experiments/runs/`: local or compact run outputs.
- `experiments/reports/`: compact reports and provenance.
- `experiments/reports/binary_legacy/`: archived binary-task tables, figures, and summary.
- `resources/`: supporting non-code resources.

Retired top-level legacy trees are intentionally not part of the active repository layout.

## Current Status

The current migration and development status is tracked in `docs/project/04_当前迁移开发验收状态_MIAS_DCMS.md`.

At the codebase level, the active tree currently provides:

- protocol freeze metadata in `configs/mias_dcms_freeze.v1.json`;
- selector-safe sample and run records;
- MIAS selection auditing metrics;
- DCMS small-batch solving, slack search, robust intervals, and utility-coverage frontier tools;
- fixed-pool preference and multiclass split helpers;
- frozen prompt-cluster assignment from precomputed embeddings for DPO observable groups and APL metadata;
- preference policy/reference logprob auditing with implicit margin checks;
- selector score sanity auditing, including top-budget reproducibility, score-length correlation, and A/B swap deltas;
- first-round preference acquisition auditing across length, source, prompt cluster, and Random reference coverage;
- DPO-side intervention auditing for length-gamma response, selector replacement, and A/B position propensity;
- DPO preference evaluation metrics, including worst-group, length-controlled win rate, and capability regression;
- selector-safe Reward Margin, APL, and fixed-pool ActiveDPO preference baseline scoring;
- preference baseline-to-DCMS candidate preparation for APL+DCMS and ActiveDPO+DCMS runs;
- preference acquisition run-summary generation for paper-level aggregation;
- DPO main-experiment run-matrix preflight for dataset/model/budget/seed/method coverage, planned artifact paths, and shared training/judge config hashes;
- preference/DPO experiment artifact preflight for active pool, oracle store, logprobs, split manifest, hidden-label isolation, and run-matrix alignment;
- real initial DPO policy checkpoint registration for Gate 4 evidence, with required adapter-file validation;
- DPO execution manifest generation for ordered selection, reveal, training, evaluation, and run-summary stages;
- DPO execution status auditing for completed, blocked, in-progress, and failed runs from manifest artifacts;
- a completed two-run CPU HelpSteer2 DPO pilot for Random and ActiveDPO+DCMS (seed 1);
- DPO run-pack validation for required method/seed coverage, visible failed runs, required metrics, and paper artifact manifest traceability;
- paired run-metric comparison for baseline-vs-treatment deltas, confidence intervals, and permutation tests;
- paper claim-to-evidence auditing for Gate 10 claim freeze and banned overclaim detection;
- result freeze-pack validation for results manifests, main/appendix tables, figure data, claim maps, frozen metrics, baselines, judge version, and freeze policy;
- Gate 0-10 experiment-readiness auditing so real-data blockers stay explicit instead of being inferred from green unit tests;
- intervention response statistics for monotonicity, slope confidence intervals, and visible failed settings;
- paper artifact generation for frozen Fig. 1-3 / Table 1-3 JSON payloads and freeze-pack manifests;
- baseline selectors, including Random, uncertainty-style selectors, and moment-matched Random;
- classification selectors including BADGE and GALAXY, plus selector-safe Entropy+DCMS and BADGE+DCMS wrappers;
- soft-group interval, calibration, and error-audit helpers;
- budget, cost, statistical, composition, and paper-table aggregation utilities.

The real AAAI experiment gates are not complete until the required datasets, model checkpoints, training runs, DPO runs, statistical tests, and paper figures are produced and audited.

### What can run now

| Line | Ready now | First missing input / action |
| --- | --- | --- |
| Binary re-audit | Split and selection materialization code | Recover per-setting raw/sample-level records before scoring. |
| Multiclass MIAS | AG News/TREC data, fixed-split command, scoring and diagnostics | Create a fixed split, then score the selector-safe active pool. |
| HelpSteer2 DPO | Fixed pool, initial adapter, selection logprobs, and two completed CPU pilot runs | Scale the pilot only after the full study matrix is frozen. |

The DPO status report is derived from the manifest. Whenever a config path
changes, rebuild the matrix and manifest before trusting an old status JSON.

## Setup

```bash
pip install -r requirements.txt
```

## Verification

Run the full regression suite:

```bash
python -m pytest
```

Run the file-tree and legacy-import cleanliness checks:

```bash
python -m pytest tests/test_codebase_cleanliness.py
```

## Minimal Run Paths

Run the commands below from the repository root. They are intentionally short
paths for getting a real input or pilot run started; they are not a substitute
for the full Gate 0-10 acceptance process.

### 0. Smoke the local code path

```bash
python scripts/run_mias_dcms_smoke.py \
  --output_path experiments/reports/smoke_mias_dcms.current.json
```

This is a deterministic CPU-only synthetic check. It verifies the pool,
selection, DCMS, reveal, and metric plumbing, but is not paper evidence.

### 1. Binary re-audit

The archived CSV tables are not enough for a re-audit. Start from a recovered
raw JSONL with stable ids and a label field, then create the selector-safe pool
and separate oracle store:

```bash
python scripts/run_binary_reaudit.py prepare \
  --input_path /path/to/recovered_binary_rows.jsonl \
  --output_dir experiments/inputs/binary/imdb_q1 \
  --dataset imdb_q1 \
  --seed_label_count 100 \
  --active_pool_size 1000 \
  --test_size 1000 \
  --seed 42 \
  --id_field id \
  --query_field query \
  --document_field document \
  --label_field groundtruth
```

After scoring the selector-safe pool with the frozen binary selector, use the
matching oracle and seed rows to materialize the method-specific selections:

```bash
python scripts/run_binary_reaudit.py select \
  --scored_path /path/to/binary_scored.jsonl \
  --oracle_store_path experiments/inputs/binary/imdb_q1/selection_oracle_store.json \
  --seed_train_rows_path experiments/inputs/binary/imdb_q1/seed_train_rows.jsonl \
  --output_dir experiments/runs/binary_reaudit/imdb_q1 \
  --dataset imdb_q1 \
  --model qwen3-0.6b \
  --methods Random,Entropy,Margin \
  --budget 100 \
  --seed 42 \
  --config_hash binary_reaudit_v1 \
  --evaluation_label_count 0
```

### 2. Multiclass MIAS pilot

AG News and TREC inputs are already under `experiments/inputs/benchmarks/`.
Create fixed ids first, score the same pool, then run the diagnostic selectors:

```bash
python scripts/prepare_multiclass_splits.py \
  --input_path experiments/inputs/benchmarks/ag_news/train.jsonl \
  --output_dir experiments/runs/multiclass/ag_news/splits \
  --seed 42 \
  --seed_size 400 \
  --active_size 5000 \
  --test_size 5000 \
  --label_field label

python scripts/benchmark_pipeline.py score-classification \
  --data-path experiments/runs/multiclass/ag_news/splits/active_pool.jsonl \
  --output-path experiments/runs/multiclass/ag_news/qwen06b_scored.jsonl \
  --model /home/ubuntu/models/qwen3-0.6b \
  --device-map auto \
  --save-representations

python scripts/benchmark_pipeline.py diagnose-classification \
  --scored-path experiments/runs/multiclass/ag_news/qwen06b_scored.jsonl \
  --oracle_store_path experiments/runs/multiclass/ag_news/splits/active_oracle_store.json \
  --output-dir experiments/runs/multiclass/ag_news/diagnostics \
  --budgets 100,500,1000 \
  --methods random,entropy,badge,galaxy,entropy+dcms,badge+dcms \
  --seed 42
```

The first pass is a natural-selection diagnostic. Do not call it a MIAS causal
result until the predeclared class-intercept and representation interventions
are also run.

### 3. HelpSteer2 DPO pilot

The current DPO config expects prompt-cluster metadata. Build prompt embeddings
and clusters once before regenerating the run matrix:

```bash
python scripts/build_embeddings.py \
  --data_path experiments/inputs/preference/helpsteer2_preference/selection_pool.jsonl \
  --output_path experiments/inputs/preference/helpsteer2_preference/selection_prompt_embeddings.npy \
  --ids_path experiments/inputs/preference/helpsteer2_preference/selection_prompt_embeddings.ids.jsonl \
  --model_path /home/ubuntu/models/qwen3-0.6b \
  --mode prompt \
  --query_field prompt \
  --id_field sample_id \
  --device auto \
  --torch_dtype float16

python scripts/prepare_prompt_clusters.py \
  --active_pool_path experiments/inputs/preference/helpsteer2_preference/selection_pool.jsonl \
  --embeddings_path experiments/inputs/preference/helpsteer2_preference/selection_prompt_embeddings.npy \
  --output_path experiments/inputs/preference/helpsteer2_preference/selection_prompt_clusters.jsonl \
  --cluster_count 8
```

Then rebuild the matrix and manifest from the current config; do not reuse the
older manifest after changing any input path:

```bash
python scripts/build_experiment_run_matrix.py \
  --config_path configs/dpo_run_matrix.current.json \
  --output_matrix_path experiments/runs/dpo_main/current/run_matrix.jsonl \
  --output_summary_path experiments/reports/dpo_main/current/run_matrix_summary.json

python scripts/build_dpo_execution_manifest.py \
  --run_matrix_path experiments/runs/dpo_main/current/run_matrix.jsonl \
  --output_path experiments/runs/dpo_main/current/execution_manifest.json \
  --config_path configs/dpo_run_matrix.current.json

python scripts/audit_preference_experiment_preflight.py \
  --active_pool_path experiments/inputs/preference/helpsteer2_preference/selection_pool.jsonl \
  --oracle_store_path experiments/inputs/preference/helpsteer2_preference/selection_oracle_store.json \
  --logprobs_path experiments/inputs/preference/helpsteer2_preference/selection_logprobs.jsonl \
  --split_manifest_path experiments/inputs/preference/helpsteer2_preference/split_manifest.json \
  --run_matrix_path experiments/runs/dpo_main/current/run_matrix.jsonl \
  --output_path experiments/reports/dpo_main/current/preflight.json \
  --expected_methods 'Random,Reward Margin,APL,ActiveDPO,APL+DCMS,ActiveDPO+DCMS' \
  --expected_seeds 1,2,3,4,5

python scripts/audit_dpo_execution_status.py \
  --manifest_path experiments/runs/dpo_main/current/execution_manifest.json \
  --output_path experiments/reports/dpo_main/current/execution_status.json
```

Start with one cheap stage before launching all 30 planned runs:

```bash
python scripts/run_dpo_manifest_stage.py \
  --manifest_path experiments/runs/dpo_main/current/execution_manifest.json \
  --config_path configs/dpo_run_matrix.current.json \
  --stage selection \
  --method Random \
  --seed 1 \
  --limit 1 \
  --report_path experiments/reports/dpo_main/current/random_selection_stage_report.json
```

Follow with `reveal`, `training`, and `evaluation` for that same run only after
checking the report. The current matrix is a HelpSteer2/Qwen-0.6B pilot, not a
complete paper matrix.

## Active Entrypoints

Primary MIAS/DCMS entrypoints include:

- `scripts/prepare_preference_pool.py`
- `scripts/generate_preference_logprobs.py`
- `scripts/audit_preference_logprobs.py`
- `scripts/audit_preference_acquisition.py`
- `scripts/audit_preference_selector_scores.py`
- `scripts/audit_preference_intervention.py`
- `scripts/audit_preference_evaluation.py`
- `scripts/score_preference_baselines.py`
- `scripts/select_preference_baseline.py`
- `scripts/select_preference_random.py`
- `scripts/prepare_preference_dcms_inputs.py`
- `scripts/prepare_preference_cpu_pilot.py`
- `scripts/reveal_preference_labels.py`
- `scripts/build_preference_run_summary.py`
- `scripts/prepare_preference_splits.py`
- `scripts/prepare_prompt_clusters.py`
- `scripts/prepare_preference_intervention_inputs.py`
- `scripts/build_experiment_run_matrix.py`
- `scripts/audit_preference_experiment_preflight.py`
- `scripts/build_dpo_execution_manifest.py`
- `scripts/train_preference_dpo_run.py`
- `scripts/register_initial_policy_checkpoint.py`
- `scripts/build_dpo_run_record.py`
- `scripts/run_dpo_manifest_stage.py`
- `scripts/audit_dpo_execution_status.py`
- `scripts/collect_dpo_run_records.py`
- `scripts/validate_dpo_run_pack.py`
- `scripts/audit_experiment_gate_readiness.py`
- `scripts/prepare_multiclass_splits.py`
- `scripts/prepare_soft_group_intervals.py`
- `scripts/select_dcms.py`
- `scripts/select_moment_matched_random.py`
- `scripts/audit_mias_selection.py`
- `scripts/audit_dcms_frontier.py`
- `scripts/audit_intervention_response.py`
- `scripts/audit_intervention_statistics.py`
- `scripts/audit_matched_utility.py`
- `scripts/audit_budget_report.py`
- `scripts/audit_soft_group_error.py`
- `scripts/aggregate_paper_metrics.py`
- `scripts/compare_run_metrics.py`
- `scripts/audit_paper_claims.py`
- `scripts/validate_result_freeze_pack.py`
- `scripts/build_paper_artifacts.py`
- `scripts/benchmark_pipeline.py`
- `scripts/run_mias_dcms_smoke.py` (deterministic synthetic CPU smoke; not paper evidence)
- `scripts/run_qwen_preference_smoke.py` (real Qwen model on a synthetic preference pool; not paper evidence)

Legacy binary-task utilities that remain usable have been moved to import from `mias_dcms/` and kept under `scripts/` only as public entrypoints. They must not recreate or depend on retired package trees.

## Benchmark Protocol

The current multiclass and preference benchmark workflow is described in:

```text
experiments/benchmark_shift_protocol.md
```

That protocol covers prepared benchmark data, zero-shot scoring, shift diagnostics, LoRA/DPO training hooks, and evaluation-comparison commands. Treat it as an execution protocol, not as proof that the real experiment gates have already passed.
