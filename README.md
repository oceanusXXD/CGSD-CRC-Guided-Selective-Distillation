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
- DPO run-pack validation for required method/seed coverage, visible failed runs, required metrics, and paper artifact manifest traceability;
- paired run-metric comparison for baseline-vs-treatment deltas, confidence intervals, and permutation tests;
- paper claim-to-evidence auditing for Gate 10 claim freeze and banned overclaim detection;
- result freeze-pack validation for results manifests, main/appendix tables, figure data, claim maps, frozen metrics, baselines, judge version, and freeze policy;
- Gate 0-10 experiment-readiness auditing so real-data blockers stay explicit instead of being inferred from green unit tests;
- intervention response statistics for monotonicity, slope confidence intervals, and visible failed settings;
- paper artifact generation for frozen Fig. 1-3 / Table 1-3 JSON payloads and freeze-pack manifests;
- baseline selectors, including Random, uncertainty-style selectors, and moment-matched Random;
- soft-group interval, calibration, and error-audit helpers;
- budget, cost, statistical, composition, and paper-table aggregation utilities.

The real AAAI experiment gates are not complete until the required datasets, model checkpoints, training runs, DPO runs, statistical tests, and paper figures are produced and audited.

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

Legacy binary-task utilities that remain usable have been moved to import from `mias_dcms/` and kept under `scripts/` only as public entrypoints. They must not recreate or depend on retired package trees.

## Benchmark Protocol

The current multiclass and preference benchmark workflow is described in:

```text
experiments/benchmark_shift_protocol.md
```

That protocol covers prepared benchmark data, zero-shot scoring, shift diagnostics, LoRA/DPO training hooks, and evaluation-comparison commands. Treat it as an execution protocol, not as proof that the real experiment gates have already passed.
