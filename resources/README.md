# Paper Resources

This directory is the sanitized resource bundle for `paper_outline_final.md`.

The CSVs are organized by comparison. A result CSV should answer one question such as "random vs difficulty" or "random vs difficulty vs PCSS". CSVs contain only anonymized dataset rows or aggregate results. They do not store source artifact paths, absolute paths, raw text, prediction JSONL names, checkpoints, API keys, or account directory names.

Relative paths are documented only in Markdown index and analysis files.

## Bundle Layout

Dataset CSVs:

- `resources/datasets/imdb.csv`: anonymized IMDb q1/q2 converted rows.
- `resources/datasets/fever.csv`: anonymized FEVER support-task converted rows.
- `resources/datasets/codebase.csv`: anonymized Codebase q1/q2/q3 converted rows.
- `resources/datasets/twitterhate.csv`: anonymized TwitterHate q1 converted rows.
- `resources/datasets/summary.csv`: dataset row and label counts only.

Result CSVs are split into two folders:

- `resources/results/required/`: Table 1-5 CSVs directly needed by `paper_outline_final.md`.
- `resources/results/extra/`: additional aggregate results kept for appendix, checking, or future writing.

Required result comparison CSVs:

- `resources/results/required/tables.md`: merged Markdown tables for Table 1-5.
- `resources/results/required/table1.csv`: SLM prior-bias profile.
- `resources/results/required/table2.csv`: training label distribution shift by method.
- `resources/results/required/table3a.csv`: overcorrection cases.
- `resources/results/required/table3b.csv`: source-vs-distribution control experiment.
- `resources/results/required/table4.csv`: capacity moderator results.
- `resources/results/required/table5.csv`: PCSS main results.

Supplementary result CSVs:

- `resources/results/extra/training_composition.csv`: selected training-set composition across the main experiment groups.
- `resources/results/extra/final_results.csv`: training composition joined with post-training evaluation metrics.
- `resources/results/extra/base_correct_wrong_results.csv`: detailed base-correct/base-wrong control results.
- `resources/results/extra/balanced_guide_results.csv`: balanced-guide common-test results for Codebase and TwitterHate.
- `resources/results/extra/imdb_full_eval.csv`: IMDb full-test evaluation and repair/breakdown counts.
- `resources/results/extra/twitterhate_methods.csv`: TwitterHate four-method comparison at budget 2231.
- `resources/results/extra/low_resource_results.csv`: Codebase/TwitterHate low-resource 50/125/250 detailed runs.
- `resources/results/extra/low_resource_pairwise.csv`: paired low-resource winner comparison.
- `resources/results/extra/low_resource_summary.csv`: low-resource mean performance by budget and method.
- `resources/results/extra/fever_budget_sweep.csv`: FEVER 0.6B budget sweep from 1500 to 6000.
- `resources/results/extra/fever_error_set_budget_metrics.csv`: FEVER error-set budget metrics.
- `resources/results/extra/fever_error_set_budget_diffs.csv`: FEVER error-set budget deltas.
- `resources/results/extra/fever_formula500.csv`: FEVER train=500 formula-ratio overcorrection case.
- `resources/results/extra/fever_17b_primary_results.csv`: FEVER Qwen3-1.7B primary multi-method results.
- `resources/results/extra/certification_defer_error.csv`: CRC defer/error summary on common tests.
- `resources/results/extra/certification_param_selected.csv`: selected best-per-method CRC parameter-search records.
- `resources/results/extra/certification_param_lowdefer.csv`: low-defer CRC parameter-search records.
- `resources/results/extra/certification_param_favorable.csv`: favorable-vs-random CRC parameter-search records.
- `resources/results/extra/deployment_cost.csv`: deployment and annotation call-count summary.

Analysis:

- `resources/results/results_analysis.md`: classification rationale and table-by-table interpretation.

The full CRC parameter grid is intentionally not included as a result CSV because it is a large search trace rather than a compact comparison table.

There is no separate manifest CSV; this README is the file index so that CSV files can stay result-only.

## Dataset Schema

The four dataset row files share this schema:

`dataset, query_id, sample_id, label, answer, source_id_hash`

- `sample_id` is a new anonymous ID local to this bundle.
- `source_id_hash` is a short hash of the original source ID for traceability without exposing the source identifier.
- Experimental partition membership is intentionally omitted; these files are pure dataset rows.
- No raw tweet, review, FEVER claim/evidence, or codebase document text is included.

`resources/datasets/summary.csv` uses:

`dataset, queries, rows, label0, label1`

It intentionally omits file paths, source artifact names, and partition counts.

## Result Column Conventions

- `true_yes_pct`: ground-truth positive-label percentage.
- `base_pred_yes_pct`: zero-shot base-model positive-prediction percentage.
- `train_yes_pct`: selected training-set positive-label percentage.
- `pred_yes_pct`: trained model positive-prediction percentage on evaluation.
- `gap_pp`: percentage-point gap from the true label distribution.
- `delta_vs_*`: metric difference against the named baseline.
- `difficulty_selector`: the exact difficulty-aware selector used in that comparison.
- `group` and `dataset_query`: human-readable grouping labels for supplemental tables.
- `common_n`, `pool_n`, `accept_n`, and `defer_n`: common-test and routing counts.
- Blank numeric cells mean the result is not available; check `status` when present.

## Privacy And Anonymization

This bundle keeps only aggregate metrics and anonymized converted dataset rows. The CSVs have no absolute paths and no source artifact paths. Relative bundle paths appear only in Markdown documentation.
