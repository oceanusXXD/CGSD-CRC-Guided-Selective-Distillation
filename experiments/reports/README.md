# Experiment Report Status

The reports in this directory are retained as binary-task evidence and provenance. They are not the canonical AAAI paper narrative.

- CRC-error-mass, NS-error-mass, and Defer-kcenter should be interpreted as legacy probes that expose selector-induced supervision shift.
- PCSS results are preliminary evidence for the binary two-stratum special case of DGA; they do not establish the final method.
- Legacy binary-task result tables, figures, and the old result summary live in `binary_legacy/`.
- `binary_legacy/sample_level_source_inventory.json` records the original binary sample-level data and prediction-log paths. Those source files are not retained in this workspace; aggregate CSVs are not substitutes for sample-level evidence.
- `dpo_pilot/current/` contains two completed CPU HelpSteer2 pilot run records (Random and ActiveDPO+DCMS, seed 1). `dpo_main/current/` remains an older partial planning surface rather than completed main-study evidence.
- `single_seed_dcms_20260715.md` records the current GPU single-seed AG News and HelpSteer2 results. The compact DPO run records and paired comparison live in `dpo_single_seed_20260715/`; neither is multi-seed evidence.
- `multiclass_ag_news_replication_20260715/` contains the frozen AG News seed-43/44 run projections and paired comparison; the downstream mean is descriptive and the realized DCMS class-TV shift is higher than Random.
- `current_comparable_main_results_20260715.md` is the cross-task status summary. It keeps binary, AG News, and HelpSteer2 results separate, records their comparison limits, and states the required next three-arm experiments.
- Current multiclass and preference protocols are documented in `../benchmark_shift_protocol.md`.
- The canonical paper direction is `../../docs/aaai_long_paper_direction.md`.
