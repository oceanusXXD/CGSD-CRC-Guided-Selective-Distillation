# Experiment Report Status

This directory contains the retained, current evidence for the binary,
multiclass, and preference-DPO stages. The current cross-task status is
`current_comparable_main_results_20260715.md`; its tables must not be pooled
across task types.

- `dpo_pilot/current/` contains two completed CPU HelpSteer2 pilot run records (Random and ActiveDPO+DCMS, seed 1). `dpo_main/current/` remains an older partial planning surface rather than completed main-study evidence.
- `single_seed_dcms_20260715.md` records the current GPU single-seed AG News and HelpSteer2 results. The compact DPO run records and paired comparison live in `dpo_single_seed_20260715/`; neither is multi-seed evidence.
- `multiclass_ag_news_replication_20260715/` contains the frozen AG News seed-43/44 run projections and paired comparison; the downstream mean is descriptive and the realized DCMS class-TV shift is higher than Random.
- `current_comparable_main_results_20260715.md` is the cross-task status summary. It keeps binary, AG News, and HelpSteer2 results separate, records their comparison limits, and states the required next three-arm experiments.
- Current multiclass and preference protocols are documented in `../benchmark_shift_protocol.md`.
- The active paper and execution direction is under `../../docs/project/`.
