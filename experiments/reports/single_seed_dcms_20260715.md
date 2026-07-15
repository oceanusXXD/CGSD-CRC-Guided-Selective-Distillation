# Single-Seed DCMS Status (2026-07-15)

This document records descriptive, single-seed evidence only. It is not a
multi-seed claim, a statistical significance result, or a license to tune
against either held-out test split.

## AG News Classification

Protocol: split seed 42, 400 labeled seed rows, 5,000-row active pool, 500
acquired rows, and 5,000 held-out test rows. Both methods train on the same
400 seed labels plus their 500 acquired labels. Entropy+DCMS was frozen at
`kappa=0.05` and selected slack `0.065`; it retained `0.990742` normalized
entropy utility.

| Method | Selection class TV | Test accuracy | Test macro-F1 | Worst-class F1 |
| --- | ---: | ---: | ---: | ---: |
| Random | 0.0294 | 0.8816 | 0.882447 | 0.827055 |
| Entropy | 0.1960 | 0.8832 | 0.884402 | 0.832621 |
| Entropy+DCMS | 0.0988 | 0.8846 | 0.885632 | 0.830929 |

Entropy+DCMS improves all three listed downstream metrics over Random for this
single run: `+0.0030` accuracy, `+0.003185` macro-F1, and `+0.003874`
worst-class F1. It also reduces entropy selection TV by `0.0972`, but remains
outside the Random 95% TV envelope (`0.0518`). These are descriptive outcomes;
two additional frozen split/training seeds are required before a robustness
claim.

Sources:

- `experiments/runs/multiclass/ag_news_qwen06b_t4_v1/single_seed_dcms/selection/classification_diagnostics.json`
- `experiments/runs/multiclass/ag_news_qwen06b_t4_v1/single_seed_dcms/evaluation/comparison_seed_random_dcms.json`

### Frozen Replications (Seeds 43 and 44)

The two additional seeds use the same split sizes, training recipe, budget,
DCMS kappa, and slack grid. They are independent of the seed-42 held-out
test set and were not tuned using test outcomes.

| Seed | Random accuracy | DCMS accuracy | Delta | Random macro-F1 | DCMS macro-F1 | Delta | Worst-class F1 delta |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 43 | 0.8904 | 0.8894 | -0.0010 | 0.889778 | 0.888726 | -0.001053 | +0.004293 |
| 44 | 0.8898 | 0.8938 | +0.0040 | 0.889449 | 0.893564 | +0.004115 | +0.003011 |
| Mean delta |  |  | +0.0015 |  |  | +0.001531 | +0.003652 |

The mean is positive on all three downstream metrics, but accuracy and
macro-F1 change sign across the two seeds. This is still descriptive two-seed
evidence, not a robust multi-seed claim. More importantly, realized class TV
remains higher for DCMS than Random in both replications: seed 43 `0.1556 vs
0.0076`, seed 44 `0.0916 vs 0.0444`. The current DCMS posterior proxy is not
yet a reliable class-balance control despite the positive mean downstream
result.

Sources:

- `experiments/runs/multiclass/ag_news_qwen06b_t4_v1/replication/seed_43/selection/classification_diagnostics.json`
- `experiments/runs/multiclass/ag_news_qwen06b_t4_v1/replication/seed_43/evaluation/comparison_random_dcms.json`
- `experiments/runs/multiclass/ag_news_qwen06b_t4_v1/replication/seed_44/selection/classification_diagnostics.json`
- `experiments/runs/multiclass/ag_news_qwen06b_t4_v1/replication/seed_44/evaluation/comparison_random_dcms.json`

## HelpSteer2 Preference DPO

Protocol: split/training seed 1, 1,000 seed labels, 10,000 selection rows,
100 acquired labels, 2,000 held-out rows, and 25 DPO optimizer steps from the
same initial policy adapter. The held-out scoring contains 1,564 usable
preference pairs after materialization.

| Metric | Random | ActiveDPO+DCMS | DCMS minus Random |
| --- | ---: | ---: | ---: |
| Acquisition TV | 0.0700 | 0.0046 | -0.0654 |
| Utility retained | 1.000000 | 0.995934 | -0.004066 |
| Preference accuracy | 0.375959 | 0.375959 | 0.000000 |
| Length-controlled win | 0.373351 | 0.373351 | 0.000000 |
| Worst-group preference accuracy | 0.354639 | 0.354639 | 0.000000 |
| AULC | 0.375959 | 0.375959 | 0.000000 |
| Capability regression | -0.019286 | -0.177040 | -0.157754 |

ActiveDPO+DCMS sharply reduces the audited selection shift, but does not
improve any reported preference outcome in this run and has a larger negative
capability regression. It must not be described as outperforming Random.

`run_records.jsonl` is a compact projection of the two completed source run
records. `comparison_seed_1.json` contains the paired metric calculation with
one paired seed; the degenerate confidence intervals and p-values are
descriptive only, not inferential evidence.

Sources:

- `experiments/runs/dpo_single_seed_20260715/helpsteer2_preference/qwen3-0.6b/budget_100/seed_1/Random/run_record.json`
- `experiments/runs/dpo_single_seed_20260715/helpsteer2_preference/qwen3-0.6b/budget_100/seed_1/ActiveDPO+DCMS/run_record.json`
