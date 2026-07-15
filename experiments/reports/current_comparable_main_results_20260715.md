# Current Comparable Main Results (2026-07-15)

This report separates the currently completed results by task type. Metrics
must not be pooled across these tables. Lower total variation (TV) means that
the selected-label distribution is closer to the corresponding pool
distribution.

All reported results remain descriptive: the binary results use one training
seed, AG News has one seed with all three arms and two further Random vs
Entropy+DCMS replications, and DPO has one paired seed. They are not
multi-seed statistical claims.

## 1. Binary Classification: Original Data

For binary next-token probabilities, Entropy and absolute probability Margin
are monotonic transforms. Their selected IDs are identical on all five gates,
so they share one downstream adapter and are shown in one column. No binary
DCMS arm has been run.

| Dataset | Random: acc / bal-acc / macro-F1 | Entropy/Margin: acc / bal-acc / macro-F1 | Random - Entropy/Margin macro-F1 | Random TV | Entropy/Margin TV |
| --- | ---: | ---: | ---: | ---: | ---: |
| codebase_q2 | .885 / .8613 / .8723 | .804 / .7530 / .7684 | +.1039 | .0762 | .0215 |
| IMDb | .845 / .8450 / .8440 | .852 / .8520 / .8517 | -.0077 | .0625 | .0313 |
| PAWS | .818 / .8228 / .8175 | .581 / .5263 / .4147 | +.4027 | .0273 | .0273 |
| TweetEval Hate | .608 / .6161 / .6072 | .549 / .5945 / .5304 | +.0768 | .0293 | .0020 |
| Twitter Hate | .911 / .8016 / .8269 | .903 / .7873 / .8114 | +.0156 | .0664 | .1680 |

The direct observation is that Entropy/Margin has lower macro-F1 than Random
in four of the five single-seed gates. Its TV is lower in three gates, equal in
PAWS, and higher in Twitter Hate. The downstream loss therefore cannot be
explained simply as a larger label-proportion TV.

Evidence and qualifications:

- Source summaries: `binary_qwen06b_t4_v1/*/single_seed_gate_summary.json`.
- `codebase_q2` and `twitter_hate_q1` use deterministic document-disjoint
  source holdouts, rather than official source test splits; they are not
  historical-result recovery claims.
- The IMDb result has a configuration-provenance exception: its selection
  records and execution snapshot use different frozen config hashes. It is
  retained for visibility but is weaker evidence than the other four rows.

## 2. Multiclass Classification: AG News

| Seed / method | Random acc -> method acc | Random macro-F1 -> method macro-F1 | Random worst-F1 -> method worst-F1 | Random TV -> method TV | DCMS utility retained |
| --- | ---: | ---: | ---: | ---: | ---: |
| 42 / Entropy | .8816 -> .8832 | .882447 -> .884402 | .827055 -> .832621 | .0294 -> .1960 | - |
| 42 / Entropy+DCMS | .8816 -> .8846 | .882447 -> .885632 | .827055 -> .830929 | .0294 -> .0988 | .9907 |
| 43 / Entropy+DCMS | .8904 -> .8894 | .889778 -> .888726 | .845627 -> .849920 | .0076 -> .1556 | .9798 |
| 44 / Entropy+DCMS | .8898 -> .8938 | .889449 -> .893564 | .840301 -> .843312 | .0444 -> .0916 | .9853 |

At seed 42, raw Entropy already exceeds Random. Adding DCMS further improves
accuracy and macro-F1, while worst-class F1 is slightly lower than raw
Entropy. Seeds 43 and 44 do not contain a raw Entropy arm, so they cannot show
that DCMS repairs a raw-Entropy failure relative to Random. DCMS TV is higher
than Random in every listed AG News seed, including both frozen replications.

Evidence: `single_seed_dcms_20260715.md` and
`multiclass_ag_news_replication_20260715/`.

## 3. Preference DPO: HelpSteer2

| Method | Selection TV | Utility retained | Preference acc | Length-controlled win | Worst-group acc | Capability regression | AULC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Random | .0700 | 1.000000 | .375959 | .373351 | .354639 | -.019286 | .375959 |
| ActiveDPO+DCMS | .0046 | .995934 | .375959 | .373351 | .354639 | -.177040 | .375959 |
| DCMS - Random | -.0654 | -.004066 | .000000 | .000000 | .000000 | -.157754 | .000000 |

ActiveDPO+DCMS sharply reduces the audited selection shift, but improves none
of the reported preference metrics in this run and has a larger negative
capability regression. It must not be described as outperforming Random.
Only Random and ActiveDPO+DCMS have been run, so this table cannot quantify
whether DCMS repairs an original ActiveDPO disadvantage.

Evidence: `dpo_single_seed_20260715/` and
`runs/dpo_single_seed_20260715/helpsteer2_preference/qwen3-0.6b/budget_100/seed_1/`.

## Conclusion And Next Work

The binary table is the present evidence for the original observation:
unconstrained Entropy/Margin usually underperforms Random. AG News and DPO
currently establish neither a corresponding raw-method failure nor a DCMS
repair of such a failure.

The priority next experiment is a strict frozen three-arm comparison on each
binary failure gate: Random, Entropy/Margin, and Entropy/Margin+DCMS. Start
with `codebase_q2`, `paws_labeled_final`, `tweeteval_hate`, and
`twitter_hate_q1`, preserving each existing pool, label budget, training
recipe, fixed test set, and seed. Record macro-F1, balanced accuracy, TV,
utility retained, constraint violation, and selected IDs. Then repeat all
three arms over the frozen training-seed set and report paired intervals. The
IMDb comparison should be rerun with a single aligned config snapshot before
it is used in the final aggregate.

After the binary three-arm gates are complete, add raw Entropy at AG News seeds
43 and 44, and add a bare ActiveDPO arm at the same HelpSteer2 seed and budget.
Those additions make the DCMS comparison causal within each task rather than a
two-arm outcome comparison.
