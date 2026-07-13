> **Status:** This file is a legacy evidence table for the binary-task experiments. It preserves measured results but is not the canonical AAAI paper narrative. Interpret CRC-error-mass, NS-error-mass, and Defer-kcenter as diagnostic probes; use `docs/aaai_long_paper_direction.md` for current claims and structure.





| --- | --- | --- | --- | --- | --- | --- | --- |
| IMDb q1 positive | Qwen3-0.6B | 49990 | 49.99 | 99.99 | 33.35 | yes | no |
| IMDb q2 negative | Qwen3-0.6B | 49990 | 27.49 | 100 | 21.57 | yes | no |
| TwitterHate | Qwen3-0.6B | 17348 | 83.2 | 100 | 45.42 | yes | no |
| Codebase q1 social link | Qwen3-0.6B | 9298 | 6.25 | 93.82 | 12.3 | yes | no |
| Codebase q2 CS interest | Qwen3-0.6B | 9298 | 61.95 | 99.45 | 39.6 | yes | no |
| Codebase q3 factual ID | Qwen3-0.6B | 9298 | 25.75 | 99.88 | 20.65 | yes | no |
| FEVER support | Qwen3-0.6B | 165447 | 52.4 | 10.01 | 44.08 | no | yes |





| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| IMDb full-test | IMDb q1 | pool-random | 2500 | yes | no | 1231 | 49.24 | 50.76 | 49.24 | 1499 | 1001 | 49.2 |
| IMDb full-test | IMDb q1 | ns-error-mass | 2500 | yes | no | 2316 | 92.64 | 7.36 | 92.64 | 819 | 1681 | 92.6 |
| IMDb full-test | IMDb q2 | pool-random | 2500 | yes | no | 1805 | 72.2 | 27.8 | 72.2 | 697 | 1803 | 72.2 |
| IMDb full-test | IMDb q2 | ns-error-mass | 2500 | yes | no | 1975 | 79 | 21 | 79 | 577 | 1923 | 79 |
| Base-correct/wrong control | Codebase q1 | base-correct-balanced | 500 | yes | no | 250 | 50 | 50 | 50 |  |  | 0 |
| Base-correct/wrong control | Codebase q2 | base-correct-random | 500 | yes | no | 4 | 0.8 | 99.2 | 0.8 |  |  | 0 |
| Base-correct/wrong control | Codebase q2 | base-wrong-random | 500 | yes | no | 500 | 100 | 0 | 100 |  |  | 100 |
| FEVER 0.6B budget | FEVER n=1500 | pool-random | 1500 | no | yes | 785 | 52.33 | 52.33 | 47.67 | 434 | 1066 | 46.33 |
| FEVER 0.6B budget | FEVER n=1500 | ns-error-mass | 1500 | no | yes | 992 | 66.13 | 66.13 | 33.87 | 380 | 1120 | 65.6 |
| FEVER 0.6B budget | FEVER n=3000 | pool-random | 3000 | no | yes | 1568 | 52.27 | 52.27 | 47.73 | 848 | 2152 | 47.57 |
| FEVER 0.6B budget | FEVER n=3000 | ns-error-mass | 3000 | no | yes | 2041 | 68.03 | 68.03 | 31.97 | 760 | 2240 | 67.37 |
| FEVER 0.6B budget | FEVER n=4500 | pool-random | 4500 | no | yes | 2366 | 52.58 | 52.58 | 47.42 | 1262 | 3238 | 47.44 |
| FEVER 0.6B budget | FEVER n=4500 | ns-error-mass | 4500 | no | yes | 3073 | 68.29 | 68.29 | 31.71 | 1140 | 3360 | 67.6 |
| FEVER 0.6B budget | FEVER n=6000 | pool-random | 6000 | no | yes | 3145 | 52.42 | 52.42 | 47.58 | 1670 | 4330 | 47.55 |
| FEVER 0.6B budget | FEVER n=6000 | ns-error-mass | 6000 | no | yes | 4092 | 68.2 | 68.2 | 31.8 | 1520 | 4480 | 67.48 |
| Low-resource 0.6B | codebase q2 n=125 | random | 125 | yes | no | 39 | 31.2 | 68.8 | 31.2 | 48 | 77 | 32 |
| Low-resource 0.6B | codebase q2 n=125 | crc-error-mass | 125 | yes | no | 41 | 32.8 | 67.2 | 32.8 | 41 | 84 | 32.8 |
| Low-resource 0.6B | codebase q3 n=250 | random | 250 | yes | no | 188 | 75.2 | 24.8 | 75.2 | 39 | 211 | 74.8 |
| Low-resource 0.6B | codebase q3 n=250 | crc-error-mass | 250 | yes | no | 192 | 76.8 | 23.2 | 76.8 | 40 | 210 | 76 |
| Low-resource 0.6B | twitter_hate q1 n=50 | random | 50 | yes | no | 11 | 22 | 78 | 22 | 41 | 9 | 22 |
| Low-resource 0.6B | twitter_hate q1 n=50 | crc-error-mass | 50 | yes | no | 17 | 34 | 66 | 34 | 27 | 23 | 34 |
| Low-resource 0.6B | twitter_hate q1 n=125 | random | 125 | yes | no | 27 | 21.6 | 78.4 | 21.6 | 102 | 23 | 21.6 |
| Low-resource 0.6B | twitter_hate q1 n=125 | crc-error-mass | 125 | yes | no | 35 | 28 | 72 | 28 | 67 | 58 | 28 |
| Low-resource 0.6B | twitter_hate q1 n=250 | random | 250 | yes | no | 47 | 18.8 | 81.2 | 18.8 | 203 | 47 | 18.8 |
| Low-resource 0.6B | twitter_hate q1 n=250 | crc-error-mass | 250 | yes | no | 69 | 27.6 | 72.4 | 27.6 | 134 | 116 | 27.6 |
| FEVER 0.6B formula500 | FEVER n=500 | defer_kcenter | 500 | no | yes | 333 | 66.6 | 66.6 | 33.4 | 65 | 435 |  |
| FEVER 0.6B formula500 | FEVER n=500 | random_defer | 500 | no | yes | 260 | 52 | 52 | 48 | 65 | 435 |  |





| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |





| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |






| --- | --- | --- | --- | --- | --- | --- | --- |



| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1500 | pool-random | 785 | 52.33 | 654 | 52.33 | 50.22 | 92.24 | 0.00 |
| 1500 | ns-error-mass | 992 | 66.13 | 976 | 66.13 | 50.61 | 93.11 | +0.87 |
| 3000 | pool-random | 1568 | 52.27 | 1347 | 52.27 | 48.82 | 92.50 | 0.00 |
| 3000 | ns-error-mass | 2041 | 68.03 | 2005 | 68.03 | 50.27 | 93.51 | +1.01 |
| 4500 | pool-random | 2366 | 52.58 | 2018 | 52.58 | 49.79 | 93.03 | 0.00 |
| 4500 | ns-error-mass | 3073 | 68.29 | 3019 | 68.29 | 50.91 | 94.03 | +1.00 |
| 6000 | pool-random | 3145 | 52.42 | 2694 | 52.42 | 50.21 | 93.97 | 0.00 |
| 6000 | ns-error-mass | 4092 | 68.20 | 4019 | 68.20 | 50.21 | 94.47 | +0.50 |



| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |



| --- | --- | --- | --- | --- | --- | --- |
| defer_kcenter | 333 | 66.60 | 66.60 | 79.97 | 60.29 | 69.45 |
| random_defer | 260 | 52.00 | 52.00 | 71.83 | 71.33 | 76.47 |

5.5 balanced-guide 500/1000


| --- | --- | --- | --- | --- | --- | --- | --- |
| codebase | 1 | 500 | crc-error-mass | 80.12 | 7744 | 380 | 4.68 |
| codebase | 1 | 500 | ns-error-mass | 83.91 | 7948 | 176 | 2.17 |
| codebase | 2 | 500 | crc-error-mass | 86.18 | 6984 | 1141 | 14.04 |
| codebase | 2 | 500 | ns-error-mass | 83.71 | 6858 | 1267 | 15.59 |
| codebase | 3 | 500 | crc-error-mass | 76.94 | 6367 | 1766 | 21.71 |
| codebase | 3 | 500 | ns-error-mass | 76.84 | 5945 | 2188 | 26.90 |
| twitter_hate | 1 | 1000 | crc-error-mass | 85.87 | 22050 | 279 | 1.25 |
| twitter_hate | 1 | 1000 | ns-error-mass | 85.69 | 21714 | 615 | 2.75 |



| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | pool-random | 2500 | 94.37 | 94.37 | 24996 | 23465 | 1283 | 22182 |
| 1 | ns-error-mass | 2500 | 92.27 | 92.25 | 24996 | 24165 | 3034 | 21131 |
| 2 | pool-random | 2500 | 88.69 | 85.72 | 36246 | 33575 | 2981 | 30594 |
| 2 | ns-error-mass | 2500 | 88.70 | 85.54 | 36246 | 33858 | 3260 | 30598 |



| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| crc-error-mass | 2231 | 1106 | 1125 | 27.66 | 6248 | 1187 | 94.84 | 90.57 | +0.21 |
| ns-difficulty-crc-split | 2231 | 1106 | 1125 | 26.45 | 6256 | 1179 | 94.94 | 90.75 | +0.38 |
| ns-difficulty-global | 2231 | 1827 | 404 | 15.82 | 6195 | 1240 | 94.34 | 89.84 | -0.52 |
| random | 2231 | 1840 | 391 | 17.48 | 6144 | 1291 | 94.54 | 90.36 | 0.00 |



| --- | --- | --- | --- | --- |
| 50 | crc-error-mass | 72.77 | 58.46 | 4 |
| 50 | crc-random delta | -6.30 | -2.65 | 4 |
| 50 | random | 79.07 | 61.11 | 4 |
| 125 | crc-error-mass | 82.48 | 63.06 | 4 |
| 125 | crc-random delta | +2.01 | +4.13 | 4 |
| 125 | random | 80.48 | 58.93 | 4 |
| 250 | crc-error-mass | 84.31 | 69.90 | 4 |
| 250 | crc-random delta | +0.05 | -1.33 | 4 |
| 250 | random | 84.26 | 71.22 | 4 |




- `csv/step1_original_bias.csv`
- `csv/step2_training_composition.csv`
- `csv/step3_final_results.csv`
- `csv/step4_capacity_17b.csv`


- `csv/round0_yesno.csv`
- `csv/query_bias_profile.csv`
- `csv/base_correct_wrong_training_results.csv`
- `csv/fever06b_counterexample_budget_1500_6000.csv`
- `csv/low_resource_counterexample_50_250.csv`
- `csv/low_resource_counterexample_pairwise.csv`
- `csv/fever06b_formula500_counterexample.csv`
- `csv/balanced_guide_common_results.csv`
- `csv/imdb_full_eval.csv`
- `csv/twitterhate_four_methods.csv`
- `csv/codebase_twitter_low_resource_summary.csv`
- `csv/fever_qwen17b_primary_results.csv`
