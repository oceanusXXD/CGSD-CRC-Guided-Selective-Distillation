



| --- | --- | --- | --- | --- | --- |


- `data.jsonl`
- `embeddings.npy`
- `embeddings.ids.jsonl`
- `embeddings.meta.json`
- `cgsd_split_ids.json`
- `test_eval_split_ids.json`
- `prepare_summary.json`



| --- | --- | ---: | --- | --- | ---: | ---: | ---: |
| codebase | 1 | 9298 | train 9298 | yes 581 / no 8717 | 6.25% | 162.7 | 24.6 |
| codebase | 2 | 9298 | train 9298 | yes 5760 / no 3538 | 61.95% | 162.7 | 24.6 |
| codebase | 3 | 9298 | train 9298 | yes 2394 / no 6904 | 25.75% | 162.7 | 24.6 |
| twitter_hate | 1 | 24783 | train 24783 | yes 20620 / no 4163 | 83.20% | 85.4 | 14.1 |

Codebase queries:

- query 1: `Does this document contain a link to social media?`
- query 2: `Does this document suggest that the user either has experience in computer science or is interested in computer science?`
- query 3: `Does this document mainly focus on factual identification details?`

TwitterHate query:

- query 1: `Does the following comment contain hate speech or offensive language?`

3. Embedding


| --- | --- | ---: | --- | ---: |
| codebase | 1 | 9298 | `experiments/inputs/codebase/query_id_1/embeddings.npy` | 1024 |
| codebase | 2 | 9298 | `experiments/inputs/codebase/query_id_2/embeddings.npy` | 1024 |
| codebase | 3 | 9298 | `experiments/inputs/codebase/query_id_3/embeddings.npy` | 1024 |
| twitter_hate | 1 | 24783 | `experiments/inputs/twitter_hate/embeddings.npy` | 1024 |



| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| codebase | 1 | 9298 | 12.30% | 12.30% | 12.25% | 12.36% | 93.82% |
| codebase | 2 | 9298 | 62.37% | 39.60% | 2.51% | 76.68% | 99.45% |
| codebase | 3 | 9298 | 25.87% | 20.65% | 0.32% | 40.99% | 99.88% |
| twitter_hate | 1 | 24783 | 83.19% | 45.41% | 0.00% | 90.83% | 99.99% |



| --- | --- | ---: | --- | ---: | --- | --- |
| codebase | 1 | 200 | yes 100 (50.00%) / no 100 (50.00%) | 9098 | yes 481 (5.29%) / no 8617 (94.71%) | `experiments/inputs/codebase/query_id_1/qwen06b_full_guide200_balanced50_seed42_split_ids.json` |
| codebase | 2 | 200 | yes 100 (50.00%) / no 100 (50.00%) | 9098 | yes 5660 (62.21%) / no 3438 (37.79%) | `experiments/inputs/codebase/query_id_2/qwen06b_full_guide200_balanced50_seed42_split_ids.json` |
| codebase | 3 | 200 | yes 100 (50.00%) / no 100 (50.00%) | 9098 | yes 2294 (25.21%) / no 6804 (74.79%) | `experiments/inputs/codebase/query_id_3/qwen06b_full_guide200_balanced50_seed42_split_ids.json` |
| twitter_hate | 1 | 500 | yes 250 (50.00%) / no 250 (50.00%) | 24283 | yes 20370 (83.89%) / no 3913 (16.11%) | `experiments/inputs/twitter_hate/qwen06b_full_guide500_balanced50_seed42_split_ids.json` |



codebase query 1

- defer sweep JSON：`experiments/inputs/codebase/query_id_1/round_0/qwen06b_full_guide200_balanced50_seed42_T1_T10_defer_rates_ns.json`

| ---: | ---: | ---: | --- | --- |
| 1 | 7048 | 77.47% | yes 368 (17.95%) / no 1682 (82.05%) | yes 113 (1.60%) / no 6935 (98.40%) |
| 2 | 7022 | 77.18% | yes 368 (17.73%) / no 1708 (82.27%) | yes 113 (1.61%) / no 6909 (98.39%) |
| 3 | 6905 | 75.90% | yes 367 (16.74%) / no 1826 (83.26%) | yes 114 (1.65%) / no 6791 (98.35%) |
| 4 | 6553 | 72.03% | yes 368 (14.46%) / no 2177 (85.54%) | yes 113 (1.72%) / no 6440 (98.28%) |
| 5 | 6553 | 72.03% | yes 368 (14.46%) / no 2177 (85.54%) | yes 113 (1.72%) / no 6440 (98.28%) |
| 6 | 6553 | 72.03% | yes 368 (14.46%) / no 2177 (85.54%) | yes 113 (1.72%) / no 6440 (98.28%) |
| 7 | 6552 | 72.02% | yes 368 (14.45%) / no 2178 (85.55%) | yes 113 (1.72%) / no 6439 (98.28%) |
| 8 | 6558 | 72.08% | yes 368 (14.49%) / no 2172 (85.51%) | yes 113 (1.72%) / no 6445 (98.28%) |
| 9 | 6566 | 72.17% | yes 368 (14.53%) / no 2164 (85.47%) | yes 113 (1.72%) / no 6453 (98.28%) |
| 10 | 6580 | 72.32% | yes 368 (14.61%) / no 2150 (85.39%) | yes 113 (1.72%) / no 6467 (98.28%) |

codebase query 2

- defer sweep JSON：`experiments/inputs/codebase/query_id_2/round_0/qwen06b_full_guide200_balanced50_seed42_T1_T10_defer_rates_ns.json`

| ---: | ---: | ---: | --- | --- |
| 1 | 6001 | 65.96% | yes 2614 (84.40%) / no 483 (15.60%) | yes 3046 (50.76%) / no 2955 (49.24%) |
| 2 | 6010 | 66.06% | yes 2605 (84.36%) / no 483 (15.64%) | yes 3055 (50.83%) / no 2955 (49.17%) |
| 3 | 6007 | 66.03% | yes 2606 (84.31%) / no 485 (15.69%) | yes 3054 (50.84%) / no 2953 (49.16%) |
| 4 | 5986 | 65.79% | yes 2607 (83.77%) / no 505 (16.23%) | yes 3053 (51.00%) / no 2933 (49.00%) |
| 5 | 5967 | 65.59% | yes 2610 (83.36%) / no 521 (16.64%) | yes 3050 (51.11%) / no 2917 (48.89%) |
| 6 | 5967 | 65.59% | yes 2610 (83.36%) / no 521 (16.64%) | yes 3050 (51.11%) / no 2917 (48.89%) |
| 7 | 5967 | 65.59% | yes 2610 (83.36%) / no 521 (16.64%) | yes 3050 (51.11%) / no 2917 (48.89%) |
| 8 | 5967 | 65.59% | yes 2610 (83.36%) / no 521 (16.64%) | yes 3050 (51.11%) / no 2917 (48.89%) |
| 9 | 5967 | 65.59% | yes 2610 (83.36%) / no 521 (16.64%) | yes 3050 (51.11%) / no 2917 (48.89%) |
| 10 | 5967 | 65.59% | yes 2610 (83.36%) / no 521 (16.64%) | yes 3050 (51.11%) / no 2917 (48.89%) |

codebase query 3

- defer sweep JSON：`experiments/inputs/codebase/query_id_3/round_0/qwen06b_full_guide200_balanced50_seed42_T1_T10_defer_rates_ns.json`

| ---: | ---: | ---: | --- | --- |
| 1 | 7985 | 87.77% | yes 364 (32.70%) / no 749 (67.30%) | yes 1930 (24.17%) / no 6055 (75.83%) |
| 2 | 7240 | 79.58% | yes 594 (31.97%) / no 1264 (68.03%) | yes 1700 (23.48%) / no 5540 (76.52%) |
| 3 | 7240 | 79.58% | yes 594 (31.97%) / no 1264 (68.03%) | yes 1700 (23.48%) / no 5540 (76.52%) |
| 4 | 7240 | 79.58% | yes 594 (31.97%) / no 1264 (68.03%) | yes 1700 (23.48%) / no 5540 (76.52%) |
| 5 | 7240 | 79.58% | yes 594 (31.97%) / no 1264 (68.03%) | yes 1700 (23.48%) / no 5540 (76.52%) |
| 6 | 7240 | 79.58% | yes 594 (31.97%) / no 1264 (68.03%) | yes 1700 (23.48%) / no 5540 (76.52%) |
| 7 | 7240 | 79.58% | yes 594 (31.97%) / no 1264 (68.03%) | yes 1700 (23.48%) / no 5540 (76.52%) |
| 8 | 7240 | 79.58% | yes 594 (31.97%) / no 1264 (68.03%) | yes 1700 (23.48%) / no 5540 (76.52%) |
| 9 | 7678 | 84.39% | yes 448 (31.55%) / no 972 (68.45%) | yes 1846 (24.04%) / no 5832 (75.96%) |
| 10 | 7678 | 84.39% | yes 448 (31.55%) / no 972 (68.45%) | yes 1846 (24.04%) / no 5832 (75.96%) |

twitter_hate query 1

- defer sweep JSON：`experiments/inputs/twitter_hate/round_0/qwen06b_full_guide500_balanced50_seed42_T1_T10_defer_rates_ns.json`

| ---: | ---: | ---: | --- | --- |
| 1 | 13561 | 55.85% | yes 10051 (93.74%) / no 671 (6.26%) | yes 10319 (76.09%) / no 3242 (23.91%) |
| 2 | 13874 | 57.13% | yes 9745 (93.62%) / no 664 (6.38%) | yes 10625 (76.58%) / no 3249 (23.42%) |
| 3 | 13874 | 57.13% | yes 9745 (93.62%) / no 664 (6.38%) | yes 10625 (76.58%) / no 3249 (23.42%) |
| 4 | 13874 | 57.13% | yes 9745 (93.62%) / no 664 (6.38%) | yes 10625 (76.58%) / no 3249 (23.42%) |
| 5 | 13874 | 57.13% | yes 9745 (93.62%) / no 664 (6.38%) | yes 10625 (76.58%) / no 3249 (23.42%) |
| 6 | 13874 | 57.13% | yes 9745 (93.62%) / no 664 (6.38%) | yes 10625 (76.58%) / no 3249 (23.42%) |
| 7 | 13874 | 57.13% | yes 9745 (93.62%) / no 664 (6.38%) | yes 10625 (76.58%) / no 3249 (23.42%) |
| 8 | 13874 | 57.13% | yes 9745 (93.62%) / no 664 (6.38%) | yes 10625 (76.58%) / no 3249 (23.42%) |
| 9 | 13874 | 57.13% | yes 9745 (93.62%) / no 664 (6.38%) | yes 10625 (76.58%) / no 3249 (23.42%) |
| 10 | 13874 | 57.13% | yes 9745 (93.62%) / no 664 (6.38%) | yes 10625 (76.58%) / no 3249 (23.42%) |



| --- | --- | ---: | ---: | ---: | --- |
| codebase | 1 | 9098 | 2518 | 6580 | 27.68% / 72.32% |
| codebase | 2 | 9098 | 3131 | 5967 | 34.41% / 65.59% |
| codebase | 3 | 9098 | 1420 | 7678 | 15.61% / 84.39% |
| twitter_hate | 1 | 24283 | 10409 | 13874 | 42.87% / 57.13% |



| --- | --- | --- | ---: | ---: | --- | --- | --- |
| codebase | 1 | crc-error-mass | 500 | 72.32% | 111 / 389 | 22.20% / 77.80% | yes 17 / no 483 |
| codebase | 1 | ns-error-mass | 500 | 72.32% | 111 / 389 | 22.20% / 77.80% | yes 22 / no 478 |
| codebase | 2 | crc-error-mass | 500 | 65.59% | 146 / 354 | 29.20% / 70.80% | yes 304 / no 196 |
| codebase | 2 | ns-error-mass | 500 | 65.59% | 146 / 354 | 29.20% / 70.80% | yes 310 / no 190 |
| codebase | 3 | crc-error-mass | 500 | 84.39% | 78 / 422 | 15.60% / 84.40% | yes 135 / no 365 |
| codebase | 3 | ns-error-mass | 500 | 84.39% | 78 / 422 | 15.60% / 84.40% | yes 143 / no 357 |
| twitter_hate | 1 | crc-error-mass | 1000 | 57.13% | 347 / 653 | 34.70% / 65.30% | yes 811 / no 189 |
| twitter_hate | 1 | ns-error-mass | 1000 | 57.13% | 347 / 653 | 34.70% / 65.30% | yes 815 / no 185 |


- `experiments/inputs/codebase/query_id_1/qwen06b_full_guide200_balanced50_seed42_T10_alpha010_train500_two_methods_seed1/`
- `experiments/inputs/codebase/query_id_2/qwen06b_full_guide200_balanced50_seed42_T10_alpha010_train500_two_methods_seed1/`
- `experiments/inputs/codebase/query_id_3/qwen06b_full_guide200_balanced50_seed42_T10_alpha010_train500_two_methods_seed1/`
- `experiments/inputs/twitter_hate/qwen06b_full_guide500_balanced50_seed42_T10_alpha010_train1000_two_methods_seed1/`





| --- | --- | ---: | ---: | ---: |
| codebase | 1 | 9298 | 2947 | 2949 |
| codebase | 2 | 9298 | 2960 | 2962 |
| codebase | 3 | 9298 | 2950 | 2952 |
| twitter_hate | 1 | 24783 | 624 | 626 |



```text
<|im_start|>system
You are a precise binary classifier. Answer only "1" or "0"./no_think<|im_end|>
<|im_start|>user
Query: {query}
Document: {document}
Return "1" if the document satisfies the query; otherwise return "0".<|im_end|>
<|im_start|>assistant
<think>

</think>

```


| --- | --- |
| base model | `Qwen3-0.6B` |
| input format | `cgsd_chat_binary_v1` |
| epochs | 4 |
| learning rate | `3e-5` |
| LoRA r | 8 |
| LoRA alpha | 16 |
| LoRA dropout | 0.05 |
| target modules | `attention_mlp` |
| lora layer | all |
| max length | 3072 |
| per-device train batch size | 2 |
| gradient accumulation steps | 8 |
| pad to multiple of | 8 |
| seed | 42 |
| precision | `bf16` |



| --- | --- | ---: | --- | ---: | --- |
| codebase | 1 | 10 | crc-error-mass | 500 | `experiments/runs/codebase_twitterhate_balanced50_lora/codebase_q1_crc_error_mass/round_1/model` |
| codebase | 1 | 10 | ns-error-mass | 500 | `experiments/runs/codebase_twitterhate_balanced50_lora/codebase_q1_ns_error_mass/round_1/model` |
| codebase | 2 | 10 | crc-error-mass | 500 | `experiments/runs/codebase_twitterhate_balanced50_lora/codebase_q2_crc_error_mass/round_1/model` |
| codebase | 2 | 10 | ns-error-mass | 500 | `experiments/runs/codebase_twitterhate_balanced50_lora/codebase_q2_ns_error_mass/round_1/model` |
| codebase | 3 | 10 | crc-error-mass | 500 | `experiments/runs/codebase_twitterhate_balanced50_lora/codebase_q3_crc_error_mass/round_1/model` |
| codebase | 3 | 10 | ns-error-mass | 500 | `experiments/runs/codebase_twitterhate_balanced50_lora/codebase_q3_ns_error_mass/round_1/model` |
| twitter_hate | 1 | 10 | crc-error-mass | 1000 | `experiments/runs/codebase_twitterhate_balanced50_lora/twitter_hate_q1_crc_error_mass/round_1/model` |
| twitter_hate | 1 | 10 | ns-error-mass | 1000 | `experiments/runs/codebase_twitterhate_balanced50_lora/twitter_hate_q1_ns_error_mass/round_1/model` |




| --- | --- |
| endpoint | `/v1/completions` |
| prompt | `format_cgsd_chat_prompt(query, document)` |
| temperature | 0 |
| max tokens | 1 |
| top logprobs | 20 |
| parallel requests | 4096 |
| request retries | 3 |
| timeout | 180s |
| max model len | 3072 |
| max num batched tokens | 524288 |
| GPU memory utilization | 0.98 |
| enforce eager | true |



| --- | --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| codebase | 1 | 9298 | 200 | 974 | 429 | 7695 | yes 416 (5.41%) / no 7279 (94.59%) | `experiments/eval_splits/codebase_q1_balanced50_common_minus_original_random_train_split_ids.json` |
| codebase | 2 | 9298 | 200 | 973 | 443 | 7682 | yes 4787 (62.31%) / no 2895 (37.69%) | `experiments/eval_splits/codebase_q2_balanced50_common_minus_original_random_train_split_ids.json` |
| codebase | 3 | 9298 | 200 | 965 | 432 | 7701 | yes 1918 (24.91%) / no 5783 (75.09%) | `experiments/eval_splits/codebase_q3_balanced50_common_minus_original_random_train_split_ids.json` |
| twitter_hate | 1 | 24783 | 500 | 1954 | 914 | 21415 | yes 18013 (84.11%) / no 3402 (15.89%) | `experiments/eval_splits/twitter_hate_q1_balanced50_common_minus_original_random_train_split_ids.json` |





| --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: |
| codebase | 1 | 10 | random | 7695 |  |  |  |  |  |  |  |
| codebase | 1 | 10 | crc-error-mass | 7695 |  |  |  |  |  |  |  |
| codebase | 1 | 10 | ns-error-mass | 7695 |  |  |  |  |  |  |  |
| codebase | 2 | 10 | random | 7682 |  |  |  |  |  |  |  |
| codebase | 2 | 10 | crc-error-mass | 7682 |  |  |  |  |  |  |  |
| codebase | 2 | 10 | ns-error-mass | 7682 |  |  |  |  |  |  |  |
| codebase | 3 | 10 | random | 7701 |  |  |  |  |  |  |  |
| codebase | 3 | 10 | crc-error-mass | 7701 |  |  |  |  |  |  |  |
| codebase | 3 | 10 | ns-error-mass | 7701 |  |  |  |  |  |  |  |
| twitter_hate | 1 | 10 | random | 21415 |  |  |  |  |  |  |  |
| twitter_hate | 1 | 10 | crc-error-mass | 21415 |  |  |  |  |  |  |  |
| twitter_hate | 1 | 10 | ns-error-mass | 21415 |  |  |  |  |  |  |  |



| --- | --- | --- | ---: | --- | --- |
| codebase | 1 | crc-error-mass | 250 | 56 / 194 | yes 10 / no 240 |
| codebase | 1 | ns-error-mass | 250 | 56 / 194 | yes 8 / no 242 |
| codebase | 2 | crc-error-mass | 250 | 73 / 177 | yes 151 / no 99 |
| codebase | 2 | ns-error-mass | 250 | 73 / 177 | yes 151 / no 99 |
| codebase | 3 | crc-error-mass | 250 | 39 / 211 | yes 66 / no 184 |
| codebase | 3 | ns-error-mass | 250 | 39 / 211 | yes 78 / no 172 |
| twitter_hate | 1 | crc-error-mass | 500 | 174 / 326 | yes 412 / no 88 |
| twitter_hate | 1 | ns-error-mass | 500 | 174 / 326 | yes 410 / no 90 |


- `experiments/inputs/codebase/query_id_1/qwen06b_full_guide200_balanced50_seed42_T10_alpha010_train250_two_methods_seed1/`
- `experiments/inputs/codebase/query_id_2/qwen06b_full_guide200_balanced50_seed42_T10_alpha010_train250_two_methods_seed1/`
- `experiments/inputs/codebase/query_id_3/qwen06b_full_guide200_balanced50_seed42_T10_alpha010_train250_two_methods_seed1/`
- `experiments/inputs/twitter_hate/qwen06b_full_guide500_balanced50_seed42_T10_alpha010_train500_two_methods_seed1/`




| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | --- | --- | --- |
| codebase | 1 | 250 | random | 7670 | 0.944589 | 0.808327 | 0.969937 | 0.646717 | yes 489 / no 7181 | yes 714 / no 6956 | 389/6856/325/100 |
| codebase | 1 | 250 | crc-error-mass | 7670 | 0.953064 | 0.773300 | 0.975172 | 0.571429 | yes 489 / no 7181 | yes 351 / no 7319 | 240/7070/111/249 |
| codebase | 1 | 125 | random | 7670 | 0.919296 | 0.758124 | 0.955567 | 0.560681 | yes 489 / no 7181 | yes 920 / no 6750 | 395/6656/525/94 |
| codebase | 1 | 125 | crc-error-mass | 7670 | 0.945502 | 0.721079 | 0.971271 | 0.470886 | yes 489 / no 7181 | yes 301 / no 7369 | 186/7066/115/303 |
| codebase | 1 | 50 | random | 7670 | 0.872751 | 0.690760 | 0.927992 | 0.453527 | yes 489 / no 7181 | yes 1297 / no 6373 | 405/6289/892/84 |
| codebase | 1 | 50 | crc-error-mass | 7670 | 0.877966 | 0.690713 | 0.931368 | 0.450059 | yes 489 / no 7181 | yes 1213 / no 6457 | 383/6351/830/106 |
| codebase | 2 | 250 | random | 7678 | 0.788226 | 0.764175 | 0.688863 | 0.839487 | yes 4771 / no 2907 | yes 5359 / no 2319 | 4252/1800/1107/519 |
| codebase | 2 | 250 | crc-error-mass | 7678 | 0.789919 | 0.780395 | 0.734660 | 0.826129 | yes 4771 / no 2907 | yes 4506 / no 3172 | 3832/2233/674/939 |
| codebase | 2 | 125 | random | 7678 | 0.708388 | 0.615162 | 0.425750 | 0.804574 | yes 4771 / no 2907 | yes 6686 / no 992 | 4609/830/2077/162 |
| codebase | 2 | 125 | crc-error-mass | 7678 | 0.750847 | 0.717297 | 0.619909 | 0.814686 | yes 4771 / no 2907 | yes 5552 / no 2126 | 4205/1560/1347/566 |
| codebase | 2 | 50 | random | 7678 | 0.716463 | 0.668481 | 0.542359 | 0.794603 | yes 4771 / no 2907 | yes 5828 / no 1850 | 4211/1290/1617/560 |
| codebase | 2 | 50 | crc-error-mass | 7678 | 0.718677 | 0.689105 | 0.593220 | 0.784989 | yes 4771 / no 2907 | yes 5275 / no 2403 | 3943/1575/1332/828 |
| codebase | 3 | 250 | random | 7670 | 0.765841 | 0.560601 | 0.860905 | 0.260297 | yes 1982 / no 5688 | yes 446 / no 7224 | 316/5558/130/1666 |
| codebase | 3 | 250 | crc-error-mass | 7670 | 0.759322 | 0.506469 | 0.859726 | 0.153211 | yes 1982 / no 5688 | yes 198 / no 7472 | 167/5657/31/1815 |
| codebase | 3 | 125 | random | 7670 | 0.740287 | 0.426850 | 0.850697 | 0.003003 | yes 1982 / no 5688 | yes 16 / no 7654 | 3/5675/13/1979 |
| codebase | 3 | 125 | crc-error-mass | 7670 | 0.741069 | 0.425640 | 0.851281 | 0.000000 | yes 1982 / no 5688 | yes 4 / no 7666 | 0/5684/4/1982 |
| codebase | 3 | 50 | random | 7670 | 0.741460 | 0.425769 | 0.851539 | 0.000000 | yes 1982 / no 5688 | yes 1 / no 7669 | 0/5687/1/1982 |
| codebase | 3 | 50 | crc-error-mass | 7670 | 0.741460 | 0.425769 | 0.851539 | 0.000000 | yes 1982 / no 5688 | yes 1 / no 7669 | 0/5687/1/1982 |
| twitter_hate | 1 | 250 | random | 21419 | 0.871936 | 0.715884 | 0.505320 | 0.926447 | yes 17960 / no 3459 | yes 19333 / no 2086 | 17275/1401/2058/685 |
| twitter_hate | 1 | 250 | crc-error-mass | 21419 | 0.870162 | 0.735751 | 0.547290 | 0.924213 | yes 17960 / no 3459 | yes 18735 / no 2684 | 16957/1681/1778/1003 |
| twitter_hate | 1 | 125 | random | 21419 | 0.851113 | 0.557039 | 0.196118 | 0.917959 | yes 17960 / no 3459 | yes 20911 / no 508 | 17841/389/3070/119 |
| twitter_hate | 1 | 125 | crc-error-mass | 21419 | 0.861945 | 0.658505 | 0.394925 | 0.922084 | yes 17960 / no 3459 | yes 19991 / no 1428 | 17497/965/2494/463 |
| twitter_hate | 1 | 50 | random | 21419 | 0.832205 | 0.659377 | 0.416748 | 0.902007 | yes 17960 / no 3459 | yes 18716 / no 2703 | 16541/1284/2175/1419 |
| twitter_hate | 1 | 50 | crc-error-mass | 21419 | 0.572856 | 0.532990 | 0.396544 | 0.669437 | yes 17960 / no 3459 | yes 9717 / no 11702 | 9264/3006/453/8696 |


| ---: | --- | ---: | ---: | ---: |
| 250 | random | 0.842648 | 0.851700 | 0.712247 |
| 250 | crc-error-mass | 0.843117 | 0.851475 | 0.698979 |
| 125 | random | 0.804771 | 0.819092 | 0.589294 |
| 125 | crc-error-mass | 0.824841 | 0.836308 | 0.630630 |
| 50 | random | 0.790720 | 0.803542 | 0.611097 |
| 50 | crc-error-mass | 0.727740 | 0.679816 | 0.584644 |
