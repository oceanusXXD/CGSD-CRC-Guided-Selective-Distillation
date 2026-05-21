# Codebase 与 TwitterHate 数据/实验状态

更新时间：2026-05-21。

## 1. 数据状态

所有已完成部分都已经实际放到 `LLM_layer_test/experiments/inputs`

| 数据集 | query_id | 输入目录 | 数据 | embedding | 0.6B round0 |
| --- | --- | --- | --- | --- | --- |
| codebase | 1 | `experiments/inputs/codebase/query_id_1/` | 已补齐 | 已补齐 | 已补齐 |
| codebase | 2 | `experiments/inputs/codebase/query_id_2/` | 已补齐 | 已补齐 | 已补齐 |
| codebase | 3 | `experiments/inputs/codebase/query_id_3/` | 已补齐 | 已补齐 | 已补齐 |
| twitter_hate | 1 | `experiments/inputs/twitter_hate/` | 已补齐 | 已补齐 | 已补齐 |

每个 codebase query 目录包含：

- `data.jsonl`
- `embeddings.npy`
- `embeddings.ids.jsonl`
- `embeddings.meta.json`
- `cgsd_split_ids.json`
- `test_eval_split_ids.json`
- `prepare_summary.json`

汇总校验文件：`experiments/inputs/codebase/manifest.json` 和 `experiments/inputs/codebase_twitterhate_metrics_summary.json`。

## 2. 数据集分析

| 数据集 | query_id | rows | split | yes/no | yes 比例 | 平均字符数 | 平均词数 |
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

## 3. Embedding


| 数据集 | query_id | rows | embedding | 维度 |
| --- | --- | ---: | --- | ---: |
| codebase | 1 | 9298 | `experiments/inputs/codebase/query_id_1/embeddings.npy` | 1024 |
| codebase | 2 | 9298 | `experiments/inputs/codebase/query_id_2/embeddings.npy` | 1024 |
| codebase | 3 | 9298 | `experiments/inputs/codebase/query_id_3/embeddings.npy` | 1024 |
| twitter_hate | 1 | 24783 | `experiments/inputs/twitter_hate/embeddings.npy` | 1024 |

## 4. 0.6B Base 结果

`Qwen3-0.6B` base 全集结果

| 数据集 | query_id | n | acc | macro-F1 | F1 no | F1 yes | pred yes |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| codebase | 1 | 9298 | 12.30% | 12.30% | 12.25% | 12.36% | 93.82% |
| codebase | 2 | 9298 | 62.37% | 39.60% | 2.51% | 76.68% | 99.45% |
| codebase | 3 | 9298 | 25.87% | 20.65% | 0.32% | 40.99% | 99.88% |
| twitter_hate | 1 | 24783 | 83.19% | 45.41% | 0.00% | 90.83% | 99.99% |

## 4. Guide 集分析

本文件使用 balanced50 guide：每个 query_id 内按标签固定 50/50 构造 guide。codebase 每个 query 取 yes 100 / no 100，twitter_hate 取 yes 250 / no 250；随机种子为 42。

| 数据集 | query_id | guide n | guide yes/no占比 | pool n | pool yes/no占比 | split 文件 |
| --- | --- | ---: | --- | ---: | --- | --- |
| codebase | 1 | 200 | yes 100 (50.00%) / no 100 (50.00%) | 9098 | yes 481 (5.29%) / no 8617 (94.71%) | `experiments/inputs/codebase/query_id_1/qwen06b_full_guide200_balanced50_seed42_split_ids.json` |
| codebase | 2 | 200 | yes 100 (50.00%) / no 100 (50.00%) | 9098 | yes 5660 (62.21%) / no 3438 (37.79%) | `experiments/inputs/codebase/query_id_2/qwen06b_full_guide200_balanced50_seed42_split_ids.json` |
| codebase | 3 | 200 | yes 100 (50.00%) / no 100 (50.00%) | 9098 | yes 2294 (25.21%) / no 6804 (74.79%) | `experiments/inputs/codebase/query_id_3/qwen06b_full_guide200_balanced50_seed42_split_ids.json` |
| twitter_hate | 1 | 500 | yes 250 (50.00%) / no 250 (50.00%) | 24283 | yes 20370 (83.89%) / no 3913 (16.11%) | `experiments/inputs/twitter_hate/qwen06b_full_guide500_balanced50_seed42_split_ids.json` |

## 5. Defer 集分析

表内 defer 统计均在 balanced50 guide 之外的剩余 pool 上计算，T=1..10，alpha=0.1，CRC 使用 neighbor-support embedding 口径。

### codebase query 1

- guide: 200 条；pool: 9098 条；split 文件：`experiments/inputs/codebase/query_id_1/qwen06b_full_guide200_balanced50_seed42_split_ids.json`
- defer sweep JSON：`experiments/inputs/codebase/query_id_1/round_0/qwen06b_full_guide200_balanced50_seed42_T1_T10_defer_rates_ns.json`

| T | defer数量 | defer比例 | accept yes/no占比 | defer yes/no占比 |
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

### codebase query 2

- guide: 200 条；pool: 9098 条；split 文件：`experiments/inputs/codebase/query_id_2/qwen06b_full_guide200_balanced50_seed42_split_ids.json`
- defer sweep JSON：`experiments/inputs/codebase/query_id_2/round_0/qwen06b_full_guide200_balanced50_seed42_T1_T10_defer_rates_ns.json`

| T | defer数量 | defer比例 | accept yes/no占比 | defer yes/no占比 |
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

### codebase query 3

- guide: 200 条；pool: 9098 条；split 文件：`experiments/inputs/codebase/query_id_3/qwen06b_full_guide200_balanced50_seed42_split_ids.json`
- defer sweep JSON：`experiments/inputs/codebase/query_id_3/round_0/qwen06b_full_guide200_balanced50_seed42_T1_T10_defer_rates_ns.json`

| T | defer数量 | defer比例 | accept yes/no占比 | defer yes/no占比 |
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

### twitter_hate query 1

- guide: 500 条；pool: 24283 条；split 文件：`experiments/inputs/twitter_hate/qwen06b_full_guide500_balanced50_seed42_split_ids.json`
- defer sweep JSON：`experiments/inputs/twitter_hate/round_0/qwen06b_full_guide500_balanced50_seed42_T1_T10_defer_rates_ns.json`

| T | defer数量 | defer比例 | accept yes/no占比 | defer yes/no占比 |
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

## 7. LoRA 训练集 分析

### 7.1 T=10 原始 pool 组成

| 数据集 | query_id | pool n | 原始accept | 原始defer | 原始accept/defer比例 |
| --- | --- | ---: | ---: | ---: | --- |
| codebase | 1 | 9098 | 2518 | 6580 | 27.68% / 72.32% |
| codebase | 2 | 9098 | 3131 | 5967 | 34.41% / 65.59% |
| codebase | 3 | 9098 | 1420 | 7678 | 15.61% / 84.39% |
| twitter_hate | 1 | 24283 | 10409 | 13874 | 42.87% / 57.13% |

### 7.2 选后训练集组成

固定 seed=1；codebase 每个 query 选 500 条，twitter_hate 选 1000 条。balanced50 版本只生成 `crc-error-mass` 和 `ns-error-mass`。`选后比例` 是选出的训练集内部组成。

| 数据集 | query_id | method | n | 原始defer比例 | 选后accept/defer数量 | 选后accept/defer比例 | 训练集yes/no |
| --- | --- | --- | ---: | ---: | --- | --- | --- |
| codebase | 1 | crc-error-mass | 500 | 72.32% | 111 / 389 | 22.20% / 77.80% | yes 17 / no 483 |
| codebase | 1 | ns-error-mass | 500 | 72.32% | 111 / 389 | 22.20% / 77.80% | yes 22 / no 478 |
| codebase | 2 | crc-error-mass | 500 | 65.59% | 146 / 354 | 29.20% / 70.80% | yes 304 / no 196 |
| codebase | 2 | ns-error-mass | 500 | 65.59% | 146 / 354 | 29.20% / 70.80% | yes 310 / no 190 |
| codebase | 3 | crc-error-mass | 500 | 84.39% | 78 / 422 | 15.60% / 84.40% | yes 135 / no 365 |
| codebase | 3 | ns-error-mass | 500 | 84.39% | 78 / 422 | 15.60% / 84.40% | yes 143 / no 357 |
| twitter_hate | 1 | crc-error-mass | 1000 | 57.13% | 347 / 653 | 34.70% / 65.30% | yes 811 / no 189 |
| twitter_hate | 1 | ns-error-mass | 1000 | 57.13% | 347 / 653 | 34.70% / 65.30% | yes 815 / no 185 |

生成目录：

- `experiments/inputs/codebase/query_id_1/qwen06b_full_guide200_balanced50_seed42_T10_alpha010_train500_two_methods_seed1/`
- `experiments/inputs/codebase/query_id_2/qwen06b_full_guide200_balanced50_seed42_T10_alpha010_train500_two_methods_seed1/`
- `experiments/inputs/codebase/query_id_3/qwen06b_full_guide200_balanced50_seed42_T10_alpha010_train500_two_methods_seed1/`
- `experiments/inputs/twitter_hate/qwen06b_full_guide500_balanced50_seed42_T10_alpha010_train1000_two_methods_seed1/`


### 7.3 LoRA 训练参数

后续所有数据集、query_id 和 method 都使用同一套 LoRA 训练参数。

长度检查使用 `Qwen3-0.6B` tokenizer，prompt 为 `format_cgsd_chat_prompt(query, document)`，训练序列长度为 prompt 加监督答案 `1<|im_end|>` 或 `0<|im_end|>`。

| 数据集 | query_id | n | max prompt tokens | max train seq tokens |
| --- | --- | ---: | ---: | ---: |
| codebase | 1 | 9298 | 2947 | 2949 |
| codebase | 2 | 9298 | 2960 | 2962 |
| codebase | 3 | 9298 | 2950 | 2952 |
| twitter_hate | 1 | 24783 | 624 | 626 |

因此 `max length` 统一用 3072，覆盖当前全集；

训练 prompt：

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

监督答案只接 `1<|im_end|>` 或 `0<|im_end|>`；prompt 部分 mask 为 `-100`，loss 只覆盖答案部分。

| 参数 | 值 |
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

### 7.4 LoRA 训练结果

8 个 balanced50 LoRA run 已完成训练，输出目录统一在 `experiments/runs/codebase_twitterhate_balanced50_lora/`。训练脚本当前没有把最终 loss 写入 `training_round_summary.json`，因此这里记录 checkpoint 与训练样本数。

| 数据集 | query_id | T | method | train n | checkpoint |
| --- | --- | ---: | --- | ---: | --- |
| codebase | 1 | 10 | crc-error-mass | 500 | `experiments/runs/codebase_twitterhate_balanced50_lora/codebase_q1_crc_error_mass/round_1/model` |
| codebase | 1 | 10 | ns-error-mass | 500 | `experiments/runs/codebase_twitterhate_balanced50_lora/codebase_q1_ns_error_mass/round_1/model` |
| codebase | 2 | 10 | crc-error-mass | 500 | `experiments/runs/codebase_twitterhate_balanced50_lora/codebase_q2_crc_error_mass/round_1/model` |
| codebase | 2 | 10 | ns-error-mass | 500 | `experiments/runs/codebase_twitterhate_balanced50_lora/codebase_q2_ns_error_mass/round_1/model` |
| codebase | 3 | 10 | crc-error-mass | 500 | `experiments/runs/codebase_twitterhate_balanced50_lora/codebase_q3_crc_error_mass/round_1/model` |
| codebase | 3 | 10 | ns-error-mass | 500 | `experiments/runs/codebase_twitterhate_balanced50_lora/codebase_q3_ns_error_mass/round_1/model` |
| twitter_hate | 1 | 10 | crc-error-mass | 1000 | `experiments/runs/codebase_twitterhate_balanced50_lora/twitter_hate_q1_crc_error_mass/round_1/model` |
| twitter_hate | 1 | 10 | ns-error-mass | 1000 | `experiments/runs/codebase_twitterhate_balanced50_lora/twitter_hate_q1_ns_error_mass/round_1/model` |

## 8. Vllm参数

后续所有 LoRA checkpoint 都使用同一套 vLLM 评测参数。

推理 prompt 与训练 prompt 完全一致

| 参数 | 值 |
| --- | --- |
| endpoint | `/v1/completions` |
| prompt | `format_cgsd_chat_prompt(query, document)` |
| chat template | 不使用 vLLM chat template；prompt 内手写 Qwen3 no-thinking block |
| temperature | 0 |
| max tokens | 1 |
| top logprobs | 20 |
| parallel requests | 4096 |
| request retries | 3 |
| timeout | 180s |
| max model len | 3072 |
| max num seqs | 4096（并发序列数上限，不是上下文长度） |
| max num batched tokens | 524288 |
| GPU memory utilization | 0.98 |
| enforce eager | true |
| 输出格式 | 单 token 二分类预测，读取 `0` / `1` |

### 8.1 测试集分析

测试集统一使用 balanced50 版本 common test：每个 query_id 内从全集中排除 balanced50 guide，并排除两种训练集 `crc-error-mass`、`ns-error-mass` 的并集。这样两种 LoRA 方法共用同一个 test set，避免评测集包含任意一种训练样本。

| 数据集 | query_id | 全集 n | guide n | 两种train并集 n | common test n | common test yes/no占比 |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| codebase | 1 | 9298 | 200 | 974 | 8124 | yes 442 (5.44%) / no 7682 (94.56%) |
| codebase | 2 | 9298 | 200 | 973 | 8125 | yes 5067 (62.36%) / no 3058 (37.64%) |
| codebase | 3 | 9298 | 200 | 965 | 8133 | yes 2029 (24.95%) / no 6104 (75.05%) |
| twitter_hate | 1 | 24783 | 500 | 1954 | 22329 | yes 18781 (84.11%) / no 3548 (15.89%) |

每个目录都包含：

- `common_test_exclude_balanced50_guide_two_train_seed1.ids.json`
- `common_test_exclude_balanced50_guide_two_train_seed1.jsonl`
- `common_test_exclude_balanced50_guide_two_train_seed1.split_ids.json`
- `common_test_exclude_balanced50_guide_two_train_seed1.summary.json`

### 8.2 Vllm结果分析

评测命令：`CODEX_RUN_PARAMS_CONFIRMED=1 bash experiments/runs/codebase_twitterhate_balanced50_lora/run_vllm_common_test_eval.sh`。汇总文件：`experiments/runs/codebase_twitterhate_balanced50_lora/vllm_common_test_T10_summary.json` 和 `experiments/runs/codebase_twitterhate_balanced50_lora/vllm_common_test_T10_summary.md`。

| 数据集 | query_id | T | method | test n | acc | macro-F1 | pos-F1 | balanced-acc | accept/defer | defer比例 | accept错误率 |
| --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: |
| codebase | 1 | 10 | crc-error-mass | 8124 | 96.22% | 80.12% | 62.24% | 77.85% | 7744 / 380 | 4.68% | 2.17% |
| codebase | 1 | 10 | ns-error-mass | 8124 | 96.33% | 83.91% | 69.78% | 87.61% | 7948 / 176 | 2.17% | 3.12% |
| codebase | 2 | 10 | crc-error-mass | 8125 | 87.05% | 86.18% | 89.65% | 86.11% | 6984 / 1141 | 14.04% | 9.18% |
| codebase | 2 | 10 | ns-error-mass | 8125 | 84.62% | 83.71% | 87.55% | 83.92% | 6858 / 1267 | 15.59% | 10.64% |
| codebase | 3 | 10 | crc-error-mass | 8133 | 81.72% | 76.94% | 66.46% | 78.67% | 6367 / 1766 | 21.71% | 17.39% |
| codebase | 3 | 10 | ns-error-mass | 8133 | 81.18% | 76.84% | 66.83% | 79.45% | 5945 / 2188 | 26.90% | 14.53% |
| twitter_hate | 1 | 10 | crc-error-mass | 22329 | 91.62% | 85.87% | 94.88% | 89.96% | 22050 / 279 | 1.25% | 8.19% |
| twitter_hate | 1 | 10 | ns-error-mass | 22329 | 91.89% | 85.69% | 95.11% | 87.85% | 21714 / 615 | 2.75% | 7.48% |

## 9. 50% 训练量子集

这组数据把 balanced50 版本的训练集再减半，用于看 macro-F1 的数据效率；测试集保持不变，直接复用上面的 common test。

| 数据集 | query_id | method | train n | accept / defer | train yes/no |
| --- | --- | --- | ---: | --- | --- |
| codebase | 1 | crc-error-mass | 250 | 56 / 194 | yes 10 / no 240 |
| codebase | 1 | ns-error-mass | 250 | 56 / 194 | yes 8 / no 242 |
| codebase | 2 | crc-error-mass | 250 | 73 / 177 | yes 151 / no 99 |
| codebase | 2 | ns-error-mass | 250 | 73 / 177 | yes 151 / no 99 |
| codebase | 3 | crc-error-mass | 250 | 39 / 211 | yes 66 / no 184 |
| codebase | 3 | ns-error-mass | 250 | 39 / 211 | yes 78 / no 172 |
| twitter_hate | 1 | crc-error-mass | 500 | 174 / 326 | yes 412 / no 88 |
| twitter_hate | 1 | ns-error-mass | 500 | 174 / 326 | yes 410 / no 90 |

生成目录：

- `experiments/inputs/codebase/query_id_1/qwen06b_full_guide200_balanced50_seed42_T10_alpha010_train250_two_methods_seed1/`
- `experiments/inputs/codebase/query_id_2/qwen06b_full_guide200_balanced50_seed42_T10_alpha010_train250_two_methods_seed1/`
- `experiments/inputs/codebase/query_id_3/qwen06b_full_guide200_balanced50_seed42_T10_alpha010_train250_two_methods_seed1/`
- `experiments/inputs/twitter_hate/qwen06b_full_guide500_balanced50_seed42_T10_alpha010_train500_two_methods_seed1/`
