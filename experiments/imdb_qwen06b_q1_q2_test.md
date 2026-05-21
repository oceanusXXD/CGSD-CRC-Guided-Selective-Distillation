# IMDB 0.6B q1/q2 实验草稿

更新时间：2026-05-21。

## 1. 数据状态

本文件记录 IMDB `Qwen3-0.6B` 在 `query_id_1` 和 `query_id_2` 上的 round0、CRC、训练集构造、LoRA 训练和 vLLM 评测计划。当前 q1/q2 的 embedding、0.6B round0、`T=1..10` sweep、`T=10` 的 2500 条训练集构造，以及四个 LoRA 训练 run 均已完成；下一步使用全集 eval split 跑 round1 vLLM 评测。base-wrong split 只作为统计和差异分析口径保留。

`query_id_3` 暂不纳入主实验：该 query 的标签噪声较明显，当前只保留为后续清洗或消融候选。

| 数据集 | query_id | base model | 输入目录 | 数据 | embedding | 0.6B round0 |
| --- | ---: | --- | --- | --- | --- | --- |
| imdb | 1 | `Qwen3-0.6B` | `experiments/inputs/imdb/query_id_1/` | 已补齐 | 已补齐 | 已补齐 |
| imdb | 2 | `Qwen3-0.6B` | `experiments/inputs/imdb/query_id_2/` | 已补齐 | 已补齐 | 已补齐 |

核心输入文件：

- `experiments/inputs/imdb/manifest.json`
- `experiments/inputs/imdb/query_id_1/data.jsonl`
- `experiments/inputs/imdb/query_id_1/source.metadata.json`
- `experiments/inputs/imdb/query_id_1/prepare_summary.json`
- `experiments/inputs/imdb/query_id_1/qwen06b_full_guide1000_seed42_split_ids.json`
- `experiments/inputs/imdb/query_id_2/data.jsonl`
- `experiments/inputs/imdb/query_id_2/source.metadata.json`
- `experiments/inputs/imdb/query_id_2/prepare_summary.json`
- `experiments/inputs/imdb/query_id_2/qwen06b_full_guide1000_seed42_split_ids.json`

## 2. Query 与标签分布

IMDB 每个 query 过滤掉 10 条 `max2048` 超长样本后，保留 49990 条。标签字段为 `groundtruth`，`0` 对应 `no`，`1` 对应 `yes`。

| query_id | query | rows | label0/no | label1/yes |
| ---: | --- | ---: | ---: | ---: |
| 1 | `Is the following review positive?` | 49990 | 24999 (50.01%) | 24991 (49.99%) |
| 2 | `Does the following review explicitly recommend that others watch the movie?` | 49990 | 36247 (72.51%) | 13743 (27.49%) |

来源说明：

- `source`: `Cascade/datasets/public/imdb/ground_truth.json`
- `documents_source`: `Cascade/datasets/public/imdb/documents.json`
- `queries_source`: `Cascade/datasets/public/imdb/queries.json`
- `overlength_removed_max2048`: 10
- `overlength_removed_manifest`: `outputs/runs/overlength_removed_records_max2048.jsonl`

## 3. Embedding 路径

embedding 已用 vLLM pooling/embed 后端生成，路径来自 `experiments/inputs/imdb/manifest.json` 和各 query 的 `prepare_summary.json`。

| query_id | embeddings | ids | meta | 状态 |
| ---: | --- | --- | --- | --- |
| 1 | `experiments/inputs/imdb/query_id_1/embeddings.npy` | `experiments/inputs/imdb/query_id_1/embeddings.ids.jsonl` | `experiments/inputs/imdb/query_id_1/embeddings.meta.json` | 已完成：49990 x 2560 |
| 2 | `experiments/inputs/imdb/query_id_2/embeddings.npy` | `experiments/inputs/imdb/query_id_2/embeddings.ids.jsonl` | `experiments/inputs/imdb/query_id_2/embeddings.meta.json` | 已完成：49990 x 2560 |

构建脚本位置：`experiments/inputs/imdb/build_embeddings.sh`。

默认 embedding 参数：

| 参数 | 值 |
| --- | --- |
| model path | `/teamspace/studios/this_studio/model/qwen3-4b-embedding` |
| backend | `vllm` |
| request batch size | 256 |
| flush rows | 1024 |
| max length | 4096 |
| torch dtype | `bfloat16` |
| tensor parallel size | 1 |
| gpu memory utilization | 0.92 |
| mode | `document` |

## 4. 随机 guide1000 split

本轮使用 `seed=42`、`random_1000_seed42_unbalanced` 的 guide split。`guide_ids` 与 `calibration_ids` 相同；`final_calibration_ids` 当前为空。

| query_id | split path | guide n | guide label0/no | guide label1/yes | pool n | pool label0/no | pool label1/yes |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | `experiments/inputs/imdb/query_id_1/qwen06b_full_guide1000_seed42_split_ids.json` | 1000 | 502 (50.20%) | 498 (49.80%) | 48990 | 24497 (50.00%) | 24493 (50.00%) |
| 2 | `experiments/inputs/imdb/query_id_2/qwen06b_full_guide1000_seed42_split_ids.json` | 1000 | 715 (71.50%) | 285 (28.50%) | 48990 | 35532 (72.53%) | 13458 (27.47%) |

辅助 guide ID 文件：

- `experiments/inputs/imdb/query_id_1/qwen06b_full_guide1000_seed42.ids.json`
- `experiments/inputs/imdb/query_id_2/qwen06b_full_guide1000_seed42.ids.json`

## 5. Round0 输出路径

round0 已完成。每个 query 使用独立 `round_0/` 目录，避免 q1/q2 产物混写。

| query_id | round0 dir | all | guide/calibration | pool | final calibration |
| ---: | --- | --- | --- | --- | --- |
| 1 | `experiments/inputs/imdb/query_id_1/round_0/` | `all_student_predictions.jsonl` | `calibration_student_predictions.jsonl` | `pool_student_predictions.jsonl` | `final_calibration_student_predictions.jsonl` 为空 |
| 2 | `experiments/inputs/imdb/query_id_2/round_0/` | `all_student_predictions.jsonl` | `calibration_student_predictions.jsonl` | `pool_student_predictions.jsonl` | `final_calibration_student_predictions.jsonl` 为空 |

round0 预测命令：

```bash
python scripts/cgsd_predict_vllm_openai.py \
  --output_dir "experiments/inputs/imdb/query_id_${QID}" \
  --round_index 0 \
  --model_path /teamspace/studios/this_studio/model/qwen3-0.6b \
  --data_path "experiments/inputs/imdb/query_id_${QID}/data.jsonl" \
  --split_ids_path "experiments/inputs/imdb/query_id_${QID}/qwen06b_full_guide1000_seed42_split_ids.json" \
  --served_model_name qwen3-0.6b \
  --start_server \
  --base_url http://127.0.0.1:18021/v1 \
  --parallel_requests 8192 \
  --request_retries 3 \
  --timeout 180 \
  --temperature 0 \
  --max_tokens 1 \
  --top_logprobs 20 \
  --max_model_len 4096 \
  --max_num_seqs 4096 \
  --max_num_batched_tokens 524288 \
  --gpu_memory_utilization 0.98 \
  --enforce_eager \
  --cache_policy reuse
```

q1 实际使用 `parallel_requests=4096`；q2 实际使用 `parallel_requests=8192`。两者 `max_num_seqs` 都保持 4096。

| query_id | split | n | pred0 | pred1 | base-error | error-rate | 状态 |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | guide/calibration | 1000 | 0 | 1000 | 502 | 50.20% | 已完成 |
| 1 | pool | 48990 | 5 | 48985 | 24494 | 50.00% | 已完成 |
| 2 | guide/calibration | 1000 | 0 | 1000 | 715 | 71.50% | 已完成 |
| 2 | pool | 48990 | 1 | 48989 | 35531 | 72.53% | 已完成 |

round0 指标：

| query_id | split | n | acc | macro-F1 | F1 no | F1 yes | balanced-acc | TP/TN/FP/FN |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | all | 49990 | 0.5000 | 0.3335 | 0.0003 | 0.6666 | 0.5001 | 24990/4/24995/1 |
| 1 | guide/calibration | 1000 | 0.4980 | 0.3324 | 0.0000 | 0.6649 | 0.5000 | 498/0/502/0 |
| 1 | pool | 48990 | 0.5000 | 0.3335 | 0.0003 | 0.6666 | 0.5001 | 24492/4/24493/1 |
| 2 | all | 49990 | 0.2749 | 0.2157 | 0.0001 | 0.4313 | 0.5000 | 13743/1/36246/0 |
| 2 | guide/calibration | 1000 | 0.2850 | 0.2218 | 0.0000 | 0.4436 | 0.5000 | 285/0/715/0 |
| 2 | pool | 48990 | 0.2747 | 0.2155 | 0.0001 | 0.4310 | 0.5000 | 13458/1/35531/0 |

结论：0.6B round0 在 q1/q2 上几乎全部预测 `yes`。q1 的错误主要是 label0 被预测为 yes；q2 的错误更集中，pool error-rate 为 72.53%，同样几乎全是 label0 -> yes。

## 6. T/alpha/defer 分析占位

目标是为 q1/q2 分别扫描 `T=1..10`，固定 `alpha=0.1`，选择能产生稳定 defer 区间且 guide 错误浓缩合理的参数。CRC 使用已生成的 neighbor-support embedding；本轮已统一选择 `T=10` 进入 2500 条训练集构造和 LoRA 训练。

CRC/neighbor-support 产物路径：

- `experiments/inputs/imdb/query_id_1/round_0/qwen06b_full_guide1000_seed42_T1_T10_defer_rates_ns.json`
- `experiments/inputs/imdb/query_id_2/round_0/qwen06b_full_guide1000_seed42_T1_T10_defer_rates_ns.json`

### q1 T sweep

| T | lambda_hat | pool n | accept | defer | defer比例 | accept错误率 | e_all | e_defer | s_defer |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.880589 | 48990 | 28969 | 20021 | 40.87% | 16.02% | 0.5020 | 0.9877 | 0.6727 |
| 2 | 0.802218 | 48990 | 29273 | 19717 | 40.25% | 17.53% | 0.5020 | 0.9782 | 0.6711 |
| 3 | 0.725897 | 48990 | 29321 | 19669 | 40.15% | 17.78% | 0.5020 | 0.9782 | 0.6710 |
| 4 | 0.670746 | 48990 | 29369 | 19621 | 40.05% | 17.78% | 0.5020 | 0.9805 | 0.6711 |
| 5 | 0.631669 | 48990 | 29465 | 19525 | 39.86% | 17.88% | 0.5020 | 0.9829 | 0.6712 |
| 6 | 0.603647 | 48990 | 29477 | 19513 | 39.83% | 17.78% | 0.5020 | 0.9902 | 0.6719 |
| 7 | 0.583817 | 48990 | 29297 | 19693 | 40.20% | 17.19% | 0.5020 | 0.9902 | 0.6722 |
| 8 | 0.567790 | 48990 | 29255 | 19735 | 40.28% | 17.00% | 0.5020 | 0.9902 | 0.6723 |
| 9 | 0.555426 | 48990 | 29201 | 19789 | 40.39% | 16.80% | 0.5020 | 0.9902 | 0.6724 |
| 10 | 0.545307 | 48990 | 29116 | 19874 | 40.57% | 16.51% | 0.5020 | 0.9902 | 0.6726 |

### q2 T sweep

| T | lambda_hat | pool n | accept | defer | defer比例 | accept错误率 | e_all | e_defer | s_defer |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1.010000 | 48990 | 0 | 48990 | 100.00% | 0.00% | 0.7150 | 0.7150 | 1.0000 |
| 2 | 0.942178 | 48990 | 13302 | 35688 | 72.85% | 35.80% | 0.7150 | 0.8567 | 0.7689 |
| 3 | 0.854713 | 48990 | 12656 | 36334 | 74.17% | 36.32% | 0.7150 | 0.8450 | 0.7769 |
| 4 | 0.789216 | 48990 | 12738 | 36252 | 74.00% | 36.58% | 0.7150 | 0.8462 | 0.7759 |
| 5 | 0.742950 | 48990 | 12815 | 36175 | 73.84% | 36.50% | 0.7150 | 0.8485 | 0.7750 |
| 6 | 0.708996 | 48990 | 13003 | 35987 | 73.46% | 36.36% | 0.7150 | 0.8520 | 0.7727 |
| 7 | 0.684469 | 48990 | 12931 | 36059 | 73.60% | 35.90% | 0.7150 | 0.8497 | 0.7734 |
| 8 | 0.665446 | 48990 | 12933 | 36057 | 73.60% | 35.54% | 0.7150 | 0.8497 | 0.7734 |
| 9 | 0.648868 | 48990 | 13430 | 35560 | 72.59% | 35.70% | 0.7150 | 0.8591 | 0.7673 |
| 10 | 0.637072 | 48990 | 13296 | 35694 | 72.86% | 35.41% | 0.7150 | 0.8591 | 0.7693 |

## 7. random vs ns-error-mass 训练集构造

本轮主对比只预留两种方法：

- `pool-random`：从 pool 中均匀随机抽样。
- `ns-error-mass`：沿用 CRC 推出的 accept/defer 预算分配，并在 accept/defer 内部按 `ns_p_error` 做无放回加权抽样。

预算固定为 `2500`。q1/q2 都选择 `T=10`、`alpha=0.1`；只构造 `pool-random` 和 `ns-error-mass` 两种 2500 条训练集。

输出目录：

- `experiments/inputs/imdb/query_id_1/qwen06b_full_guide1000_seed42_T10_alpha010_train2500_two_methods_seed1/`
- `experiments/inputs/imdb/query_id_2/qwen06b_full_guide1000_seed42_T10_alpha010_train2500_two_methods_seed1/`

训练集构成：

| query_id | train n | method | selected n | accept | defer | defer比例 | base-error | error-rate | label0/no | label1/yes | summary |
| ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | 2500 | `pool-random` | 2500 | 1499 | 1001 | 40.04% | 1230 | 49.20% | 1231 | 1269 | `experiments/inputs/imdb/query_id_1/qwen06b_full_guide1000_seed42_T10_alpha010_train2500_two_methods_seed1/seed1/pool_random_2500_seed1.summary.json` |
| 1 | 2500 | `ns-error-mass` | 2500 | 819 | 1681 | 67.24% | 2315 | 92.60% | 2316 | 184 | `experiments/inputs/imdb/query_id_1/qwen06b_full_guide1000_seed42_T10_alpha010_train2500_two_methods_seed1/seed1/ns_error_mass_2500_seed1.summary.json` |
| 2 | 2500 | `pool-random` | 2500 | 697 | 1803 | 72.12% | 1805 | 72.20% | 1805 | 695 | `experiments/inputs/imdb/query_id_2/qwen06b_full_guide1000_seed42_T10_alpha010_train2500_two_methods_seed1/seed1/pool_random_2500_seed1.summary.json` |
| 2 | 2500 | `ns-error-mass` | 2500 | 577 | 1923 | 76.92% | 1975 | 79.00% | 1975 | 525 | `experiments/inputs/imdb/query_id_2/qwen06b_full_guide1000_seed42_T10_alpha010_train2500_two_methods_seed1/seed1/ns_error_mass_2500_seed1.summary.json` |

聚合 summary：

- `experiments/inputs/imdb/query_id_1/qwen06b_full_guide1000_seed42_T10_alpha010_train2500_two_methods_seed1/train2500_two_methods_summary.json`
- `experiments/inputs/imdb/query_id_2/qwen06b_full_guide1000_seed42_T10_alpha010_train2500_two_methods_seed1/train2500_two_methods_summary.json`

## 8. LoRA 训练

四个 `T=10`、`alpha=0.1`、train n=2500 的 LoRA run 已完成。参数名已按 `scripts/cgsd_train_round.py` 和 `scripts/cgsd_cli_common.py` 的 argparse 接口核对。

| 参数 | 值 |
| --- | --- |
| base model | `Qwen3-0.6B` |
| model path | `/teamspace/studios/this_studio/model/qwen3-0.6b` |
| input format | `cgsd_chat_binary_v1` |
| round index | 1 |
| epochs | 4 |
| learning rate | `1e-4` |
| LoRA r | 8 |
| LoRA alpha | 16 |
| LoRA dropout | 0.05 |
| target modules | `attention_mlp` |
| lora layer | all |
| max length | 4096 |
| train batch size | 4 |
| gradient accumulation steps | 6 |
| pad to multiple of | 8 |
| seed | 42 |
| precision | `bfloat16` |
| cache policy | `reuse` |

命令骨架：

```bash
python scripts/cgsd_train_round.py \
  --output_dir "$RUN_DIR" \
  --round_index 1 \
  --model_path /teamspace/studios/this_studio/model/qwen3-0.6b \
  --data_path "experiments/inputs/imdb/query_id_${QID}/data.jsonl" \
  --split_ids_path "experiments/inputs/imdb/query_id_${QID}/qwen06b_full_guide1000_seed42_split_ids.json" \
  --train_rows_path "$TRAIN_ROWS" \
  --max_length 4096 \
  --epochs 4 \
  --lr 1e-4 \
  --batch_size 4 \
  --gradient_accumulation_steps 6 \
  --pad_to_multiple_of 8 \
  --torch_dtype bfloat16 \
  --lora_r 8 \
  --lora_alpha 16 \
  --lora_dropout 0.05 \
  --lora_target_modules attention_mlp \
  --lora_layer_scope all \
  --seed 42 \
  --cache_policy reuse
```

实际 run root：

- `experiments/runs/imdb_qwen06b_T10_alpha010_train2500_lora/`

训练产物：

| query_id | method | train rows | checkpoint | 训练记录 |
| ---: | --- | ---: | --- | --- |
| 1 | `pool-random` | 2500 | `experiments/runs/imdb_qwen06b_T10_alpha010_train2500_lora/q1_pool_random/round_1/model` | `experiments/runs/imdb_qwen06b_T10_alpha010_train2500_lora/q1_pool_random/round_1/training_round_summary.json` |
| 1 | `ns-error-mass` | 2500 | `experiments/runs/imdb_qwen06b_T10_alpha010_train2500_lora/q1_ns_error_mass/round_1/model` | `experiments/runs/imdb_qwen06b_T10_alpha010_train2500_lora/q1_ns_error_mass/round_1/training_round_summary.json` |
| 2 | `pool-random` | 2500 | `experiments/runs/imdb_qwen06b_T10_alpha010_train2500_lora/q2_pool_random/round_1/model` | `experiments/runs/imdb_qwen06b_T10_alpha010_train2500_lora/q2_pool_random/round_1/training_round_summary.json` |
| 2 | `ns-error-mass` | 2500 | `experiments/runs/imdb_qwen06b_T10_alpha010_train2500_lora/q2_ns_error_mass/round_1/model` | `experiments/runs/imdb_qwen06b_T10_alpha010_train2500_lora/q2_ns_error_mass/round_1/training_round_summary.json` |

校验结果：四个 run 的 `training_rows_used.jsonl` 均为 2500 行，`selection_round_counts={"0": 2500}`，adapter 为 LoRA `r=8`、`alpha=16`，target modules 展开为 `q_proj/k_proj/v_proj/o_proj/gate_proj/up_proj/down_proj`。

## 9. vLLM 评测

四个 round1 vLLM full eval 已完成。参数名已按 `scripts/cgsd_predict_vllm_openai.py` 的 argparse 接口核对。

本轮 round1 vLLM 评测使用全集测试集，`pool_ids/test_ids` 都是该 query 的 49990 条样本，`calibration_ids` 为空：

| query_id | split 文件 | test n | label0/no | label1/yes |
| ---: | --- | ---: | ---: | ---: |
| 1 | `experiments/eval_splits/imdb_q1_full_all_split_ids.json` | 49990 | 24999 | 24991 |
| 2 | `experiments/eval_splits/imdb_q2_full_all_split_ids.json` | 49990 | 36247 | 13743 |

base-wrong split 只用于统计“base 做错全集”的构成，不作为主评测集：

| query_id | stats split 文件 | base-wrong n | label0/no | label1/yes |
| ---: | --- | ---: | ---: | ---: |
| 1 | `experiments/eval_splits/imdb_q1_base_wrong_all_split_ids.json` | 24996 | 24995 | 1 |
| 2 | `experiments/eval_splits/imdb_q2_base_wrong_all_split_ids.json` | 36246 | 36246 | 0 |

| 参数 | 值 |
| --- | --- |
| base URL | `http://127.0.0.1:18021/v1` |
| api key | `EMPTY` |
| model path | `/teamspace/studios/this_studio/model/qwen3-0.6b` |
| checkpoint dir | LoRA run 的 `round_1/model` |
| round index | 1 |
| temperature | 0.0 |
| max tokens | 1 |
| top logprobs | 20 |
| parallel requests | 8192 |
| request retries | 3 |
| timeout | 180 |
| max model len | 4096 |
| max num seqs | 4096 |
| max num batched tokens | 1048576 |
| gpu memory utilization | 0.98 |
| enforce eager | true |

命令骨架：

```bash
python scripts/cgsd_predict_vllm_openai.py \
  --output_dir "$EVAL_DIR" \
  --round_index 1 \
  --model_path /teamspace/studios/this_studio/model/qwen3-0.6b \
  --data_path "experiments/inputs/imdb/query_id_${QID}/data.jsonl" \
  --split_ids_path "$EVAL_SPLIT_IDS" \
  --checkpoint_dir "$RUN_DIR/round_1/model" \
  --base_url http://127.0.0.1:18021/v1 \
  --api_key EMPTY \
  --temperature 0.0 \
  --max_tokens 1 \
  --top_logprobs 20 \
  --parallel_requests 8192 \
  --request_retries 3 \
  --timeout 180 \
  --max_model_len 4096 \
  --max_num_seqs 4096 \
  --max_num_batched_tokens 1048576 \
  --gpu_memory_utilization 0.98
```

输出目录：

- `experiments/evals/imdb_qwen06b_T10_alpha010_train2500_lora/q1_pool_random_full/`
- `experiments/evals/imdb_qwen06b_T10_alpha010_train2500_lora/q1_ns_error_mass_full/`
- `experiments/evals/imdb_qwen06b_T10_alpha010_train2500_lora/q2_pool_random_full/`
- `experiments/evals/imdb_qwen06b_T10_alpha010_train2500_lora/q2_ns_error_mass_full/`

聚合结果：

- `experiments/reports/imdb_qwen06b_T10_alpha010_train2500_full_eval_summary.json`

## 10. 结果表格

round1 full eval 详细指标：

| query_id | method | n | acc | macro-F1 | F1 no | F1 yes | balanced-acc | pred0 | pred1 | TP/TN/FP/FN |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | `pool-random` | 49990 | 94.37% | 94.37% | 94.34% | 94.40% | 94.37% | 24,751 | 25,239 | 23,708/23,468/1,531/1,283 |
| 1 | `ns-error-mass` | 49990 | 92.27% | 92.25% | 92.60% | 91.91% | 92.27% | 27,202 | 22,788 | 21,957/24,168/831/3,034 |
| 2 | `pool-random` | 49990 | 88.69% | 85.72% | 92.24% | 79.20% | 85.47% | 36,557 | 13,433 | 10,762/33,576/2,671/2,981 |
| 2 | `ns-error-mass` | 49990 | 88.70% | 85.54% | 92.30% | 78.78% | 84.85% | 37,119 | 12,871 | 10,483/33,859/2,388/3,260 |

按 base 原本是否正确拆分：

base 原本正确集合，主要看 LoRA 是否破坏原本正确样本。

| query_id | method | train n | subset n | lora still correct | lora flips wrong | lora acc on subset | pred0 | pred1 | TP/TN/FP/FN |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | `pool-random` | 2500 | 24,994 | 23,711 | 1,283 | 94.87% | 1,287 | 23,707 | 23,707/4/0/1,283 |
| 1 | `ns-error-mass` | 2500 | 24,994 | 21,960 | 3,034 | 87.86% | 3,038 | 21,956 | 21,956/4/0/3,034 |
| 2 | `pool-random` | 2500 | 13,744 | 10,763 | 2,981 | 78.31% | 2,982 | 10,762 | 10,762/1/0/2,981 |
| 2 | `ns-error-mass` | 2500 | 13,744 | 10,484 | 3,260 | 76.28% | 3,261 | 10,483 | 10,483/1/0/3,260 |

base 原本错误集合，主要看 LoRA 能纠正多少原本错误样本。

| query_id | method | train n | subset n | lora corrects base wrong | still wrong | lora acc on subset | pred0 | pred1 | TP/TN/FP/FN |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | `pool-random` | 2500 | 24,996 | 23,465 | 1,531 | 93.88% | 23,464 | 1,532 | 1/23,464/1,531/0 |
| 1 | `ns-error-mass` | 2500 | 24,996 | 24,165 | 831 | 96.68% | 24,164 | 832 | 1/24,164/831/0 |
| 2 | `pool-random` | 2500 | 36,246 | 33,575 | 2,671 | 92.63% | 33,575 | 2,671 | 0/33,575/2,671/0 |
| 2 | `ns-error-mass` | 2500 | 36,246 | 33,858 | 2,388 | 93.41% | 33,858 | 2,388 | 0/33,858/2,388/0 |
