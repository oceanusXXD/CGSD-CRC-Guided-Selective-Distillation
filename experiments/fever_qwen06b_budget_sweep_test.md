# FEVER 0.6B Budget Sweep 数据/实验状态

更新时间：2026-05-21。

## 1. 数据状态

本文件记录 FEVER `Qwen3-0.6B` round0 cache 上的训练预算扫描。当前正式保留 `pool-random` 和 `ns-error-mass` 两种方法，各 6 档训练量，一共 12 个训练集。

| 数据集 | base model | 输入目录 | 数据 | embedding | 0.6B round0 | 生成目录 |
| --- | --- | --- | --- | --- | --- | --- |
| fever | `Qwen3-0.6B` | `experiments/inputs/fever/` | 已补齐 | 已补齐，2560 维 | 已补齐 | `experiments/inputs/fever/qwen06b_round0_t1_alpha010_budget_sweep_1500_9000_seed1/` |

核心输入文件：

- `experiments/inputs/fever/data.jsonl`
- `experiments/inputs/fever/embeddings.npy`
- `experiments/inputs/fever/embeddings.ids.jsonl`
- `experiments/inputs/fever/embeddings.meta.json`
- `experiments/inputs/fever/cgsd_split_ids.json`
- `experiments/inputs/fever/round_0/all_student_predictions.jsonl`
- `experiments/inputs/fever/round_0/calibration_student_predictions.jsonl`
- `experiments/inputs/fever/round_0/pool_student_predictions.jsonl`
- `experiments/inputs/fever/round_0/final_calibration_student_predictions.jsonl`

## 2. Split 与预算

`cgsd_split_ids.json` 使用 seed=1。训练预算都从 pool 中选取；3000 条相当于 pool 的 1.83%，相当于 round0 全量的 1.81%。

| split | n | 说明 |
| --- | ---: | --- |
| calibration | 1000 | CRC calibration/guide |
| final_calibration | 200 | 保留校准/评估用 |
| pool | 164247 | 本次 12 个训练集都从这里抽取 |
| round0 all | 165447 | calibration + pool + final_calibration |

| train n | 占 pool 比例 | 占 round0 全量比例 |
| ---: | ---: | ---: |
| 1500 | 0.91% | 0.91% |
| 3000 | 1.83% | 1.81% |
| 4500 | 2.74% | 2.72% |
| 6000 | 3.65% | 3.63% |
| 7500 | 4.57% | 4.53% |
| 9000 | 5.48% | 5.44% |

## 3. Guide 集分析

本轮的 guide 集就是 `calibration` split，大小为 1000。它没有作为 LoRA 训练集样本参与 12 个 budget sweep 训练集；它只用于 CRC 校准、估计 `lambda_hat`、计算 guide 上的 accept/defer/error 统计，并作为 pool 样本 neighbor-support 的 support set。

| split | n | label0 | label1 | pred0 | pred1 | base-error | error-rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| guide/calibration | 1000 | 461 | 539 | 892 | 108 | 479 | 47.90% |
| final_calibration | 200 | 94 | 106 | 176 | 24 | 92 | 46.00% |
| pool | 164247 | 78191 | 86056 | 147824 | 16423 | 78871 | 48.02% |

在 `T=1`、`alpha=0.1` 下，guide 经 CRC routing 后的组成如下：

| split | n | accept | defer | defer比例 | defer error | defer error-rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| guide/calibration | 1000 | 287 | 713 | 71.30% | 380 | 53.30% |

对应文件：

- guide 原始 round0 预测：`experiments/inputs/fever/round_0/calibration_student_predictions.jsonl`
- guide CRC 判定：`experiments/inputs/fever/qwen06b_round0_t1_alpha010_budget_sweep_1500_9000_seed1/seed1/round_0/guide_crc_predictions.jsonl`
- split 定义：`experiments/inputs/fever/cgsd_split_ids.json`

## 4. CRC 参数选择

目标是让 0.6B 在 pool 上的 defer 比例落在 60% 到 85% 之间。已扫描 `T` 和 `alpha`，本轮采用 `T=1`，`alpha=0.1`：defer 比例为 72.24%，在目标区间内，而且与前面 FEVER/TwitterHate 的主实验参数保持一致。

扫描缓存文件：`experiments/inputs/fever/qwen06b_round0_crc_sweep_tmp.json`。

| T | alpha | lambda_hat | pool n | accept | defer | defer比例 | accept错误率 | e_all | e_defer |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.1 | 0.906121 | 164247 | 45599 | 118648 | 72.24% | 37.16% | 47.90% | 53.30% |

本次 `ns-error-mass` 使用 CRC 推出的自适应 accept/defer 预算分配，并在 accept/defer 内部按 `ns_p_error` 做无放回加权抽样。前一版 `crc-error-mass` 只比 `pool-random` 轻微提高 defer 比例，selected base-error 与 random 很接近，因此不再作为正式对比方法保留。

| 参数 | 值 |
| --- | ---: |
| calibration n | 1000 |
| calibration defer n | 713 |
| calibration error n | 479 |
| calibration defer error n | 380 |
| r_C | 0.713 |
| r_U | 0.722375 |
| tau_crc | 2.267171 |
| c_crc | 1.112650 |
| eta_crc | 0.315556 |
| s_accept | 0.253303 |
| s_defer | 0.746697 |

## 5. 训练集构成

生成命令：

```bash
python scripts/cgsd_make_fever_budget_sweep_sets.py \
  --output_dir experiments/inputs/fever/qwen06b_round0_t1_alpha010_budget_sweep_1500_9000_seed1 \
  --budgets 1500,3000,4500,6000,7500,9000 \
  --seed 1 \
  --temperature 1 \
  --alpha 0.1 \
  --embedding_dim 2560
```

| train n | method | selected n | accept | defer | defer比例 | base-error | error-rate | label0 | label1 |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1500 | pool-random | 1500 | 434 | 1066 | 71.07% | 695 | 46.33% | 715 | 785 |
| 1500 | ns-error-mass | 1500 | 380 | 1120 | 74.67% | 984 | 65.60% | 508 | 992 |
| 3000 | pool-random | 3000 | 848 | 2152 | 71.73% | 1427 | 47.57% | 1432 | 1568 |
| 3000 | ns-error-mass | 3000 | 760 | 2240 | 74.67% | 2021 | 67.37% | 959 | 2041 |
| 4500 | pool-random | 4500 | 1262 | 3238 | 71.96% | 2135 | 47.44% | 2134 | 2366 |
| 4500 | ns-error-mass | 4500 | 1140 | 3360 | 74.67% | 3042 | 67.60% | 1427 | 3073 |
| 6000 | pool-random | 6000 | 1670 | 4330 | 72.17% | 2853 | 47.55% | 2855 | 3145 |
| 6000 | ns-error-mass | 6000 | 1520 | 4480 | 74.67% | 4049 | 67.48% | 1908 | 4092 |
| 7500 | pool-random | 7500 | 2080 | 5420 | 72.27% | 3617 | 48.23% | 3535 | 3965 |
| 7500 | ns-error-mass | 7500 | 1900 | 5600 | 74.67% | 5066 | 67.55% | 2391 | 5109 |
| 9000 | pool-random | 9000 | 2510 | 6490 | 72.11% | 4305 | 47.83% | 4274 | 4726 |
| 9000 | ns-error-mass | 9000 | 2280 | 6720 | 74.67% | 6089 | 67.66% | 2862 | 6138 |

### 5.1 与 random 的差异

`ns-error-mass` 使用 `ns_p_error` 加权后，selected base-error 相比 `pool-random` 明显提高。

| train n | ns-error-mass vs random base-error |
| ---: | ---: |
| 1500 | +19.27 pp |
| 3000 | +19.80 pp |
| 4500 | +20.16 pp |
| 6000 | +19.93 pp |
| 7500 | +19.32 pp |
| 9000 | +19.82 pp |

| train n | method | selected mean ns_p_error |
| ---: | --- | ---: |
| 1500 | pool-random | 0.4812 |
| 1500 | ns-error-mass | 0.6898 |
| 3000 | pool-random | 0.4842 |
| 3000 | ns-error-mass | 0.6953 |
| 4500 | pool-random | 0.4862 |
| 4500 | ns-error-mass | 0.6954 |
| 6000 | pool-random | 0.4878 |
| 6000 | ns-error-mass | 0.6936 |
| 7500 | pool-random | 0.4908 |
| 7500 | ns-error-mass | 0.6939 |
| 9000 | pool-random | 0.4898 |
| 9000 | ns-error-mass | 0.6928 |

## 6. 训练集文件索引

汇总文件：`experiments/inputs/fever/qwen06b_round0_t1_alpha010_budget_sweep_1500_9000_seed1/budget_sweep_summary.json`。

| train n | method | train rows | summary |
| ---: | --- | --- | --- |
| 1500 | pool-random | `seed1/pool_random_1500_seed1.train_rows.jsonl` | `seed1/pool_random_1500_seed1.summary.json` |
| 1500 | ns-error-mass | `seed1/ns_error_mass_1500_seed1.train_rows.jsonl` | `seed1/ns_error_mass_1500_seed1.summary.json` |
| 3000 | pool-random | `seed1/pool_random_3000_seed1.train_rows.jsonl` | `seed1/pool_random_3000_seed1.summary.json` |
| 3000 | ns-error-mass | `seed1/ns_error_mass_3000_seed1.train_rows.jsonl` | `seed1/ns_error_mass_3000_seed1.summary.json` |
| 4500 | pool-random | `seed1/pool_random_4500_seed1.train_rows.jsonl` | `seed1/pool_random_4500_seed1.summary.json` |
| 4500 | ns-error-mass | `seed1/ns_error_mass_4500_seed1.train_rows.jsonl` | `seed1/ns_error_mass_4500_seed1.summary.json` |
| 6000 | pool-random | `seed1/pool_random_6000_seed1.train_rows.jsonl` | `seed1/pool_random_6000_seed1.summary.json` |
| 6000 | ns-error-mass | `seed1/ns_error_mass_6000_seed1.train_rows.jsonl` | `seed1/ns_error_mass_6000_seed1.summary.json` |
| 7500 | pool-random | `seed1/pool_random_7500_seed1.train_rows.jsonl` | `seed1/pool_random_7500_seed1.summary.json` |
| 7500 | ns-error-mass | `seed1/ns_error_mass_7500_seed1.train_rows.jsonl` | `seed1/ns_error_mass_7500_seed1.summary.json` |
| 9000 | pool-random | `seed1/pool_random_9000_seed1.train_rows.jsonl` | `seed1/pool_random_9000_seed1.summary.json` |
| 9000 | ns-error-mass | `seed1/ns_error_mass_9000_seed1.train_rows.jsonl` | `seed1/ns_error_mass_9000_seed1.summary.json` |

完整文件每组还包括 `.ids.json` 和 `.jsonl`。相对路径均基于 `experiments/inputs/fever/qwen06b_round0_t1_alpha010_budget_sweep_1500_9000_seed1/`。

## 7. LoRA 训练参数

后续 12 个 FEVER 数据集都使用同一套 LoRA 训练参数。训练 prompt 为 `format_cgsd_chat_prompt(query, document)`，监督答案只接 `1<|im_end|>` 或 `0<|im_end|>`；prompt 部分 mask 为 `-100`，loss 只覆盖答案部分。

| 参数 | 值 |
| --- | --- |
| base model | `Qwen3-0.6B` |
| model path | `model/qwen3-0.6b` |
| input format | `cgsd_chat_binary_v1` |
| epochs | 4 |
| learning rate | `1e-4` |
| LoRA r | 8 |
| LoRA alpha | 16 |
| LoRA dropout | 0.05 |
| target modules | `attention_mlp` |
| lora layer | all |
| max length | 4096 |
| per-device train batch size | 2 |
| gradient accumulation steps | 8 |
| pad to multiple of | 8 |
| seed | 42 |
| precision | `bf16` |

建议输出目录：

`experiments/runs/fever_lora_qwen06b_round0_t1_alpha010_budget_sweep/lr1e-4_e4_r8_a16_all_ml4096_bs2_ga8/`

### 7.1 训练命令

默认正式 run 使用 `max_length=4096`，与当前测试集的 `max_model_len=4096` 口径一致。若要完全避免训练截断，把下面的 `TRAIN_MAX_LENGTH=4096` 改成 `8192`，并同步改输出目录名中的 `ml4096`。

```bash
export INPUT_ROOT=experiments/inputs/fever/qwen06b_round0_t1_alpha010_budget_sweep_1500_9000_seed1
export RUN_ROOT=experiments/runs/fever_lora_qwen06b_round0_t1_alpha010_budget_sweep/lr1e-4_e4_r8_a16_all_ml4096_bs2_ga8
export MODEL_PATH=/teamspace/studios/this_studio/model/qwen3-0.6b
export DATA_PATH=experiments/inputs/fever/data.jsonl
export TRAIN_MAX_LENGTH=4096

mkdir -p "$RUN_ROOT"

for BUDGET in 1500 3000 4500 6000 7500 9000; do
  for METHOD in pool-random ns-error-mass; do
    METHOD_TAG="${METHOD//-/_}"
    RUN_NAME="${METHOD_TAG}_${BUDGET}_seed1"
    TRAIN_ROWS="$INPUT_ROOT/seed1/${RUN_NAME}.train_rows.jsonl"
    RUN_DIR="$RUN_ROOT/${RUN_NAME}"

    python scripts/cgsd_train_round.py \
      --output_dir "$RUN_DIR" \
      --round_index 1 \
      --model_path "$MODEL_PATH" \
      --data_path "$DATA_PATH" \
      --split_ids_path "$INPUT_ROOT/seed1/split_ids.json" \
      --train_rows_path "$TRAIN_ROWS" \
      --max_length "$TRAIN_MAX_LENGTH" \
      --epochs 4 \
      --lr 1e-4 \
      --batch_size 2 \
      --gradient_accumulation_steps 8 \
      --pad_to_multiple_of 8 \
      --torch_dtype bfloat16 \
      --lora_r 8 \
      --lora_alpha 16 \
      --lora_dropout 0.05 \
      --lora_target_modules attention_mlp \
      --lora_layer_scope all \
      --seed 42 \
      --cache_policy reuse
  done
done
```

### 7.2 max length 校验

校验使用 `/teamspace/studios/this_studio/model/qwen3-0.6b` tokenizer，输入格式为 `cgsd_chat_binary_v1`。`train_seq_tokens` 等于 prompt tokens 加监督答案 tokens；当前答案 `0<|im_end|>` / `1<|im_end|>` 均为 2 tokens。

12 个正式训练集按 run 累计共有 63000 行，去重后 18043 个样本。`max_length=4096` 会截断 78 个 run-row，占 0.12%；按去重样本算是 21 个样本，占 0.12%。如果要完全不截断当前训练集，需要把 `max_length` 提到 8192。

| 范围 | n | p50 | p90 | p95 | p99 | max | >4096 | >3072 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 12 个 train run-row | 63000 | 439 | 930 | 1150 | 1837 | 7978 | 78 | 177 |
| 去重 train sample | 18043 |  |  |  |  | 7978 | 21 | 50 |
| 4096-safe test | 10000 | 361 | 911 | 1103 | 1787 | 4023 | 0 | 12 |

最长训练样本：

| id | train seq tokens | prompt tokens | label |
| --- | ---: | ---: | ---: |
| fever_evidence_79366 | 7978 | 7976 | 1 |
| fever_evidence_104782 | 7099 | 7097 | 0 |
| fever_evidence_63731 | 6640 | 6638 | 0 |
| fever_evidence_152479 | 6367 | 6365 | 1 |
| fever_evidence_149091 | 6333 | 6331 | 1 |

## 8. Vllm 参数

后续所有 LoRA checkpoint 都使用同一套 vLLM 评测参数。

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
| max model len | 4096 |
| max num seqs | 4096 |
| max num batched tokens | 524288 |
| GPU memory utilization | 0.98 |
| enforce eager | true |
| 输出格式 | 单 token 二分类预测，读取 `0` / `1` |

### 8.1 vLLM 评测命令

评测使用第 9 节的 10000 条 `max_model_len=4096` 安全测试集。下面的 loop 会对 12 个 LoRA checkpoint 逐个启动 vLLM server 并写出 `round_1/pool_student_predictions.jsonl`；如果已有常驻 vLLM server，可去掉 `--start_server` 并把 `--base_url` 指向现有服务。

```bash
export INPUT_ROOT=experiments/inputs/fever/qwen06b_round0_t1_alpha010_budget_sweep_1500_9000_seed1
export RUN_ROOT=experiments/runs/fever_lora_qwen06b_round0_t1_alpha010_budget_sweep/lr1e-4_e4_r8_a16_all_ml4096_bs2_ga8
export MODEL_PATH=/teamspace/studios/this_studio/model/qwen3-0.6b
export DATA_PATH=experiments/inputs/fever/data.jsonl
export TEST_SPLIT="$INPUT_ROOT/seed1/test_sets/balanced_test_10000_ml4096safe_exclude_guide_final_train12_seed20260521.vllm_pool_split_ids.json"

for BUDGET in 1500 3000 4500 6000 7500 9000; do
  for METHOD in pool-random ns-error-mass; do
    METHOD_TAG="${METHOD//-/_}"
    RUN_NAME="${METHOD_TAG}_${BUDGET}_seed1"
    TRAIN_ROWS="$INPUT_ROOT/seed1/${RUN_NAME}.train_rows.jsonl"
    RUN_DIR="$RUN_ROOT/${RUN_NAME}"

    python scripts/cgsd_predict_vllm_openai.py \
      --output_dir "$RUN_DIR" \
      --round_index 1 \
      --model_path "$MODEL_PATH" \
      --data_path "$DATA_PATH" \
      --split_ids_path "$TEST_SPLIT" \
      --selected_train_rows_path "$TRAIN_ROWS" \
      --served_model_name qwen3-0.6b \
      --lora_model_name "fever_${RUN_NAME}_round1" \
      --start_server \
      --base_url http://127.0.0.1:18021/v1 \
      --parallel_requests 4096 \
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
  done
done
```

如果评测 hard stress test，不要写回同一个 `RUN_DIR/round_1/pool_student_predictions.jsonl`，否则会和 balanced test 的 cache/output 混在一起。hard test 建议单独写到 eval 子目录，并显式指定 LoRA checkpoint：

```bash
export INPUT_ROOT=experiments/inputs/fever/qwen06b_round0_t1_alpha010_budget_sweep_1500_9000_seed1
export RUN_ROOT=experiments/runs/fever_lora_qwen06b_round0_t1_alpha010_budget_sweep/lr1e-4_e4_r8_a16_all_ml4096_bs2_ga8
export MODEL_PATH=/teamspace/studios/this_studio/model/qwen3-0.6b
export DATA_PATH=experiments/inputs/fever/data.jsonl
export TEST_SPLIT="$INPUT_ROOT/seed1/test_sets/hard70err_test_10000_ml4096safe_exclude_guide_final_train12_seed20260521.vllm_pool_split_ids.json"
export EVAL_TAG=eval_hard70err_test10000

for BUDGET in 1500 3000 4500 6000; do
  for METHOD in pool-random ns-error-mass; do
    METHOD_TAG="${METHOD//-/_}"
    RUN_NAME="${METHOD_TAG}_${BUDGET}_seed1"
    TRAIN_ROWS="$INPUT_ROOT/seed1/${RUN_NAME}.train_rows.jsonl"
    RUN_DIR="$RUN_ROOT/${RUN_NAME}"
    EVAL_DIR="$RUN_DIR/$EVAL_TAG"

    python scripts/cgsd_predict_vllm_openai.py \
      --output_dir "$EVAL_DIR" \
      --round_index 1 \
      --checkpoint_dir "$RUN_DIR/round_1/model" \
      --model_path "$MODEL_PATH" \
      --data_path "$DATA_PATH" \
      --split_ids_path "$TEST_SPLIT" \
      --selected_train_rows_path "$TRAIN_ROWS" \
      --served_model_name qwen3-0.6b \
      --lora_model_name "fever_${RUN_NAME}_round1_hard70err" \
      --start_server \
      --base_url http://127.0.0.1:18021/v1 \
      --parallel_requests 4096 \
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
  done
done
```

## 9. 测试集口径

结果比较时应使用同一个 FEVER 测试集，并排除 `calibration`、`final_calibration` 以及 12 个训练集的并集，避免训练样本泄漏到评测中。

已生成两个 10000 条测试集。`balanced_test` 用于常规同标签平衡评测；`hard70err_test` 用于 stress test，直接按 round0 真实错误选题，保证 error/correct = 7/3，不再强行做 label 0/1 平衡。两者都排除 guide、final_calibration 和 12 个正式训练集的并集，并保证 `cgsd_chat_binary_v1` 下 `train_seq_tokens <= 4096`，避免 vLLM `max_model_len=4096` 评测时报超长 prompt。

| 测试集 | 状态 | split 文件 | n | label0 | label1 | base correct | base error | defer | max seq tokens |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| balanced_test_10000_ml4096safe_exclude_guide_final_train12_seed20260521 | 已生成 | `experiments/inputs/fever/qwen06b_round0_t1_alpha010_budget_sweep_1500_9000_seed1/seed1/test_sets/balanced_test_10000_ml4096safe_exclude_guide_final_train12_seed20260521.vllm_pool_split_ids.json` | 10000 | 5000 | 5000 | 5436 | 4564 / 45.64% | 7200 / 72.00% | 4023 |
| hard70err_test_10000_ml4096safe_exclude_guide_final_train12_seed20260521 | 已生成 | `experiments/inputs/fever/qwen06b_round0_t1_alpha010_budget_sweep_1500_9000_seed1/seed1/test_sets/hard70err_test_10000_ml4096safe_exclude_guide_final_train12_seed20260521.vllm_pool_split_ids.json` | 10000 | 3013 | 6987 | 3000 | 7000 / 70.00% | 7498 / 74.98% | 3966 |

`hard70err_test` 的选择规则：

- 只用 round0 真实错误信号：`prediction != groundtruth`。
- 从 maxlen-safe 且未被 guide/final/train-union 占用的 pool 中，随机抽 7000 条 base-error 和 3000 条 base-correct。
- 不做 0/1 label 平衡；当前 label 分布自然变成 label0=3013、label1=6987，因为 0.6B round0 的 FEVER 错误主要集中在 label1。
- 由于 hard test 标签不平衡，后续主看 `macro-F1`、`balanced-acc` 和分标签 F1；raw acc 只作辅助。

生成命令：

```bash
python scripts/cgsd_make_fever_hard_test_set.py
```

测试集候选池统计：

| 项 | 数量 |
| --- | ---: |
| excluded calibration | 1000 |
| excluded final_calibration | 200 |
| excluded formal train union | 18043 |
| blocked union | 19243 |
| candidates after block | 146204 |
| candidates label0 | 71014 |
| candidates label1 | 75190 |
| maxlen-safe candidates after block | 146094 |
| maxlen-safe candidates base-error | 68411 |
| maxlen-safe candidates base-correct | 77683 |

对应文件：

- balanced ids/jsonl/split/vLLM split/summary 都在 `experiments/inputs/fever/qwen06b_round0_t1_alpha010_budget_sweep_1500_9000_seed1/seed1/test_sets/`，文件名前缀为 `balanced_test_10000_ml4096safe_exclude_guide_final_train12_seed20260521`。
- hard70err ids/jsonl/split/vLLM split/summary 都在同一目录，文件名前缀为 `hard70err_test_10000_ml4096safe_exclude_guide_final_train12_seed20260521`。

## 10. Vllm结果分析

当前表格是 balanced test 结果。这个测试集虽然标签平衡，但 base-error 只有 45.64%，明显低于 `ns-error-mass` 训练集的 65.60%-67.66% base-error 浓度，因此会稀释“错误样本选择”的收益。即便如此，1500/3000/4500/6000 四档 `ns-error-mass` 都高于 `pool-random`，macro-F1 分别提升 +0.0087、+0.0101、+0.0100、+0.0050。

hard70err test 还未评测。它把 round0 base-error 提到 70.00%，更接近当前方法真正优化的区域，后续应作为 stress-test 主表单独汇报。

| train n | method | test n | acc | macro-F1 | F1 no | F1 yes | balanced-acc | pred yes | TP/TN/FP/FN |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1500 | pool-random | 10000 | 0.9224 | 0.9224 | 0.9222 | 0.9226 | 0.9224 | 5022 | 4623/4601/399/377 |
| 1500 | ns-error-mass | 10000 | 0.9311 | 0.9311 | 0.9307 | 0.9315 | 0.9311 | 5061 | 4686/4625/375/314 |
| 3000 | pool-random | 10000 | 0.9250 | 0.9250 | 0.9259 | 0.9241 | 0.9250 | 4882 | 4566/4684/316/434 |
| 3000 | ns-error-mass | 10000 | 0.9351 | 0.9351 | 0.9349 | 0.9353 | 0.9351 | 5027 | 4689/4662/338/311 |
| 4500 | pool-random | 10000 | 0.9303 | 0.9303 | 0.9304 | 0.9302 | 0.9303 | 4979 | 4641/4662/338/359 |
| 4500 | ns-error-mass | 10000 | 0.9403 | 0.9403 | 0.9398 | 0.9408 | 0.9403 | 5091 | 4747/4656/344/253 |
| 6000 | pool-random | 10000 | 0.9397 | 0.9397 | 0.9396 | 0.9398 | 0.9397 | 5021 | 4709/4688/312/291 |
| 6000 | ns-error-mass | 10000 | 0.9447 | 0.9447 | 0.9446 | 0.9448 | 0.9447 | 5021 | 4734/4713/287/266 |
| 7500 | pool-random | 10000 |  |  |  |  |  |  |  |
| 7500 | ns-error-mass | 10000 |  |  |  |  |  |  |  |
| 9000 | pool-random | 10000 |  |  |  |  |  |  |  |
| 9000 | ns-error-mass | 10000 |  |  |  |  |  |  |  |

### 10.1 base-error 子集评测

用户要求的“全集 - 训练集，再按 base 原本对/错拆分”口径已生成。8 个已训练 run 的训练集并集为 11939 条；`all - train8` 后为 153508 条。由于 vLLM `max_model_len=4096` 且 `max_tokens=1`，删除 prompt tokens >4095 的 116 条长样本，最终可评测全集为 153392 条。

| split | n | label0 | label1 | 说明 |
| --- | ---: | ---: | ---: | --- |
| all-minus-train8-ml4096safe | 153392 | 73955 | 79437 | 全集减 8 个已训练 run 的训练集并集，再删 116 条超长样本 |
| base 原本对 | 80873 | 69510 | 11363 | round0 base `prediction == groundtruth` |
| base 原本错 | 72519 | 4445 | 68074 | round0 base `prediction != groundtruth` |

第一轮完整跑了 `pool_random_1500_seed1`，确认 base 原本对并不是“全对区”：该子集仍有 5608 个样本被 LoRA 预测错。因此 full/base-correct/base-error 三张表都可保留；若只快速看修错能力，base-error 子集最直接。

| subset | n | acc | macro-F1 | balanced-acc | pred yes | TP/TN/FP/FN |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| full | 153392 | 0.9233 | 0.9232 | 0.9232 | 79182 | 73424/68197/5758/6013 |
| base 原本对 | 80873 | 0.9307 | 0.8748 | 0.9323 | 15485 | 10620/64645/4865/743 |
| base 原本错 | 72519 | 0.9150 | 0.7443 | 0.8608 | 63697 | 62804/3552/893/5270 |

base-error 子集结果如下。这里 `macro-F1`/`balanced-acc` 比 raw accuracy 更重要，因为该子集的真实标签明显偏 label1。

| train n | method | n | acc | macro-F1 | balanced-acc | F1 yes | pred yes | TP/TN/FP/FN |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1500 | pool-random | 72519 | 0.9150 | 0.7443 | 0.8608 | 0.9532 | 63697 | 62804/3552/893/5270 |
| 1500 | ns-error-mass | 72519 | 0.9286 | 0.7718 | 0.8747 | 0.9610 | 64556 | 63726/3615/830/4348 |
| 3000 | pool-random | 72519 | 0.9107 | 0.7411 | 0.8713 | 0.9507 | 63143 | 62371/3673/772/5703 |
| 3000 | ns-error-mass | 72519 | 0.9265 | 0.7696 | 0.8800 | 0.9597 | 64279 | 63510/3676/769/4564 |
| 4500 | pool-random | 72519 | 0.9232 | 0.7614 | 0.8720 | 0.9579 | 64157 | 63329/3617/828/4745 |
| 4500 | ns-error-mass | 72519 | 0.9388 | 0.7961 | 0.8899 | 0.9667 | 65109 | 64372/3708/737/3702 |
| 6000 | pool-random | 72519 | 0.9337 | 0.7848 | 0.8860 | 0.9638 | 64759 | 64011/3697/748/4063 |
| 6000 | ns-error-mass | 72519 | 0.9395 | 0.7989 | 0.8942 | 0.9671 | 65090 | 64390/3745/700/3684 |

`ns-error-mass - pool-random` 差值：

| train n | Δ acc | Δ macro-F1 | Δ balanced-acc | Δ F1 yes | Δ TP | Δ TN | Δ FP | Δ FN |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1500 | +0.0136 | +0.0275 | +0.0139 | +0.0077 | +922 | +63 | -63 | -922 |
| 3000 | +0.0157 | +0.0286 | +0.0087 | +0.0091 | +1139 | +3 | -3 | -1139 |
| 4500 | +0.0156 | +0.0348 | +0.0179 | +0.0088 | +1043 | +91 | -91 | -1043 |
| 6000 | +0.0059 | +0.0141 | +0.0082 | +0.0033 | +379 | +48 | -48 | -379 |

![FEVER error-set budget diff](reports/fever_qwen06b_error_set_budget_diff.png)

对应汇总文件：

- `experiments/reports/fever_qwen06b_error_set_budget_diff_summary.json`
- `experiments/reports/fever_qwen06b_error_set_budget_metrics.csv`
- `experiments/reports/fever_qwen06b_error_set_budget_diffs.csv`
- `experiments/reports/fever_qwen06b_error_set_budget_diff.png`

## 11. 待填结论

当前已完成 1500/3000/4500/6000 四档共 8 个 LoRA run 的同口径 vLLM 评测，7500/9000 暂停未纳入本轮结果。

| train n | ns-error-mass macro-F1 - random macro-F1 |
| ---: | ---: |
| 1500 | +0.0087 |
| 3000 | +0.0101 |
| 4500 | +0.0100 |
| 6000 | +0.0050 |

在当前 10000 条 FEVER 测试集上，`ns-error-mass` 在四个已测预算档都高于 `pool-random`。提升在 1500-4500 档约 0.9-1.0 个百分点，6000 档缩小到约 0.5 个百分点。
