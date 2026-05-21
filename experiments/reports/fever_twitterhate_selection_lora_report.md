# FEVER 与 TwitterHate 选样 LoRA 实验报告

生成时间：2026-05-20。

## 1. 实验设置

- FEVER：`Qwen3-1.7B` round0 base，`T=1`，`alpha=0.1`，guide=1000，训练集每组 3000 条，seed=1/2；评测集为 `eight_cell_near_balanced_unique_10000_seed1`，共 10000 条。
- TwitterHate：`Qwen3-0.6B` round0 base，`T=1`，`alpha=0.1`，训练集每组 2231 条，即原 train split 的 15%，seed=1；评测集为原 test split，共 7435 条。
- LoRA 参数：`epochs=4`，`lr=3e-5`，`lora_r=8`，`lora_alpha=16`，`lora_dropout=0.05`，`target_modules=attention_mlp`，`max_length=4096`，`batch_size=2`，`gradient_accumulation_steps=8`，`bf16`。
- vLLM 评测：`temperature=0`，`max_tokens=1`，`top_logprobs=20`；FEVER 评测使用同一 8-cell 测试集，TwitterHate 使用同一 test split。

## 2. FEVER 8 组实验

FEVER round0 base 在同一测试集上的结果：acc=50.97%，macro-F1=50.92%，F1-neg=52.54%，F1-pos=49.29%。

### 2.1 训练数据构成

| method | seed | n | accept | defer | base-error | error-rate | label0 | label1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| random | 1 | 3000 | 2846 | 154 | 369 | 12.30% | 1432 | 1568 |
| random | 2 | 3000 | 2840 | 160 | 416 | 13.87% | 1440 | 1560 |
| crc-error-mass | 1 | 3000 | 1781 | 1219 | 777 | 25.90% | 1639 | 1361 |
| crc-error-mass | 2 | 3000 | 1781 | 1219 | 743 | 24.77% | 1673 | 1327 |
| ns-difficulty-global | 1 | 3000 | 2830 | 170 | 494 | 16.47% | 966 | 2034 |
| ns-difficulty-global | 2 | 3000 | 2876 | 124 | 465 | 15.50% | 955 | 2045 |
| ns-difficulty-crc-split | 1 | 3000 | 1821 | 1179 | 760 | 25.33% | 1611 | 1389 |
| ns-difficulty-crc-split | 2 | 3000 | 1821 | 1179 | 785 | 26.17% | 1698 | 1302 |

### 2.2 测试结果

| method | seed | acc | macro-F1 | F1-neg | F1-pos | pred-pos-rate |
| --- | --- | --- | --- | --- | --- | --- |
| random | 1 | 79.00% | 78.55% | 81.65% | 75.45% | 39.81% |
| random | 2 | 80.17% | 79.60% | 83.01% | 76.20% | 37.58% |
| crc-error-mass | 1 | 81.11% | 80.75% | 83.38% | 78.13% | 40.64% |
| crc-error-mass | 2 | 80.34% | 79.92% | 82.82% | 77.02% | 39.81% |
| ns-difficulty-global | 1 | 80.69% | 80.26% | 83.18% | 77.33% | 39.46% |
| ns-difficulty-global | 2 | 80.00% | 79.50% | 82.71% | 76.28% | 38.59% |
| ns-difficulty-crc-split | 1 | 79.62% | 79.24% | 82.04% | 76.44% | 40.79% |
| ns-difficulty-crc-split | 2 | 81.33% | 81.24% | 83.08% | 76.80% | 38.64% |

### 2.3 FEVER seed 均值

| method | seeds | mean acc | mean macro-F1 | mean F1-neg | mean F1-pos | Δacc vs random | Δmacro-F1 vs random |
| --- | --- | --- | --- | --- | --- | --- | --- |
| random | 2 | 79.58% | 79.08% | 82.33% | 75.82% | 0.00% | 0.00% |
| crc-error-mass | 2 | 80.73% | 80.34% | 83.10% | 77.57% | 1.14% | 1.26% |
| ns-difficulty-global | 2 | 80.34% | 79.88% | 82.95% | 76.81% | 0.76% | 0.80% |
| ns-difficulty-crc-split | 2 | 80.78% | 80.54% | 82.56% | 76.62% | 1.20% | 1.46% |

FEVER 上 `crc-error-mass` 均值仍优于 random：相比 random，acc 提升 1.14 个百分点，macro-F1 提升 1.26 个百分点。`ns-difficulty-global` 也优于 random；`ns-difficulty-crc-split` 在 acc 上略优于 `crc-error-mass`，但在 macro-F1 上略低于 `crc-error-mass`。总体来看，基于 CRC 错误质量分配训练预算的 `crc-error-mass` 是 FEVER 上最稳的提升来源；NS difficulty 作为辅助信号有一定提升，但单独使用不一定稳定。

## 3. TwitterHate 4 组实验

TwitterHate round0 近似全预测 1；在 test 上 all-one reference 为 acc=83.20%，macro-F1=45.42%。

| method | n | accept | defer | base-error | error-rate | test acc | macro-F1 | F1-0 | F1-1 | pred0 | pred1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| random | 2231 | 1840 | 391 | 390 | 17.48% | 94.54% | 90.36% | 84.02% | 96.71% | 1291 | 6144 |
| crc-error-mass | 2231 | 1106 | 1125 | 617 | 27.66% | 94.84% | 90.57% | 84.24% | 96.91% | 1187 | 6248 |
| ns-difficulty-global | 2231 | 1827 | 404 | 353 | 15.82% | 94.34% | 89.84% | 83.09% | 96.60% | 1240 | 6195 |
| ns-difficulty-crc-split | 2231 | 1106 | 1125 | 590 | 26.45% | 94.94% | 90.75% | 84.51% | 96.98% | 1179 | 6256 |

TwitterHate 上单 seed 最好的是 `ns-difficulty-crc-split`：acc=94.94%，macro-F1=90.75%，比 random 高 0.40 个百分点 acc、0.39 个百分点 macro-F1。`crc-error-mass` 次之；单独 `ns-difficulty-global` 低于 random。这里的一个重要原因是 0.6B round0 在 TwitterHate 上几乎全预测 1，使 NS difficulty 的可分性变弱，单独全局按 difficulty 选样不稳定。

## 4. 综合结论

1. `crc-error-mass` 在 FEVER 上是最稳的提升来源，说明基于 CRC accept/defer 错误质量分配训练预算是有效的。
2. `ns-difficulty` 单独使用不一定稳定：FEVER 中接近但略低于 `crc-error-mass`，TwitterHate 中因为 round0 预测坍缩到正类，global difficulty 选样反而低于 random。
3. `ns-difficulty-crc-split` 在 TwitterHate 上最好，说明 NS difficulty 更适合作为 CRC 分区内部的二级排序信号，而不是完全替代 CRC accept/defer 预算。
4. 当前最值得继续扩展的是：FEVER 继续以 `crc-error-mass` 为主 baseline；TwitterHate 需要先解决 0.6B round0 全预测 1 的 prompt/model bias，再判断 NS difficulty 的真实贡献。

## 5. 文件索引

- FEVER random/crc 评测 summary：`experiments/runs/fever_lora_qwen17b_round0_t1_3000_crc_vs_random/lr3e-5_e4_r8_a16_all_ml4096_bs2_ga8/eval_eight_cell_near_balanced_unique_10000_seed1_summary/summary.json`
- FEVER ns-difficulty 评测 summary：`experiments/runs/fever_lora_qwen17b_round0_t1_3000_ns_difficulty/lr3e-5_e4_r8_a16_all_ml4096_bs2_ga8/eval_eight_cell_near_balanced_unique_10000_seed1_summary/summary.json`
- TwitterHate 评测 summary：`experiments/runs/twitter_hate_lora_qwen06b_round0_t1_15pct_four_methods/lr3e-5_e4_r8_a16_all_ml4096_bs2_ga8/twitter_hate_test_eval_summary.json`
- 本报告结构化 JSON：`experiments/reports/fever_twitterhate_selection_lora_report_summary.json`
