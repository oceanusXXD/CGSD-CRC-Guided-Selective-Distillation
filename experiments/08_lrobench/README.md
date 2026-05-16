# LROBench 实验

当前代码可以跑 per-query 独立 CGSD；cross-query transfer 和 LoRA 参数平均初始化没有实现。

## 数据前置

1. 当前已有合并版入口：`experiments/inputs/lrobench/data.jsonl` 和 `experiments/inputs/lrobench/embeddings.npy`，可作为 pooled multi-query smoke test。
2. 严格 LROBench 结果仍建议拆成 per-query JSONL：每行至少包含 `id/query/document/groundtruth`，可额外保留 `query_id/document_id/review_id/parsed_answer/parsed_confidence`。
3. 每个 query 准备独立 embedding 文件，覆盖该 query 下所有 row；放到 `experiments/inputs/lrobench_<query>/embeddings.npy`。
4. 每个 query 使用独立 `RUN_NAME`，例如 `q01_seed1`，输出在 `experiments/runs/lrobench_<query>/q01_seed1/`。
5. `id` 建议包含 query 编号和 row 编号，例如 `q01_row003`，避免跨 query 合并时冲突。
6. `document` 可以是原始 row 文本，也可以是稳定序列化后的字段串；同一实验里必须固定格式。
7. 如果每个 query 只有几十条样本，先确认 `n_calibration + budget + anchor_count` 小于该 query 的样本数。

## Baseline 要求和复用

LROBench 的 baseline 以 per-query 为单位记录，不能把不同 query 的 split、embedding 或校准结果混用。

1. Per-query zero-shot CRC：每个 query 先保留 `round_0/pool_student_predictions.jsonl`、`round_0/pool_crc_predictions.jsonl` 和 `round_0/round_summary.json`，后续 alpha 调整或小样本可行性检查可以复用这些预测。
2. Per-query CGSD：同一 query、同一 seed、同一 `n_calibration/budget/anchor_count` 下，可以复用 split 和 round0 缓存；selection、train、round1 predict/calibrate 仍要按该 query 单独输出。
3. Random 或 Uncertainty 小样本 baseline 如果要加，需要按实验 2 的训练行格式为每个 query 单独生成 `$OUT/cgsd_train_rows.jsonl`，不能复用其他 query 的训练行。
4. Cross-query transfer 和 LoRA 参数平均当前没有实现；如果需要作为 baseline，只能使用外部结果文件或另行实现后再记录，不应写成当前代码可直接跑。
5. 每个 query 都要单独记录 `crc.grid_feasible`、`crc.risk_bound`、`lambda_hat`、defer rate、teacher/groundtruth call 和 token，方便后续按 query 汇总均值和方差。

## Per-query 独立训练

先把合并版数据拆成 per-query 输入目录：

```bash
experiments/bin/cgsd_split_lrobench_inputs.py \
  --data_path experiments/inputs/lrobench/data.jsonl \
  --embeddings_path experiments/inputs/lrobench/embeddings.npy \
  --output_root experiments/inputs \
  --prefix lrobench
```

然后对每个 query 单独运行：

```bash
export DATASET=lrobench_q01
export RUN_NAME=q01_seed1
export DIM=2560
export N_CALIBRATION=10

BUDGET=10 ANCHOR_COUNT=0 experiments/bin/cgsd_round0_select.sh
ROUND=1 experiments/bin/cgsd_train_round.sh
ROUND=1 experiments/bin/cgsd_eval_round.sh
```

## 小样本注意事项

1. LROBench 每个 query 样本少，`--n_calibration` 需要小于该 query 样本数。
2. `--budget` 也要按 query 样本数缩小，避免 selection 没有足够候选。
3. 小校准集下 CRC 修正项很大，defer 率可能偏高，这是实验本身要观察的点。
4. 小样本下建议显式设置 `--anchor_count 0` 或很小的值，避免 easy anchor 占掉过多训练标签。
5. 如果只用 groundtruth 替代 teacher，usage 里的调用会计入 `groundtruth_substitute_calls`，不是 `teacher_api_file_calls`。
6. 使用 `--n_calibration 10` 时有限样本修正下界约为 `1 / (10 + 1) = 0.091`；如果 `--alpha 0.07` 导致可行阈值只剩全 defer，需要在该 query 的实验记录里写明 `crc.grid_feasible`、`crc.risk_bound`、`lambda_hat` 和最终 defer 率，并注明是否改用更大的校准集或更宽松的 alpha。

## 当前不能直接跑的模式

Cross-query transfer 需要把前 15 个 query 的 LoRA adapter 合并或作为初始化；当前 `cgsd_train_round.py` 始终从 base model 训练，没有提供 adapter 平均或跨 query 初始化参数。
