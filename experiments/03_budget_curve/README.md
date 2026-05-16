# 实验 3：标注预算曲线

当前代码可以跑 DBDS 的预算扫描；Random 和 Uncertainty 曲线需要按实验 2 的方式手工生成 baseline 训练行。

## 数据前置

1. 准备同一个数据文件、embedding 文件和可选 teacher 文件。
2. 每个预算 `m` 使用独立 `RUN_NAME`，例如 `exp3_m050_seed1`，输出在 `experiments/runs/<dataset>/<run_name>/`。
3. `m=0` 不运行 selection 和 train，只记录 zero-shot 的 CRC 结果。
4. 每个预算点使用相同 `--n_calibration`、`--seed`、`--temperature 15` 和 `--alpha`，否则曲线不可比。
5. 这里的 `m` 是 DBDS defer 样本预算；默认 easy anchor 会额外加入 `floor(0.1 * m)` 个训练样本。若横轴必须表示总 teacher 训练标签数，运行时加 `--easy_anchor_ratio 0`。

## Baseline 要求和复用

预算曲线默认只画 DBDS；如果要叠加 Random 或 Uncertainty 曲线，每个 `m` 都要按实验 2 的训练行格式单独生成 baseline 的 `$OUT/cgsd_train_rows.jsonl`，并使用同一个 split、teacher/groundtruth 口径、seed、temperature 和 alpha。

1. `m=0` 可以复用实验 1 的 `round_0/pool_student_predictions.jsonl`、`round_0/pool_crc_predictions.jsonl` 和 `round_0/round_summary.json`；如果当前实验目录需要独立 `cgsd_summary.json`，只补跑 `cgsd_finalize.py --round_index 0`。
2. `m>0` 可以复用实验 1 的 `cgsd_split_ids.json` 和 round0 预测/CRC 缓存，但每个预算点必须重新跑 selection、train、round1 predict/calibrate/finalize。
3. 如果某个 `m=500` 的 DBDS 设置和实验 2 完全一致，可以复用实验 2 的训练行或结果；否则只复用 round0 缓存。
4. Random/Uncertainty baseline 曲线不能复用 DBDS 的 `cgsd_train_rows.jsonl`，只能复用 split、round0 预测和 teacher/groundtruth 标签。

## 预算列表

按实验文档运行：

```text
0, 50, 100, 200, 300, 500, 700, 1000, 2000
```

## m=0 的运行方式

```bash
export DATASET=lrobench
export RUN_NAME=exp3_m000_seed1
export DIM=2560
experiments/bin/cgsd_round0_eval.sh
ROUND=0 experiments/bin/cgsd_finalize.sh
```

## m>0 的运行方式

1. 先跑 `prepare -> predict round0 -> calibrate round0`。
2. 用当前预算运行 selection：

```bash
RUN_NAME=exp3_m050_seed1 experiments/bin/cgsd_round0_eval.sh
ROUND=0 BUDGET=50 experiments/bin/cgsd_select_round.sh
```

3. 训练 round1 并重校准：

```bash
ROUND=1 experiments/bin/cgsd_train_round.sh
ROUND=1 experiments/bin/cgsd_eval_round.sh
ROUND=1 experiments/bin/cgsd_finalize.sh
```

## 需要记录

1. `round_1/round_summary.json` 的 `pool_summary.defer_rate` 和 `pool_summary.accept_error_rate`。
2. `cgsd_summary.json` 的 teacher 调用摘要。
3. 成本曲线可用 `experiments/bin/cgsd_collect_results.py` 汇总 CSV，再用外部绘图工具画图。
4. 严格成本以 `predict_usage.json`、`select_usage.json`、`train_usage.json` 和 `finalize_usage.json` 为准；公式里的 `m` 要加上 easy anchor，实验总成本还要加每轮全量 student 推理。
5. 每个预算点都要保留 call/token 账本，后续统一聚合：
   - `round_*/predict_usage.json`：student/base 或 LoRA 推理行数、估算 prompt/completion token。
   - `round_*/calibrate_usage.json`：CRC 校准行数、校准标签来源、估算输入 token。
   - `round_*/select_usage.json`：DBDS 选中行数、embedding 行数/维度、teacher/groundtruth 标签调用数、选中样本估算 token。
   - `round_*/train_usage.json`：训练样本数、训练 step 估算、训练 token 估算、训练标签来源。
   - `finalize_usage.json`：最终 deployment defer 的 teacher/groundtruth 调用数和 defer prompt token。

聚合示例：

```bash
experiments/bin/cgsd_collect_results.py \
  --runs 'experiments/runs/lrobench/exp3_m*_seed1' \
  --output_csv experiments/runs/lrobench/exp3_budget_curve_seed1.csv
```
