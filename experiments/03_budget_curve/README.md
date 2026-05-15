# 实验 3：标注预算曲线

当前代码可以跑 DBDS 的预算扫描；Random 和 Uncertainty 曲线需要按实验 2 的方式手工生成 baseline 训练行。

## 数据前置

1. 准备同一个数据文件、embedding 文件和可选 teacher 文件。
2. 每个预算 `m` 使用独立输出目录，例如 `outputs/cgsd_exp3_m050_seed1`。
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
python scripts/cgsd_prepare.py --data_path "$DATA" --embeddings_path "$EMB" --output_dir "$OUT" --n_calibration 200 --seed 1
python scripts/cgsd_predict.py --output_dir "$OUT" --round_index 0 --model_path "$MODEL" --data_path "$DATA" --teacher_labels_path "$TEACHER"
python scripts/cgsd_calibrate.py --output_dir "$OUT" --round_index 0 --temperature 15 --alpha 0.07
python scripts/cgsd_finalize.py --output_dir "$OUT" --round_index 0
```

## m>0 的运行方式

1. 先跑 `prepare -> predict round0 -> calibrate round0`。
2. 用当前预算运行 selection：

```bash
python scripts/cgsd_select.py \
  --output_dir "$OUT" \
  --round_index 0 \
  --embeddings_path "$EMB" \
  --budget "$M"
```

3. 训练 round1 并重校准：

```bash
python scripts/cgsd_train_round.py --output_dir "$OUT" --round_index 1 --model_path "$MODEL" --data_path "$DATA"
python scripts/cgsd_predict.py --output_dir "$OUT" --round_index 1 --model_path "$MODEL" --data_path "$DATA" --checkpoint_dir "$OUT/round_1/model" --teacher_labels_path "$TEACHER"
python scripts/cgsd_calibrate.py --output_dir "$OUT" --round_index 1 --temperature 15 --alpha 0.07
python scripts/cgsd_finalize.py --output_dir "$OUT" --round_index 1
```

## 需要记录

1. `round_1/round_summary.json` 的 `pool_summary.defer_rate` 和 `pool_summary.accept_error_rate`。
2. `cgsd_summary.json` 的 teacher 调用摘要。
3. 成本曲线需要额外聚合脚本：`C(m) = (m + n_cal + rho * N) * c_T + N * c_S`。
4. 严格成本以 `predict_usage.json`、`select_usage.json`、`train_usage.json` 和 `finalize_usage.json` 为准；公式里的 `m` 要加上 easy anchor，实验总成本还要加每轮全量 student 推理。
5. 每个预算点都要保留 call/token 账本，后续统一聚合：
   - `round_*/predict_usage.json`：student/base 或 LoRA 推理行数、估算 prompt/completion token。
   - `round_*/calibrate_usage.json`：CRC 校准行数、校准标签来源、估算输入 token。
   - `round_*/select_usage.json`：DBDS 选中行数、embedding 行数/维度、teacher/groundtruth 标签调用数、选中样本估算 token。
   - `round_*/train_usage.json`：训练样本数、训练 step 估算、训练 token 估算、训练标签来源。
   - `finalize_usage.json`：最终 deployment defer 的 teacher/groundtruth 调用数和 defer prompt token。
