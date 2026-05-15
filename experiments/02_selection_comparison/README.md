# 实验 2：数据选择策略对比

当前代码可以直接跑 DBDS；Random、Uncertainty、k-Center、Defer-Random 还没有独立 selection CLI，需要先手工生成对应的 `cgsd_train_rows.jsonl`，再复用 `cgsd_train_round.py`、`cgsd_predict.py`、`cgsd_calibrate.py`。

## 数据前置

1. 先完成一次 round0：`prepare -> predict round0 -> calibrate round0`。
2. 确保有 `round_0/pool_student_predictions.jsonl` 和 `round_0/pool_crc_predictions.jsonl`。
3. 确保有覆盖全量 pool 的 embedding 文件。
4. 所有策略必须使用同一个 `cgsd_split_ids.json`、同一份 teacher/groundtruth 标签和同一个 seed。
5. 每种策略使用单独输出目录，例如 `outputs/cgsd_exp2_dbds_seed1`、`outputs/cgsd_exp2_random_seed1`；复制或复用同一份 round0 artifact 时要显式传 `--*_path`，不要混写到同一个目录。

## Baseline 要求和复用

可以复用实验 1 的 round0 缓存，前提是 `DATA`、`EMB`、`TEACHER` 或 groundtruth 口径、`MODEL`、`seed`、`n_calibration`、`temperature` 和 `alpha` 完全一致。复用时在每个 baseline 的输出目录里显式传入实验 1 的 `cgsd_split_ids.json`、`round_0/pool_student_predictions.jsonl`、`round_0/pool_crc_predictions.jsonl` 和 `round_0/round_summary.json`。

1. DBDS：可以直接用 `cgsd_select.py` 从复用的 round0 defer/score 结果选样。
2. Random：从同一个 pool 随机抽样，不允许包含 calibration ID；需要写出独立的 `$OUT/cgsd_train_rows.jsonl`。
3. Uncertainty：按同一份 round0 `routing_score` 升序选样；需要写出独立的 `$OUT/cgsd_train_rows.jsonl`。
4. k-Center：用同一份 embedding 和同一个 pool 做 k-Center；需要写出独立的 `$OUT/cgsd_train_rows.jsonl`。
5. Defer-Random：只从同一份 `round_0/pool_crc_predictions.jsonl` 里 `defer=true` 的样本随机抽样；需要写出独立的 `$OUT/cgsd_train_rows.jsonl`。
6. 所有 baseline 训练后都必须重新跑 `train_round -> predict -> calibrate`；不能复用 DBDS 或实验 1 的 LoRA 模型结果。

## Baseline 训练行格式

其他策略手工生成的 `$OUT/cgsd_train_rows.jsonl` 必须是 JSONL，每行至少包含：

```json
{"id":"sample_001","query":"...","document":"...","label":1,"groundtruth":1,"teacher_label":1,"teacher_confidence":1.0,"teacher_source":"groundtruth_substitute_for_real_teacher_api","sample_weight":1.0,"selection_round":0,"selection_role":"baseline_random"}
```

要求：

1. 不要包含 calibration ID。
2. `label/groundtruth/teacher_label` 必须是同一 oracle 口径。
3. 如果使用真实 teacher，把 `teacher_source` 设为 `teacher_api_file`，并保留 teacher 置信度。
4. `sample_weight` 可统一写 `1.0`；如果复用 teacher 加权，按 `(teacher_confidence ** beta)` 写入。

## DBDS 运行方式

1. 选择 DBDS 训练样本：

```bash
python scripts/cgsd_select.py \
  --output_dir "$OUT" \
  --round_index 0 \
  --split_ids_path "$OUT/cgsd_split_ids.json" \
  --pool_crc_predictions_path "$OUT/round_0/pool_crc_predictions.jsonl" \
  --pool_student_predictions_path "$OUT/round_0/pool_student_predictions.jsonl" \
  --round_summary_path "$OUT/round_0/round_summary.json" \
  --embeddings_path "$EMB" \
  --budget 500
```

2. 训练并重新评估：

```bash
python scripts/cgsd_train_round.py \
  --output_dir "$OUT" \
  --round_index 1 \
  --model_path "$MODEL" \
  --data_path "$DATA" \
  --split_ids_path "$OUT/cgsd_split_ids.json" \
  --train_rows_path "$OUT/cgsd_train_rows.jsonl"

python scripts/cgsd_predict.py \
  --output_dir "$OUT" \
  --round_index 1 \
  --model_path "$MODEL" \
  --data_path "$DATA" \
  --split_ids_path "$OUT/cgsd_split_ids.json" \
  --checkpoint_dir "$OUT/round_1/model" \
  --teacher_labels_path "$TEACHER"

python scripts/cgsd_calibrate.py \
  --output_dir "$OUT" \
  --round_index 1 \
  --temperature 15 \
  --alpha 0.07
```

## 其他策略的处理方式

1. Random：从 `round_0/pool_student_predictions.jsonl` 随机抽 500 行，写成 `$OUT/cgsd_train_rows.jsonl`。
2. Uncertainty：按 `routing_score` 升序取 500 行，写成 `$OUT/cgsd_train_rows.jsonl`。
3. k-Center：从全 pool 用 embedding 做 k-Center 选 500 行，写成 `$OUT/cgsd_train_rows.jsonl`。
4. Defer-Random：从 `round_0/pool_crc_predictions.jsonl` 中 `defer=true` 的样本随机抽 500 行，写成 `$OUT/cgsd_train_rows.jsonl`。
5. 写好训练行后，统一执行上面的 `train_round -> predict -> calibrate`。

## 需要记录

1. 每个策略 round1 的 `round_1/round_summary.json`。
2. 每个策略的 `round_1/train_usage.json` 和 `round_1/calibrate_usage.json`。
3. DBDS 与每个 baseline 的 `pool_summary.defer_rate` 差异；p-value 需要额外统计脚本。
4. DBDS 默认会额外选择 easy anchor；若要和 baseline 保持严格相同训练样本数，给 DBDS 加 `--easy_anchor_ratio 0`，或给 baseline 也额外加入同等数量的 anchor。
