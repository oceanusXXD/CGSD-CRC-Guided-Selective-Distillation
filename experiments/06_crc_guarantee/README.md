# 实验 6：CRC 保证验证

当前代码可以跑多 seed、多 alpha 的 CRC 验证，并可用 `experiments/bin/cgsd_collect_results.py` 汇总均值和违反率。

## 数据前置

1. 准备同一数据和 embedding。
2. 对每个 seed 使用独立输出目录。
3. 如果验证最终 CGSD 后的保证，先按实验 1 跑到最终 round；如果只验证 zero-shot CRC，只跑 round0。
4. 严格保证实验不要复用参与 selection、temperature 搜索或 early stop 的同一份 calibration 标签；如果只跑当前默认 pipeline，请把结果表述成“经验验证”而不是 theorem-level guarantee。

## Baseline 要求和复用

本实验至少区分 zero-shot CRC baseline 和最终 CGSD CRC 两类结果。

1. Zero-shot CRC：可以复用实验 1 同 seed 的 `cgsd_split_ids.json`、`round_0/calibration_predictions.jsonl`、`round_0/pool_student_predictions.jsonl` 和 `round_0/pool_crc_predictions.jsonl`；做 alpha sweep 时只需要对同一份预测重新跑 `cgsd_calibrate.py --alpha <值>`。
2. 最终 CGSD CRC：如果只是经验验证，可以复用实验 1 最终 round 的模型预测和 usage；如果要 theorem-level final guarantee，需要额外准备独立最终校准集并生成独立 `calibration_predictions_path`。
3. 多 seed 验证可以复用各 seed 在实验 1 已经生成的 split 和 round0 预测；缺哪个 seed 就只补哪个 seed，不需要重跑已有缓存。
4. 多 alpha 验证不要重复 student 推理；固定同一份预测文件后只重跑 calibrate，并把 alpha、`crc.empirical_risk`、`crc.risk_bound` 和 defer rate 记到汇总表。

## 单个 seed、单个 alpha

```bash
python scripts/cgsd_prepare.py \
  --data_path "$DATA" \
  --embeddings_path "$EMB" \
  --output_dir "$OUT" \
  --n_calibration 200 \
  --seed "$SEED"

python scripts/cgsd_predict.py \
  --output_dir "$OUT" \
  --round_index 0 \
  --model_path "$MODEL" \
  --data_path "$DATA" \
  --teacher_labels_path "$TEACHER"

python scripts/cgsd_calibrate.py \
  --output_dir "$OUT" \
  --round_index 0 \
  --temperature 15 \
  --alpha "$ALPHA"
```

## 多 seed 执行

对 `SEED=1..20` 重复上面的命令，并把输出目录设为：

```text
experiments/runs/<dataset>/exp6_alpha007_seed01
experiments/runs/<dataset>/exp6_alpha007_seed02
...
```

只跑 zero-shot CRC 时可用：

```bash
SEED=1 RUN_NAME=exp6_alpha007_seed01 ALPHA=0.07 experiments/bin/cgsd_round0_eval.sh
SEED=2 RUN_NAME=exp6_alpha007_seed02 ALPHA=0.07 experiments/bin/cgsd_round0_eval.sh
```

## 多 alpha 执行

重复 alpha：

```text
0.03, 0.05, 0.07, 0.09, 0.12
```

## 汇总

```bash
experiments/bin/cgsd_collect_results.py \
  --runs 'experiments/runs/lrobench/exp6_alpha*_seed*' \
  --output_csv experiments/runs/lrobench/exp6_crc_runs.csv \
  --crc_summary_csv experiments/runs/lrobench/exp6_crc_summary.csv
```

## 需要记录

1. `round_*/round_summary.json` 的 `pool_summary.accept_error_rate`。
2. `round_*/round_summary.json` 的 `pool_summary.defer_rate`。
3. 经验违反率：`pool_summary.wrong_accept_count / pool_summary.total > alpha` 的次数除以运行次数。
4. 证明口径还要记录 `crc.empirical_risk` 和 `crc.risk_bound`；`pool_summary.accept_error_rate` 是 accept 子集条件错误率，不应直接拿来和 `alpha` 比。
