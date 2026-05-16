# 实验 4：消融实验

当前代码可直接跑 `delta`、band 比例、迭代轮次、LoRA rank、teacher beta、easy anchor。

## 数据前置

1. 使用同一份数据、embedding 和 teacher 文件。
2. 每个消融设置使用独立输出目录。
3. 每个设置至少跑到 round1：`prepare -> predict0 -> calibrate0 -> select0 -> train1 -> predict1 -> calibrate1`。
4. 除被消融的参数外，固定 `--seed`、`--temperature 15`、`--alpha 0.07`、`--budget 500`。
5. 如果比较训练样本数，注意 easy anchor 默认额外加入 10%；可以用 `--easy_anchor_ratio 0` 关闭。

## Baseline 要求和复用

消融的 baseline 是默认 DBDS 设置：`delta=0.1`、`teacher_beta=1`、`easy_anchor_ratio=0.1`、`lora_r=1`、固定预算和固定轮次。每个消融只改一个变量，结果表必须写清楚被改变量和 baseline 输出目录。

1. 可以复用实验 1 的 `cgsd_split_ids.json`、`round_0/pool_student_predictions.jsonl`、`round_0/pool_crc_predictions.jsonl` 和 `round_0/round_summary.json`，避免每个消融重复 round0 推理。
2. `delta`、`teacher_beta`、easy anchor 消融必须重新跑 selection，因为训练行会变；之后重新 train/predict/calibrate。
3. LoRA rank 消融可以复用同一份 baseline `cgsd_train_rows.jsonl`，只重新 train/predict/calibrate，保证训练数据不变。
4. 迭代轮次消融只能复用 round0；round1 之后的预测、CRC、selection 都要按该轮次设置重新生成。
5. Teacher 加权消融需要 teacher 文件含 `teacher_confidence` 或 `teacher_logit_margin`；如果只用 groundtruth 替代，这组消融只能作为无效/对照记录，不应解释为 teacher weighting 的效果。

## delta 消融

用 `cgsd_select.py --delta` 控制 band 宽度：

```bash
python scripts/cgsd_select.py \
  --output_dir "$OUT" \
  --round_index 0 \
  --embeddings_path "$EMB" \
  --budget 500 \
  --delta 0.05
```

重复 `0.05, 0.1, 0.15, 0.2`。

## Band 比例消融

用 `BAND_RATIOS` 控制 DBDS 的 B/M/F 比例：

```bash
export DATASET=lrobench
export RUN_NAME=exp4_band_all_boundary_seed1
export DIM=2560
BUDGET=500 BAND_RATIOS=1,0,0 experiments/bin/cgsd_round0_select.sh
ROUND=1 experiments/bin/cgsd_train_round.sh
ROUND=1 experiments/bin/cgsd_eval_round.sh
```

重复：

```text
1,0,0
0.6,0.3,0.1
0.33,0.34,0.33
0,0,1
0,1,0
```

## 迭代轮次消融

用预算分配控制轮次：

1. `Tmax=1`：只跑 `budget=500` 到 round1。
2. `Tmax=2`：跑两次 select/train，预算 `250,250`。
3. `Tmax=3`：跑三次 select/train，预算 `167,167,166`。
4. `Tmax=5`：跑五次 select/train，预算 `100,100,100,100,100`。

每轮都执行 `predict -> calibrate -> select -> train`，最后用 `cgsd_finalize.py --round_index <最终轮>`。

## LoRA rank 消融

在训练 stage 改 `--lora_r`：

```bash
python scripts/cgsd_train_round.py \
  --output_dir "$OUT" \
  --round_index 1 \
  --model_path "$MODEL" \
  --data_path "$DATA" \
  --train_rows_path "$OUT/cgsd_train_rows.jsonl" \
  --lora_r 4
```

重复 `1, 2, 4, 8`。

## Teacher 加权消融

在 selection stage 改 `--teacher_beta`：

```bash
python scripts/cgsd_select.py \
  --output_dir "$OUT" \
  --round_index 0 \
  --embeddings_path "$EMB" \
  --budget 500 \
  --teacher_beta 0.5
```

重复 `0, 0.5, 1, 2`。

输入要求：teacher 文件最好包含 `teacher_confidence` 或 `teacher_logit_margin`；如果只用 groundtruth 替代，所有 teacher confidence 都是 `1.0`，这个消融不会产生实际差异。

## Easy anchor 消融

在 selection stage 改 `--easy_anchor_ratio`，或用 `--anchor_count` 固定条数：

```bash
python scripts/cgsd_select.py \
  --output_dir "$OUT" \
  --round_index 0 \
  --embeddings_path "$EMB" \
  --budget 500 \
  --easy_anchor_ratio 0.2
```

重复 `0, 0.05, 0.1, 0.2`。

## 聚合

```bash
experiments/bin/cgsd_collect_results.py \
  --runs 'experiments/runs/lrobench/exp4_*_seed1' \
  --output_csv experiments/runs/lrobench/exp4_ablation_seed1.csv
```
