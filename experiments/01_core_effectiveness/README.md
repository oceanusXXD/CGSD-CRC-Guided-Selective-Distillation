# 实验 1：CGSD 核心有效性

当前代码可以完成这个实验的主流程：多轮 `predict -> calibrate -> select -> train`，并在每轮输出 defer rate、CRC 阈值、诊断用 accept error、teacher 调用和 usage 账本。

## 数据前置

1. 使用 `experiments/inputs/<dataset>/data.jsonl` 和 `experiments/inputs/<dataset>/embeddings.npy` 作为实验入口；输出统一写到 `experiments/runs/<dataset>/<run_name>/`。
2. 当前 `lrobench` 已有数据和 2560 维 embedding，可以直接跑。`fever` 已有数据，但还缺 `experiments/inputs/fever/embeddings.npy`。
3. 每行至少有 `id/query/document/groundtruth`；FEVER 的 `query` 是固定任务句，`document` 包含 Claim 和 Evidence。
4. 可选准备真实 teacher 文件：至少包含 `id` 和 `teacher_label`，可选 `teacher_confidence` 或 `teacher_logit_margin`；不准备时使用 `groundtruth` 离线替代。
5. 为 5 个 seed 分别设置独立 `RUN_NAME`，例如 `exp1_seed1`。
6. 严格实验固定温度 `--temperature 15`，不要用 round0 温度扫描作为正式结果。

## 输入检查

1. `python scripts/cgsd_prepare.py ...` 成功后检查 `$OUT/prepare_usage.json`，确认 `data_rows = calibration_size + pool_size`。
2. 检查 `$OUT/cgsd_split_ids.json`，确认 calibration 和 pool 没有重复 ID。
3. 如果使用真实 teacher，检查 `$OUT/round_0/predict_usage.json` 中 `teacher_api_file_calls` 是否等于 `student_model_calls`；否则说明有样本回落到了 groundtruth 替代。
4. `--budget 250/150/100` 是 DBDS defer 样本数；默认还会额外加入 10% easy anchor 并消耗 teacher/groundtruth 标签。
5. 每个 round 都要保留 `pool_student_predictions.jsonl` 和 `pool_crc_predictions.jsonl`，后面用来检查上一轮 accept 样本在 LoRA 后是否退化。

## 可复用缓存

实验 1 可以作为后续实验的主缓存。只有当 `DATA`、`EMB`、`TEACHER` 或 groundtruth 口径、`MODEL`、`seed`、`n_calibration`、`temperature` 和 `alpha` 都一致时，才复用同一个 `$OUT` 下的 artifact。

1. `$OUT/cgsd_split_ids.json`：后续实验 2/3/4/6 可以复用，确保 calibration/pool 划分一致。
2. `$OUT/round_0/pool_student_predictions.jsonl`、`$OUT/round_0/pool_crc_predictions.jsonl`、`$OUT/round_0/round_summary.json`：后续实验 2 的 baseline 选样、实验 3 的 `m=0` 和所有从 round0 开始的预算/消融实验可以复用。
3. `$OUT/round_*/pool_student_predictions.jsonl` 和 `$OUT/round_*/pool_crc_predictions.jsonl`：后续只用于 accept/defer 退化校验；不要用上一轮 defer 子集替代正式全量预测。
4. `$OUT/cgsd_train_rows.jsonl`：只在预算、anchor、`teacher_beta`、selection round 完全一致时复用；Random、Uncertainty、k-Center、Defer-Random baseline 必须各自生成训练行。
5. `$OUT/cgsd_summary.json`、`$OUT/deployment_decisions.jsonl` 和各 stage usage JSON：如果实验 5 采用同一套 CGSD 配置，可以直接作为 CGSD 端到端结果；否则只作为参考缓存，不要混入结果表。

## 轻量脚本运行步骤

先设置一次变量：

```bash
export DATASET=lrobench
export RUN_NAME=exp1_seed1
export MODEL=model/qwen3-0.6b
export DIM=2560
export SEED=1
export ALPHA=0.07
export TEMP=15
```

如果要重算某一步：

```bash
export CACHE_POLICY=overwrite
```

1. 一口气跑到 round0 选样，默认 budget 250：

```bash
experiments/bin/cgsd_round0_select.sh
```

2. 训练 round1：

```bash
ROUND=1 experiments/bin/cgsd_train_round.sh
```

3. 评估 round1：

```bash
ROUND=1 experiments/bin/cgsd_eval_round.sh
```

看 `experiments/runs/$DATASET/$RUN_NAME/round_1/round_summary.json`。如果继续：

```bash
ROUND=1 BUDGET=150 experiments/bin/cgsd_select_round.sh
ROUND=2 experiments/bin/cgsd_train_round.sh
ROUND=2 experiments/bin/cgsd_eval_round.sh
```

再看 `round_2/round_summary.json`。如果继续第三轮：

```bash
ROUND=2 BUDGET=100 experiments/bin/cgsd_select_round.sh
ROUND=3 experiments/bin/cgsd_train_round.sh
ROUND=3 experiments/bin/cgsd_eval_round.sh
ROUND=3 experiments/bin/cgsd_finalize.sh
```

确认配置没问题、想 overnight 跑完整默认三轮时，可以用：

```bash
experiments/bin/cgsd_run_exp1_default_3rounds.sh
```

## 原始 CLI 运行步骤

1. 固定划分并校验 embedding：

```bash
python scripts/cgsd_prepare.py \
  --data_path "$DATA" \
  --embeddings_path "$EMB" \
  --output_dir "$OUT" \
  --split_ids_path "$OUT/cgsd_split_ids.json" \
  --n_calibration 200 \
  --seed 1
```

2. 跑 round0 zero-shot 预测：

```bash
python scripts/cgsd_predict.py \
  --output_dir "$OUT" \
  --round_index 0 \
  --model_path "$MODEL" \
  --data_path "$DATA" \
  --split_ids_path "$OUT/cgsd_split_ids.json" \
  --teacher_labels_path "$TEACHER"
```

3. 跑 round0 CRC 校准，实验文档修正版要求固定温度时使用 `--temperature 15`：

```bash
python scripts/cgsd_calibrate.py \
  --output_dir "$OUT" \
  --round_index 0 \
  --temperature 15 \
  --alpha 0.07
```

4. 从 round0 defer 集选择 250 个训练样本：

```bash
python scripts/cgsd_select.py \
  --output_dir "$OUT" \
  --round_index 0 \
  --split_ids_path "$OUT/cgsd_split_ids.json" \
  --embeddings_path "$EMB" \
  --budget 250 \
  --delta 0.1 \
  --teacher_beta 1
```

5. 用累计训练样本训练 round1 LoRA：

```bash
python scripts/cgsd_train_round.py \
  --output_dir "$OUT" \
  --round_index 1 \
  --model_path "$MODEL" \
  --data_path "$DATA" \
  --split_ids_path "$OUT/cgsd_split_ids.json" \
  --train_rows_path "$OUT/cgsd_train_rows.jsonl" \
  --lora_r 1 \
  --lora_target_modules qv \
  --lora_layer_scope all \
  --epochs 3 \
  --lr 2e-4
```

6. 对 round1 重复预测和校准：

```bash
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
  --alpha 0.07 \
  --previous_round_summary_path "$OUT/round_0/round_summary.json" \
  --previous_selection_summary_path "$OUT/round_0/selection_summary.json" \
  --train_rows_path "$OUT/cgsd_train_rows.jsonl"
```

7. 如果 `round_1/round_summary.json` 里没有决定停止，继续 `select --budget 150`、`train_round --round_index 2`、再 `predict/calibrate`。第 2-4 轮先全量重跑 `D_cal + U_pool` 作为校验基准；只有在 accept 集校验通过后，才可以把“后续只重跑上一轮 defer 集”作为单独的工程优化实验记录。

8. 第三轮预算用 100，结束后生成最终部署决策：

```bash
python scripts/cgsd_finalize.py \
  --output_dir "$OUT" \
  --round_index 3
```

## 需要记录

1. 每轮 `round_*/round_summary.json` 的 `pool_summary.defer_rate`、`lambda_hat`、`pool_summary.accept_error_rate`。
2. 每轮 `round_*/calibrate_usage.json`、`select_usage.json`、`train_usage.json` 的调用和 token 账本。
3. 最终 `cgsd_summary.json` 和 `deployment_decisions.jsonl`。
4. CRC 证明口径记录 `round_summary.json` 的 `crc.empirical_risk` 和 `crc.risk_bound`；`pool_summary.accept_error_rate` 只是 accept 子集条件错误率。
5. 第 2-4 轮的 defer/accept 对照校验：以上一轮 `pool_crc_predictions.jsonl` 分出的 defer ID 和 accept ID 为基准，分别在本轮 `pool_student_predictions.jsonl` 和 `pool_crc_predictions.jsonl` 中复查。
6. Defer 集记录改善：上一轮 defer ID 中本轮变 accept 的比例、仍 defer 的比例、预测是否变对、平均 `score/routing_score` 变化。
7. Accept 集记录退化：上一轮 accept ID 中本轮预测翻转、score 明显下降、或从 accept 变 defer 的比例；如果出现退化，要把 damage rate 和对应 ID 样例写进实验记录。
8. 只有当 accept 集退化率足够低且记录完整时，才允许把“只重跑 defer 集”作为工程近似；正式效果和 CRC 口径仍以全量重跑结果为准。

## 严格性说明

当前默认流程复用同一个 `D_cal` 做每轮 CRC、selection defer 集识别和停止判断，能完成工程实验，但不能单独支撑 theorem-level final CRC guarantee。若论文结果需要严格最终保证，应额外准备独立最终校准集，并用 `cgsd_calibrate.py --calibration_predictions_path <final_cal_predictions>` 单独校准最终模型。
