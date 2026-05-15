# 实验 5：端到端系统对比

当前代码可以产出 CGSD 和本地 0.6B LoRA/CRC 结果；Full GPT-5、4B zero-shot cascade、4B 二次分流需要外部系统或已有结果文件。

## 数据前置

1. 所有方法必须使用同一份 `DATA` 和同一套 oracle 标签。
2. CGSD 需要 embedding 文件；4B cascade 和 GPT-5 baseline 的输出不由当前仓库生成。
3. 每个方法单独输出目录，例如 `outputs/exp5_cgsd`、`outputs/exp5_random_sft`。
4. 外部方法需要提前整理成同一张结果表，至少包含 `method/raw_accuracy/final_accuracy/defer_rate/teacher_calls/student_calls/teacher_prompt_tokens/teacher_completion_tokens/student_prompt_tokens/student_completion_tokens/estimated_cost`。
5. 如果 oracle 用真实 API teacher，先生成覆盖全量样本的 teacher 文件；如果 oracle 用公开 groundtruth，所有方法都必须使用同一份 `groundtruth`。

## Baseline 要求和复用

本实验的每一行方法都要在本目录结果表里有独立记录，不要只引用其他文档里的数字。

1. CGSD：如果采用实验 1 的完整多轮配置，可以直接复用实验 1 的 `cgsd_summary.json`、`deployment_decisions.jsonl`、各 `round_*/round_summary.json` 和 usage JSON；如果改成单轮 `budget=500`，必须单独输出并在表里标成不同 variant。
2. Random SFT + CRC：可以复用实验 2 的 Random 训练行和 split，前提是数据、oracle、seed 和训练预算完全一致；否则按本实验重新生成 `$OUT/cgsd_train_rows.jsonl`。
3. 普通 LoRA 参考线：只用于裸准确率参考，不自带 CRC/defer 口径；若要放进端到端表，需要额外说明没有 teacher defer 或另行接 CRC。
4. Full GPT-5：需要外部结果或重新跑 teacher 全量推理，至少记录 teacher calls、prompt/completion token、最终准确率和成本。
5. Qwen3-4B zero-shot cascade 与 Qwen3-4B 二次分流：需要前序系统输出或外部重跑结果，至少记录 raw/final accuracy、defer rate、student/teacher calls、token 和成本。
6. 所有外部 baseline 必须使用同一份 `DATA` 和同一套 oracle；如果来源只给聚合数字，要在结果表里注明数据版本、oracle 来源和是否可逐 ID 对齐。

## CGSD 方法

按实验 1 跑完整 CGSD，建议固定总预算 500 和温度 15。

```bash
python scripts/cgsd_prepare.py --data_path "$DATA" --embeddings_path "$EMB" --output_dir "$OUT" --n_calibration 200 --seed 1
python scripts/cgsd_predict.py --output_dir "$OUT" --round_index 0 --model_path "$MODEL" --data_path "$DATA" --teacher_labels_path "$TEACHER"
python scripts/cgsd_calibrate.py --output_dir "$OUT" --round_index 0 --temperature 15 --alpha 0.07
python scripts/cgsd_select.py --output_dir "$OUT" --round_index 0 --embeddings_path "$EMB" --budget 500
python scripts/cgsd_train_round.py --output_dir "$OUT" --round_index 1 --model_path "$MODEL" --data_path "$DATA"
python scripts/cgsd_predict.py --output_dir "$OUT" --round_index 1 --model_path "$MODEL" --data_path "$DATA" --checkpoint_dir "$OUT/round_1/model" --teacher_labels_path "$TEACHER"
python scripts/cgsd_calibrate.py --output_dir "$OUT" --round_index 1 --temperature 15 --alpha 0.07
python scripts/cgsd_finalize.py --output_dir "$OUT" --round_index 1
```

## Random SFT + CRC

1. 手工从 pool 中随机生成 `$OUT/cgsd_train_rows.jsonl`，格式与 `cgsd_select.py` 输出一致。
2. 复用 `cgsd_train_round.py -> cgsd_predict.py -> cgsd_calibrate.py -> cgsd_finalize.py`。

## 普通 LoRA 参考线

如果只需要普通 supervised LoRA 的裸准确率，用：

```bash
python scripts/train.py \
  --mode lora_attention_mlp \
  --model_path "$MODEL" \
  --data_path "$DATA" \
  --output_dir outputs/exp5_lora_reference \
  --balance_train_classes

python scripts/evaluate.py \
  --checkpoint_dir outputs/exp5_lora_reference \
  --data_path "$DATA" \
  --split_name test
```

## 外部结果放置

把外部系统结果统一记录到本实验目录的结果表中：

1. Full GPT-5：teacher 总调用数、最终准确率、成本。
2. 4B ZS Cascade：defer 率、最终准确率、成本。
3. 4B 二次分流：defer 率、最终准确率、成本。

建议外部结果文件格式：

```json
{"method":"full_gpt5","raw_accuracy":null,"final_accuracy":1.0,"defer_rate":null,"teacher_calls":165447,"student_calls":0,"teacher_prompt_tokens":0,"teacher_completion_tokens":165447,"student_prompt_tokens":0,"student_completion_tokens":0,"estimated_cost":81.63}
```

CGSD 的 teacher/student 调用以 `cgsd_summary.json` 和各 stage usage JSON 为准，不要只用手写公式。凡是调用模型或复用 teacher 标签的地方都要保留 call/token 来源：`predict_usage.json` 记录 student 推理，`calibrate_usage.json` 记录校准标签使用，`select_usage.json` 记录选中样本 teacher/groundtruth 标签和 embedding 使用，`train_usage.json` 记录训练样本与训练 token 估算，`finalize_usage.json` 记录最终 defer teacher 调用。
