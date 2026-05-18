# CGSD-CRC-Guided-Selective-Distillation

本仓库实现 **CGSD：CRC 引导的选择性蒸馏**。任务形式是二分类：
给定 `query` 和 `document`，小模型判断文档是否满足 query，统一输出
`1/0`：`1` 表示满足，`0` 表示不满足。

核心流程是：

1. 小模型先对引导集、认证集和候选池做预测，并保存 `1/0` logprob margin。
2. CRC 用引导集校准 accept/defer 阈值。
3. 从 defer 集里选样本标注，当前主策略是 k-Center Greedy。
4. 用累计标注样本训练下一轮 LoRA。
5. 循环到预算用完或 defer 率下降不明显。
6. 最终用一直隔离的认证集做最终 CRC 认证。

## 环境

```bash
pip install -r requirements.txt
```

默认本地模型路径：

```text
model/qwen3-0.6b
```

如果模型在别处，用各脚本的 `--model_path` 覆盖。

## 数据与目录

输入数据使用 JSONL，每行一个样本：

```json
{"id":"sample_1","query":"Does the evidence support the claim?","document":"Claim: ... Evidence: ...","groundtruth":1}
```

必需字段：

- `id`：稳定唯一 ID。
- `query`：查询或判断条件。
- `document`：待判断文档。
- `groundtruth`：二分类标签，`1` 表示满足，`0` 表示不满足。

推荐输出目录结构：

```text
experiments/runs/<task>/<run_name>/
  cgsd_split_ids.json
  cgsd_train_rows.jsonl
  round_1/
    all_student_predictions.jsonl
    calibration_student_predictions.jsonl
    final_calibration_student_predictions.jsonl
    pool_student_predictions.jsonl
    pool_crc_predictions.jsonl
    selected_train_rows.jsonl
    model/
    round_summary.json
```

## 准备 split 和 embedding

先准备引导集、认证集和候选池 ID。`D_guide` 用于中间 CRC，`D_cert`
只用于最终认证，不能参与中间选样或训练。

```bash
python scripts/cgsd_prepare.py \
  --data_path experiments/inputs/fever/data.jsonl \
  --embeddings_path experiments/inputs/fever/embeddings.npy \
  --output_dir experiments/runs/fever/example_run \
  --embedding_dim 2560 \
  --n_calibration 200 \
  --n_final_calibration 200 \
  --seed 1
```

如果还没有 embedding，可先构建：

```bash
python scripts/cgsd_build_embeddings.py \
  --data_path experiments/inputs/fever/data.jsonl \
  --output_path experiments/inputs/fever/embeddings.npy \
  --ids_path experiments/inputs/fever/embeddings.ids.jsonl
```

embedding 会按批写入 `.npy` memmap，并把已完成样本 ID 追加到
`.ids.jsonl`。如果任务中断，保留这两个文件后重跑同一命令，会从
已完成前缀之后继续；需要从头重建时再加 `--overwrite`。

## 预测

`scripts/cgsd_predict_vllm_openai.py` 通过 OpenAI-compatible vLLM server 做高吞吐推理。
这是唯一的 CGSD student 推理入口，避免多套预测实现产生协议漂移。

```bash
python scripts/cgsd_predict_vllm_openai.py \
  --output_dir experiments/runs/fever/example_run \
  --round_index 1 \
  --model_path /teamspace/studios/this_studio/model/qwen3-0.6b \
  --data_path experiments/inputs/fever/data.jsonl \
  --start_server \
  --base_url http://127.0.0.1:18021/v1 \
  --parallel_requests 1024 \
  --max_model_len 40960 \
  --max_num_seqs 4096 \
  --max_num_batched_tokens 524288 \
  --gpu_memory_utilization 0.98 \
  --temperature 0 \
  --max_tokens 1 \
  --top_logprobs 20 \
  --cache_policy overwrite
```

预测也会写 `all_student_predictions.partial.jsonl`。中断后重跑同一
命令时会读取 partial，跳过已经完成的样本，并重新生成最终 JSONL。

输出文件：

- `all_student_predictions.jsonl`：引导集 + 认证集 + pool 的全量预测。
- `calibration_student_predictions.jsonl`：中间 CRC 的引导集预测。
- `final_calibration_student_predictions.jsonl`：最终认证集预测。
- `pool_student_predictions.jsonl`：候选池预测。

预测行里的核心字段：

- `score`：`1_logprob - 0_logprob`，即 logit/logprob margin。
- `prediction`：`score > 0` 为 `1`，否则为 `0`。
- `probability`：有方向的 `sigmoid(score)`，不是 CRC 路由分数。

## CRC 校准

中间轮 CRC 使用 `D_guide` 的 `calibration_student_predictions.jsonl`
校准阈值，再对 pool 写出 accept/defer 决策：

```bash
python scripts/cgsd_calibrate.py \
  --output_dir experiments/runs/fever/example_run \
  --round_index 1 \
  --alpha 0.1 \
  --temperature 15 \
  --embeddings_path experiments/inputs/fever/embeddings.npy \
  --cache_policy overwrite
```

核心计算在 `algorithms/cgsd.py`：

- `routing_score = sigmoid(abs(score) / temperature)`。
- 无 neighbor support 时：`routing_score >= lambda_hat` 则 accept。
- 传入 `--embeddings_path` 时启用 neighbor support，自适应调整每条样本的
  `decision_threshold`。
- 风险修正为 `n/(n+1) * empirical_risk + 1/(n+1)`。

主要输出：

- `pool_crc_predictions.jsonl`：每条 pool 样本的 `routing_score`、
  `decision_threshold`、`crc_decision`、`defer`。
- `round_summary.json`：`lambda_hat`、`temperature`、`pool_summary`、
  `pool_metrics`、停止判断。

## 选数据

主策略是从当前 defer 集里做 k-Center Greedy：

```bash
python scripts/cgsd_select.py \
  --output_dir experiments/runs/fever/example_run \
  --round_index 1 \
  --embeddings_path experiments/inputs/fever/embeddings.npy \
  --embedding_dim 2560 \
  --budget 150 \
  --seed 1 \
  --cache_policy overwrite
```

逻辑：

- 候选集来自 `pool_crc_predictions.jsonl` 中 `defer=true` 的样本。
- 排除已经标注过的样本和引导集样本。
- embedding 先单位归一化。
- 第一个点选离 defer 候选质心最远的样本。
- 后续每轮选“到已选集合最近距离最大”的样本。

对照 baseline 可用：

```bash
python scripts/cgsd_make_baseline_rows.py \
  --output_dir experiments/runs/fever/baseline_random \
  --round_index 1 \
  --strategy defer-random \
  --budget 150 \
  --pool_student_predictions_path experiments/runs/fever/baseline_random/round_1/pool_student_predictions.jsonl \
  --pool_crc_predictions_path experiments/runs/fever/baseline_random/round_1/pool_crc_predictions.jsonl
```

支持的 baseline：`random`、`uncertainty`、`k-center`、`defer-random`。

## 训练

用累计的 `cgsd_train_rows.jsonl` 训练一个 LoRA round：

```bash
python scripts/cgsd_train_round.py \
  --output_dir experiments/runs/fever/example_run \
  --round_index 1 \
  --model_path /teamspace/studios/this_studio/model/qwen3-0.6b \
  --data_path experiments/inputs/fever/data.jsonl \
  --lora_r 1 \
  --lora_alpha 16 \
  --lora_dropout 0.05 \
  --lora_target_modules attention_mlp \
  --lora_layer_scope all \
  --lr 0.0002 \
  --epochs 3 \
  --batch_size 4 \
  --gradient_accumulation_steps 4 \
  --max_length 512 \
  --cache_policy overwrite
```

输出：

- `round_<n>/model/adapter/`：LoRA adapter。
- `round_<n>/model/model_config.json`：训练超参和 label 分布。
- `round_<n>/training_round_summary.json`：训练样本数和 checkpoint 路径。

## 评估

`scripts/evaluate.py` 直接加载 checkpoint，在 PyTorch/HF 路径下评估：

```bash
python scripts/evaluate.py \
  --checkpoint_dir experiments/runs/fever/example_run/round_1/model \
  --model_path /teamspace/studios/this_studio/model/qwen3-0.6b \
  --data_path experiments/inputs/fever/data.jsonl \
  --max_length 40960 \
  --batch_size 4 \
  --predictions_path experiments/runs/fever/example_run/round_1/eval_predictions.jsonl \
  --metrics_path experiments/runs/fever/example_run/round_1/eval_metrics.json
```

指标包括 `accuracy`、`precision`、`recall`、`f1`、`macro_F1`。

## 评估：预测文件

已落盘预测 JSONL 可以直接计算 accuracy、正类 F1、
macro F1 和混淆矩阵：

```bash
python - <<'PY'
import json
from pathlib import Path

path = Path("experiments/runs/fever/example_run/round_1/pool_student_predictions.jsonl")
tp = tn = fp = fn = n = 0
for line in path.open():
    row = json.loads(line)
    y = int(row.get("label", row.get("groundtruth")))
    p = int(row["prediction"]) if "prediction" in row else int(float(row["score"]) > 0)
    n += 1
    tp += y == 1 and p == 1
    tn += y == 0 and p == 0
    fp += y == 0 and p == 1
    fn += y == 1 and p == 0

acc = (tp + tn) / n
precision = tp / (tp + fp) if tp + fp else 0.0
recall = tp / (tp + fn) if tp + fn else 0.0
f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
neg_precision = tn / (tn + fn) if tn + fn else 0.0
specificity = tn / (tn + fp) if tn + fp else 0.0
neg_f1 = 2 * neg_precision * specificity / (neg_precision + specificity) if neg_precision + specificity else 0.0
print(json.dumps({
    "n": n,
    "accuracy": acc,
    "f1": f1,
    "macro_F1": (f1 + neg_f1) / 2,
    "tp": tp,
    "tn": tn,
    "fp": fp,
    "fn": fn,
}, indent=2))
PY
```

如果要评估 CRC 后的小模型+教师路由系统，用 `pool_crc_predictions.jsonl`
或最终 `deployment_decisions.jsonl` 统计 accept/defer、teacher 调用量和最终输出。

## 最终认证

中间轮的 `round_summary.json` 用的是 `D_guide`，主要服务于选样和停止判断。
最终数学认证应当在训练结束后，用最终模型对 `D_cert` 预测，再只用
`final_calibration_student_predictions.jsonl` 做一次 CRC 校准，得到
最终阈值 `lambda_hat*`。`D_cert` 在此之前不能参与训练、选样或中间 CRC。

## 结束部署摘要

根据某一轮 `round_summary.json` 和 `pool_crc_predictions.jsonl` 生成部署决策：

```bash
python scripts/cgsd_finalize.py \
  --output_dir experiments/runs/fever/example_run \
  --round_index 1
```

输出：

- `deployment_decisions.jsonl`：accept 样本用小模型输出，defer 样本需要 teacher。
- `cgsd_summary.json`：最终 round、阈值、defer 调用量和 checkpoint 路径。

## 常用排错

- `missing 1/0 logprobs`：提高 `--top_logprobs`，确认 prompt 要求只输出 1/0。
- k-Center 报 embedding 缺失：确认 embedding 文件覆盖所有 pool 样本 ID。
- 最终认证不要复用中间 `round_summary.json` 的 `lambda_hat` 作为严格保证阈值。
