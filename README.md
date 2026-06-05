# 通用二分类选择管线

这个仓库保留一条通用链路：把外部数据转成统一的 query-document
二分类格式，得到 guide/final/pool 集合，按需构建 embedding，做 student
推理，用 CRC 得到 defer 集，再按 random、PCSS 或 CRC error-mass 选训练数据，
最后训练和推理。

本文档使用匿名化占位符，不包含真实项目名、数据源、模型路径、服务器地址或账号信息：

- `/path/to/project/`：公开仓库或实验工作目录。
- `/path/to/data/`：外部数据所在目录。
- `/path/to/model/base-model`：基础模型路径。
- `/path/to/model/embedding-model`：embedding 模型路径。
- `[API_ENDPOINT]`：私有推理服务地址。
- `dataset_a`、`run_a`：匿名化数据集和实验运行名。

模型输出协议固定为：

- `1`：document 满足 query。
- `0`：document 不满足 query。
- `score = logit/logprob(1) - logit/logprob(0)`。

## 代码入口

- `src/data.py`：对外 dataloader，读取 `id/query/document/groundtruth` JSONL。
- `src/crc.py`：CRC 阈值、defer 集、random、PCSS 和 CRC error-mass 选样。
- `src/embeddings.py`：读取和校验 embedding 工件。
- `src/model.py`、`src/trainer.py`：LoRA 训练和本地 PyTorch 打分。
- `scripts/convert_jsonl.py`：把外部 JSONL 转成统一格式。
- `scripts/build_embeddings.py`：构建本地或 vLLM embedding。
- `scripts/prepare.py`：生成 `guide_ids/final_ids/pool_ids`。
- `scripts/predict_vllm_openai.py`：vLLM 推理，适合基座 round0 和大规模评估。
- `scripts/predict_local.py`：本地 PyTorch 推理，适合已训练 checkpoint 的小规模评估。
- `scripts/compute_crc.py`：用 guide/pool 预测计算 CRC 和 defer 集。
- `scripts/select_random.py`：从 pool 均匀随机选择训练数据。
- `scripts/select_pcss.py`：按 guide 标签比例和不确定性选择训练数据。
- `scripts/select_crc_error_mass.py`：按 CRC error-mass 预算拆分选择训练数据。
- `scripts/train_round.py`：训练 LoRA round。

## 环境

```bash
pip install -r requirements.txt
```

示例模型路径：

```text
/path/to/model/base-model
```

## 最小运行顺序

```text
外部数据 -> convert_jsonl.py -> data.jsonl
data.jsonl -> prepare.py -> split_ids.json
data.jsonl -> build_embeddings.py -> embeddings.npy / embeddings.ids.jsonl（按需）
round0 基座模型 -> predict_vllm_openai.py -> guide/final/pool 预测
guide/pool 预测 -> compute_crc.py -> guide/pool CRC 和 defer 集
CRC 结果 -> select_random.py / select_pcss.py / select_crc_error_mass.py -> train_rows.jsonl
train_rows.jsonl -> train_round.py -> round1 LoRA checkpoint
round1 checkpoint -> predict_local.py 或 predict_vllm_openai.py -> 评估/推理
```

## 路径和复跑约定

建议把外部数据整理到 `/path/to/project/experiments/inputs/<dataset>/data.jsonl`，
把一次实验的中间产物放在
`/path/to/project/experiments/runs/<dataset>/<run_name>/`。下面所有命令里的
`dataset_a` 和 `run_a` 都是匿名化占位名，可以替换为公开版本中的数据集和运行名。

各 stage 默认 `--cache_policy reuse`：如果目标产物已经完整存在，脚本会直接复用
并打印已有结果；要重跑就显式传 `--cache_policy overwrite`。如果只想看某个
stage 已经生成的摘要，用 `--show_result`，不会改文件。

每个脚本都有可选的显式输入/输出路径参数；不传时会按 `--output_dir` 和
`--round_index` 推导默认路径。README 示例优先使用默认路径，只有跨 stage
必须指定的源文件才写出来。

## 1. 数据格式

所有外部数据集都先转换成统一 JSONL，每行一个样本：

```json
{"id":"sample_1","query":"Does the document satisfy the requirement?","document":"...","groundtruth":1}
```

必需字段：

- `id`：稳定唯一 ID，不能重复。
- `query`：判断条件、问题或任务描述；必须能回答“document 是否满足 query”。
- `document`：待判断文本。
- `groundtruth`：标签，只接受 `1/0` 或字符串 `"1"` / `"0"`。`1` 表示
  document 满足 query，`0` 表示不满足。外部数据如果原始正负类语义相反，
  转换前要先翻成这个口径。

可选字段：

- `sample_weight` 或 `weight`：训练样本权重，默认 `1.0`。
- 其他字段会保留在 metadata 中，并带到预测、CRC 和选样输出里。

代码里直接用 dataloader：

```python
from src.data import load_examples

examples = load_examples("/path/to/project/experiments/inputs/dataset_a/data.jsonl")
```

`load_examples` 会检查字段、标签和重复 ID，并返回 `PairExample` 列表。

## 2. 外部数据转换

如果外部数据已经是 JSONL，只是字段名不同：

```bash
python scripts/convert_jsonl.py \
  --input_path /path/to/data/dataset_a/raw.jsonl \
  --output_path /path/to/project/experiments/inputs/dataset_a/data.jsonl \
  --id_field uid \
  --query_field question \
  --document_field text \
  --label_field label
```

如果整个数据集共用同一个 query：

```bash
python scripts/convert_jsonl.py \
  --input_path /path/to/data/dataset_a/raw_shared_query.jsonl \
  --output_path /path/to/project/experiments/inputs/dataset_a/data.jsonl \
  --id_field sample_id \
  --document_field document_text \
  --label_field binary_label \
  --fixed_query "Does the document satisfy the target condition?"
```

输出的 `data.jsonl` 后面所有脚本都能直接用。

`convert_jsonl.py` 只做字段映射和 `1/0` 标签校验；如果原始标签是
`yes/no`、`positive/negative`、`true/false` 这类字符串，需要先在外部脚本里
归一化成 `1/0`。

## 3. 构建 Embedding

embedding 是按 `data.jsonl` 行顺序生成的。当前 random、PCSS 和 CRC error-mass
选样不强制依赖 embedding；这个 stage 保留给需要向量工件的方法，以及
`prepare.py --embeddings_path` 的覆盖校验。输出：

```text
/path/to/project/experiments/inputs/dataset_a/
  embeddings.npy
  embeddings.ids.jsonl
  embeddings.meta.json
```

本地 Transformers：

```bash
python scripts/build_embeddings.py \
  --data_path /path/to/project/experiments/inputs/dataset_a/data.jsonl \
  --output_path /path/to/project/experiments/inputs/dataset_a/embeddings.npy \
  --ids_path /path/to/project/experiments/inputs/dataset_a/embeddings.ids.jsonl \
  --model_path /path/to/model/embedding-model \
  --backend transformers \
  --request_batch_size 16 \
  --max_length 4096
```

vLLM pooling：

```bash
python scripts/build_embeddings.py \
  --data_path /path/to/project/experiments/inputs/dataset_a/data.jsonl \
  --output_path /path/to/project/experiments/inputs/dataset_a/embeddings.npy \
  --ids_path /path/to/project/experiments/inputs/dataset_a/embeddings.ids.jsonl \
  --model_path /path/to/model/embedding-model \
  --backend vllm \
  --request_batch_size 128 \
  --max_length 4096 \
  --tensor_parallel_size 1 \
  --gpu_memory_utilization 0.92
```

断点续跑：保留 `.npy` 和 `.ids.jsonl` 后重跑同一命令会跳过已完成前缀。
从头重建时加 `--overwrite`。

## 4. 生成 Guide / Final / Pool

`prepare.py` 的输出只包含三个 ID 字段：

- `guide_ids`：给 CRC 校准和方法统计使用。
- `final_ids`：最终保留集，不参与中间选样和训练。
- `pool_ids`：候选池，用于推理、CRC defer 集和训练样本选择。

```bash
python scripts/prepare.py \
  --data_path /path/to/project/experiments/inputs/dataset_a/data.jsonl \
  --output_dir /path/to/project/experiments/runs/dataset_a/run_a \
  --n_guide 1000 \
  --n_final 200 \
  --seed 1 \
  --cache_policy overwrite
```

如果要顺便检查 embedding 覆盖：

```bash
python scripts/prepare.py \
  --data_path /path/to/project/experiments/inputs/dataset_a/data.jsonl \
  --output_dir /path/to/project/experiments/runs/dataset_a/run_a \
  --n_guide 1000 \
  --n_final 200 \
  --embeddings_path /path/to/project/experiments/inputs/dataset_a/embeddings.npy \
  --embedding_dim 2560 \
  --seed 1 \
  --cache_policy overwrite
```

输出：

```text
/path/to/project/experiments/runs/dataset_a/run_a/
  split_ids.json
  prepare_usage.json
```

## 5. Round0 基座推理

第一轮选数据前还没有 LoRA checkpoint，所以通常用 vLLM 跑基座模型 round0。
这个命令会一次性输出 guide、final 和 pool 预测。

```bash
python scripts/predict_vllm_openai.py \
  --output_dir /path/to/project/experiments/runs/dataset_a/run_a \
  --round_index 0 \
  --model_path /path/to/model/base-model \
  --data_path /path/to/project/experiments/inputs/dataset_a/data.jsonl \
  --split_ids_path /path/to/project/experiments/runs/dataset_a/run_a/split_ids.json \
  --start_server \
  --base_url [API_ENDPOINT] \
  --parallel_requests 256 \
  --max_model_len 40960 \
  --max_num_seqs 1024 \
  --max_num_batched_tokens 131072 \
  --gpu_memory_utilization 0.90 \
  --temperature 0 \
  --max_tokens 1 \
  --top_logprobs 20 \
  --cache_policy overwrite
```

输出：

```text
/path/to/project/experiments/runs/dataset_a/run_a/round_0/
  all_student_predictions.jsonl
  guide_student_predictions.jsonl
  final_student_predictions.jsonl
  pool_student_predictions.jsonl
  all_student_predictions.partial.jsonl
  predict_usage.json
```

CRC 需要的是：

- `guide_student_predictions.jsonl`
- `pool_student_predictions.jsonl`

每行至少包含 `id/score/prediction/label` 或 `id/score/prediction/groundtruth`。
其中 `prediction=1` 表示 student 预测 yes，`prediction=0` 表示预测 no；
`score` 越大越偏向 yes。

## 6. 计算 CRC 和 Defer 集

核心函数在 `src/crc.py`：

- `calibrate_crc(guide_predictions, alpha, temperature)`
  用 guide 预测校准阈值，返回 `CRCResult`，其中包括 `lambda_hat`、
  `risk_bound`、guide accept/defer 数量。
- `apply_crc_defer_set(predictions, lambda_hat, temperature)`
  给任意预测行追加 `routing_score`、`crc_decision`、`defer`、
  `decision_threshold`、`tau_crc`。`defer=true` 的行就是 defer 集。
- `compute_pcss_plan(guide_decisions, pool_decisions, budget, ...)`
  用 guide 真实标签估计目标标签比例，并计算 PCSS 的 `B_label0/B_label1`
  预算。

CLI：

```bash
python scripts/compute_crc.py \
  --output_dir /path/to/project/experiments/runs/dataset_a/run_a \
  --round_index 0 \
  --guide_predictions_path /path/to/project/experiments/runs/dataset_a/run_a/round_0/guide_student_predictions.jsonl \
  --pool_predictions_path /path/to/project/experiments/runs/dataset_a/run_a/round_0/pool_student_predictions.jsonl \
  --alpha 0.1 \
  --temperature 15 \
  --selection_budget 500 \
  --cache_policy overwrite
```

`--selection_budget` 是可选项。传了以后，`crc_summary.json` 里会额外写入
PCSS 的预算拆分预案，方便直接检查 `B_label0/B_label1`。真正选训练数据时仍以
对应选样入口的 `--budget` 为准。

输出：

```text
/path/to/project/experiments/runs/dataset_a/run_a/round_0/
  guide_crc_predictions.jsonl
  pool_crc_predictions.jsonl
  crc_summary.json
  crc_usage.json
```

`pool_crc_predictions.jsonl` 是 pool 的 CRC 决策文件；筛选 `defer=true`
就是 defer 集。

## 7. 选训练数据

选训练数据前必须先跑完第 6 步 `compute_crc.py`，生成：

- `guide_crc_predictions.jsonl`：带 `defer` 决策的 guide 行，用来估计 guide
  上的错误率、defer 率和错误浓缩度。
- `pool_crc_predictions.jsonl`：带 `defer` 决策的 pool 行，是训练样本的候选池。
- `crc_summary.json`：记录 `alpha/temperature/lambda_hat`，选样时会复用这些参数。

PCSS 必须依赖这个前置 CRC 打分文件；没有 `pool_crc_predictions.jsonl`
里的 `routing_score`，就无法在每个标签层内按不确定性选择样本。

选样函数在 `src/crc.py`：

- `select_training_ids(..., method="random")`：从 pool 均匀随机选 `budget` 条。
- `select_training_ids(..., method="pcss")`：先按 guide 标签比例拆成
  proxy label 0/1 两个预算，再在每个层内按 `routing_score` 从低到高选择。

random baseline：

```bash
python scripts/select_random.py \
  --output_dir /path/to/project/experiments/runs/dataset_a/run_a \
  --round_index 0 \
  --budget 500 \
  --seed 1 \
  --cache_policy overwrite
```

PCSS 方法：

```bash
python scripts/select_pcss.py \
  --output_dir /path/to/project/experiments/runs/dataset_a/run_a \
  --round_index 0 \
  --budget 500 \
  --seed 1 \
  --cache_policy overwrite
```

CRC error-mass 方法：

```bash
python scripts/select_crc_error_mass.py \
  --output_dir /path/to/project/experiments/runs/dataset_a/run_a \
  --round_index 0 \
  --budget 500 \
  --accept_strategy random \
  --defer_strategy high-confidence \
  --seed 1 \
  --cache_policy overwrite
```

输出：

```text
/path/to/project/experiments/runs/dataset_a/run_a/round_0/
  selected_train_rows.jsonl
  selection_summary.json
  select_usage.json

/path/to/project/experiments/runs/dataset_a/run_a/
  train_rows.jsonl
```

`selected_train_rows.jsonl` 是本轮新增样本；`train_rows.jsonl` 是累计训练集，
训练脚本默认读取它。

`random` 会从 pool 全体均匀抽样，不使用 accept/defer 两侧预算；它仍读取
`pool_crc_predictions.jsonl`，只是把 CRC 文件当作带完整 metadata 的 pool 行。
`pcss` 会用 guide 上的真实标签比例约束训练集 proxy 标签分布，然后在每个
proxy 标签层内优先选择 `routing_score` 更低、更不确定的样本。
`crc-error-mass` 会先按 CRC error-mass 计划拆分 accept/defer 预算，再分别按
`random` 或 `high-confidence` 策略选择两侧样本。

## 8. 训练 Round1

```bash
python scripts/train_round.py \
  --output_dir /path/to/project/experiments/runs/dataset_a/run_a \
  --round_index 1 \
  --model_path /path/to/model/base-model \
  --data_path /path/to/project/experiments/inputs/dataset_a/data.jsonl \
  --train_rows_path /path/to/project/experiments/runs/dataset_a/run_a/train_rows.jsonl \
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

```text
/path/to/project/experiments/runs/dataset_a/run_a/round_1/
  model/
    adapter/
    model_config.json
  training_rows_used.jsonl
  train_label_snapshot.json
  training_round_summary.json
  train_usage.json
```

训练不会用 guide 做 early stopping 或最佳 epoch 选择；guide 只作为保留集合。

## 9. Round1 推理

### 本地 PyTorch

本地推理需要已有 checkpoint，适合小规模检查：

```bash
python scripts/predict_local.py \
  --checkpoint_dir /path/to/project/experiments/runs/dataset_a/run_a/round_1/model \
  --model_path /path/to/model/base-model \
  --data_path /path/to/project/experiments/inputs/dataset_a/data.jsonl \
  --split_ids_path /path/to/project/experiments/runs/dataset_a/run_a/split_ids.json \
  --split_name pool \
  --max_length 4096 \
  --batch_size 4 \
  --predictions_path /path/to/project/experiments/runs/dataset_a/run_a/round_1/pool_student_predictions.local.jsonl \
  --metrics_path /path/to/project/experiments/runs/dataset_a/run_a/round_1/pool_metrics.local.json
```

`--split_name` 可选 `all/guide/final/pool`。

### vLLM

vLLM 可加载 LoRA checkpoint，适合大规模评估：

```bash
python scripts/predict_vllm_openai.py \
  --output_dir /path/to/project/experiments/runs/dataset_a/run_a \
  --round_index 1 \
  --checkpoint_dir /path/to/project/experiments/runs/dataset_a/run_a/round_1/model \
  --model_path /path/to/model/base-model \
  --data_path /path/to/project/experiments/inputs/dataset_a/data.jsonl \
  --split_ids_path /path/to/project/experiments/runs/dataset_a/run_a/split_ids.json \
  --start_server \
  --base_url [API_ENDPOINT] \
  --parallel_requests 256 \
  --max_model_len 40960 \
  --max_num_seqs 1024 \
  --max_num_batched_tokens 131072 \
  --gpu_memory_utilization 0.90 \
  --temperature 0 \
  --max_tokens 1 \
  --top_logprobs 20 \
  --cache_policy overwrite
```

## 10. 代码检查

```bash
python -m unittest discover -s tests -v
```
