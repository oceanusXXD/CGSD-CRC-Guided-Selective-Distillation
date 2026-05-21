# LLM Layer Test

这个仓库现在只保留一条通用链路：把外部数据转成统一的 query-document
二分类格式，得到 guide/final/pool 集合，按需构建 embedding，做 student
推理，用 CRC 得到 defer 集，再按 random 或 CRC error-mass 选训练数据，
最后训练和推理。

模型输出协议固定为：

- `1`：document 满足 query。
- `0`：document 不满足 query。
- `score = logit/logprob(1) - logit/logprob(0)`。

## 代码入口

- `src/data.py`：对外 dataloader，读取 `id/query/document/groundtruth` JSONL。
- `src/crc.py`：CRC 阈值、defer 集、random 和 CRC error-mass 选样。
- `src/embeddings.py`：读取和校验 embedding 工件。
- `src/model.py`、`src/trainer.py`：LoRA 训练和本地 PyTorch 打分。
- `scripts/cgsd_convert_jsonl.py`：把外部 JSONL 转成统一格式。
- `scripts/cgsd_build_embeddings.py`：构建本地或 vLLM embedding。
- `scripts/cgsd_prepare.py`：生成 `guide_ids/final_ids/pool_ids`。
- `scripts/cgsd_predict_vllm_openai.py`：vLLM 推理，适合基座 round0 和大规模评估。
- `scripts/cgsd_predict_local.py`：本地 PyTorch 推理，适合已训练 checkpoint 的小规模评估。
- `scripts/cgsd_compute_crc.py`：用 guide/pool 预测计算 CRC 和 defer 集。
- `scripts/cgsd_select_data.py`：按 random 或 CRC error-mass 选择训练数据。
- `scripts/cgsd_train_round.py`：训练 LoRA round。

## 环境

```bash
pip install -r requirements.txt
```

示例模型路径：

```text
/teamspace/studios/this_studio/model/qwen3-0.6b
```

## 最小运行顺序

```text
外部数据 -> cgsd_convert_jsonl.py -> data.jsonl
data.jsonl -> cgsd_prepare.py -> cgsd_split_ids.json
data.jsonl -> cgsd_build_embeddings.py -> embeddings.npy / embeddings.ids.jsonl（按需）
round0 基座模型 -> cgsd_predict_vllm_openai.py -> guide/final/pool 预测
guide/pool 预测 -> cgsd_compute_crc.py -> guide/pool CRC 和 defer 集
CRC 结果 -> cgsd_select_data.py -> cgsd_train_rows.jsonl
cgsd_train_rows.jsonl -> cgsd_train_round.py -> round1 LoRA checkpoint
round1 checkpoint -> cgsd_predict_local.py 或 cgsd_predict_vllm_openai.py -> 评估/推理
```

## 路径和复跑约定

建议把外部数据放在 `experiments/inputs/<task>/data.jsonl`，把一次实验的中间产物
放在 `experiments/runs/<task>/<run_name>/`。下面所有命令里的 `my_task` 和
`example_run` 都是占位名，可以换成自己的数据集和实验名。

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

examples = load_examples("experiments/inputs/my_task/data.jsonl")
```

`load_examples` 会检查字段、标签和重复 ID，并返回 `PairExample` 列表。

## 2. 外部数据转换

如果外部数据已经是 JSONL，只是字段名不同：

```bash
python scripts/cgsd_convert_jsonl.py \
  --input_path raw/my_dataset.jsonl \
  --output_path experiments/inputs/my_task/data.jsonl \
  --id_field uid \
  --query_field question \
  --document_field text \
  --label_field label
```

如果整个数据集共用同一个 query：

```bash
python scripts/cgsd_convert_jsonl.py \
  --input_path raw/reviews.jsonl \
  --output_path experiments/inputs/reviews/data.jsonl \
  --id_field review_id \
  --document_field review_text \
  --label_field is_positive \
  --fixed_query "Is this review positive?"
```

输出的 `data.jsonl` 后面所有脚本都能直接用。

`cgsd_convert_jsonl.py` 只做字段映射和 `1/0` 标签校验；如果原始标签是
`yes/no`、`positive/negative`、`true/false` 这类字符串，需要先在外部脚本里
归一化成 `1/0`。

## 3. 构建 Embedding

embedding 是按 `data.jsonl` 行顺序生成的。当前 random 和 CRC error-mass
选样不强制依赖 embedding；这个 stage 保留给需要向量工件的方法，以及
`cgsd_prepare.py --embeddings_path` 的覆盖校验。输出：

```text
experiments/inputs/my_task/
  embeddings.npy
  embeddings.ids.jsonl
  embeddings.meta.json
```

本地 Transformers：

```bash
python scripts/cgsd_build_embeddings.py \
  --data_path experiments/inputs/my_task/data.jsonl \
  --output_path experiments/inputs/my_task/embeddings.npy \
  --ids_path experiments/inputs/my_task/embeddings.ids.jsonl \
  --model_path /teamspace/studios/this_studio/model/qwen3-4b-embedding \
  --backend transformers \
  --request_batch_size 16 \
  --max_length 4096
```

vLLM pooling：

```bash
python scripts/cgsd_build_embeddings.py \
  --data_path experiments/inputs/my_task/data.jsonl \
  --output_path experiments/inputs/my_task/embeddings.npy \
  --ids_path experiments/inputs/my_task/embeddings.ids.jsonl \
  --model_path /teamspace/studios/this_studio/model/qwen3-4b-embedding \
  --backend vllm \
  --request_batch_size 128 \
  --max_length 4096 \
  --tensor_parallel_size 1 \
  --gpu_memory_utilization 0.92
```

断点续跑：保留 `.npy` 和 `.ids.jsonl` 后重跑同一命令会跳过已完成前缀。
从头重建时加 `--overwrite`。

## 4. 生成 Guide / Final / Pool

`cgsd_prepare.py` 的输出只包含三个 ID 字段：

- `guide_ids`：给 CRC 校准和方法统计使用。
- `final_ids`：最终保留集，不参与中间选样和训练。
- `pool_ids`：候选池，用于推理、CRC defer 集和训练样本选择。

```bash
python scripts/cgsd_prepare.py \
  --data_path experiments/inputs/my_task/data.jsonl \
  --output_dir experiments/runs/my_task/example_run \
  --n_guide 1000 \
  --n_final 200 \
  --seed 1 \
  --cache_policy overwrite
```

如果要顺便检查 embedding 覆盖：

```bash
python scripts/cgsd_prepare.py \
  --data_path experiments/inputs/my_task/data.jsonl \
  --output_dir experiments/runs/my_task/example_run \
  --n_guide 1000 \
  --n_final 200 \
  --embeddings_path experiments/inputs/my_task/embeddings.npy \
  --embedding_dim 2560 \
  --seed 1 \
  --cache_policy overwrite
```

输出：

```text
experiments/runs/my_task/example_run/
  cgsd_split_ids.json
  prepare_usage.json
```

## 5. Round0 基座推理

第一轮选数据前还没有 LoRA checkpoint，所以通常用 vLLM 跑基座模型 round0。
这个命令会一次性输出 guide、final 和 pool 预测。

```bash
python scripts/cgsd_predict_vllm_openai.py \
  --output_dir experiments/runs/my_task/example_run \
  --round_index 0 \
  --model_path /teamspace/studios/this_studio/model/qwen3-0.6b \
  --data_path experiments/inputs/my_task/data.jsonl \
  --split_ids_path experiments/runs/my_task/example_run/cgsd_split_ids.json \
  --start_server \
  --base_url http://127.0.0.1:18021/v1 \
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
experiments/runs/my_task/example_run/round_0/
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
- `compute_crc_error_mass_plan(guide_decisions, pool_decisions, budget, ...)`
  计算你的方法需要的 `r_U/r_C/e_all/e_defer/c_crc/eta_crc/s_accept/s_defer`
  和 `B_accept/B_defer`。

CLI：

```bash
python scripts/cgsd_compute_crc.py \
  --output_dir experiments/runs/my_task/example_run \
  --round_index 0 \
  --guide_predictions_path experiments/runs/my_task/example_run/round_0/guide_student_predictions.jsonl \
  --pool_predictions_path experiments/runs/my_task/example_run/round_0/pool_student_predictions.jsonl \
  --alpha 0.1 \
  --temperature 15 \
  --selection_budget 500 \
  --cache_policy overwrite
```

`--selection_budget` 是可选项。传了以后，`crc_summary.json` 里会额外写入
CRC error-mass 的预算拆分预案，方便直接检查 `B_accept/B_defer`。真正选训练
数据时仍以 `cgsd_select_data.py --budget` 为准。

输出：

```text
experiments/runs/my_task/example_run/round_0/
  guide_crc_predictions.jsonl
  pool_crc_predictions.jsonl
  crc_summary.json
  crc_usage.json
```

`pool_crc_predictions.jsonl` 是 pool 的 CRC 决策文件；筛选 `defer=true`
就是 defer 集。

## 7. 选训练数据

选训练数据前必须先跑完第 6 步 `cgsd_compute_crc.py`，生成：

- `guide_crc_predictions.jsonl`：带 `defer` 决策的 guide 行，用来估计 guide
  上的错误率、defer 率和错误浓缩度。
- `pool_crc_predictions.jsonl`：带 `defer` 决策的 pool 行，是训练样本的候选池。
- `crc_summary.json`：记录 `alpha/temperature/lambda_hat`，选样时会复用这些参数。

`crc-error-mass` 必须依赖这个前置 defer 集；没有 `pool_crc_predictions.jsonl`
里的 `defer=true/false`，就无法把预算拆成 accept/defer 两侧。

选样函数在 `src/crc.py`：

- `select_training_ids(..., method="random")`：从 pool 均匀随机选 `budget` 条。
- `select_training_ids(..., method="crc-error-mass")`：用 CRC error-mass plan
  把预算拆成 accept/defer 两侧，再分别抽样。

random baseline：

```bash
python scripts/cgsd_select_data.py \
  --output_dir experiments/runs/my_task/example_run \
  --round_index 0 \
  --method random \
  --budget 500 \
  --seed 1 \
  --cache_policy overwrite
```

CRC error-mass 方法：

```bash
python scripts/cgsd_select_data.py \
  --output_dir experiments/runs/my_task/example_run \
  --round_index 0 \
  --method crc-error-mass \
  --budget 500 \
  --accept_strategy random \
  --defer_strategy random \
  --seed 1 \
  --cache_policy overwrite
```

输出：

```text
experiments/runs/my_task/example_run/round_0/
  selected_train_rows.jsonl
  selection_summary.json
  select_usage.json

experiments/runs/my_task/example_run/
  cgsd_train_rows.jsonl
```

`selected_train_rows.jsonl` 是本轮新增样本；`cgsd_train_rows.jsonl` 是累计训练集，
训练脚本默认读取它。

`random` 会从 pool 全体均匀抽样，不使用 accept/defer 两侧预算；它仍读取
`pool_crc_predictions.jsonl`，只是把 CRC 文件当作带完整 metadata 的 pool 行。
`crc-error-mass` 会用 guide 上的错误浓缩度和 pool 的 defer 率，把预算拆成
accept/defer 两部分。`--accept_strategy high-confidence` 会优先选
`routing_score` 高的 accept 样本；defer 侧目前只支持随机抽样。

## 8. 训练 Round1

```bash
python scripts/cgsd_train_round.py \
  --output_dir experiments/runs/my_task/example_run \
  --round_index 1 \
  --model_path /teamspace/studios/this_studio/model/qwen3-0.6b \
  --data_path experiments/inputs/my_task/data.jsonl \
  --train_rows_path experiments/runs/my_task/example_run/cgsd_train_rows.jsonl \
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
experiments/runs/my_task/example_run/round_1/
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
python scripts/cgsd_predict_local.py \
  --checkpoint_dir experiments/runs/my_task/example_run/round_1/model \
  --model_path /teamspace/studios/this_studio/model/qwen3-0.6b \
  --data_path experiments/inputs/my_task/data.jsonl \
  --split_ids_path experiments/runs/my_task/example_run/cgsd_split_ids.json \
  --split_name pool \
  --max_length 4096 \
  --batch_size 4 \
  --predictions_path experiments/runs/my_task/example_run/round_1/pool_student_predictions.local.jsonl \
  --metrics_path experiments/runs/my_task/example_run/round_1/pool_metrics.local.json
```

`--split_name` 可选 `all/guide/final/pool`。

### vLLM

vLLM 可加载 LoRA checkpoint，适合大规模评估：

```bash
python scripts/cgsd_predict_vllm_openai.py \
  --output_dir experiments/runs/my_task/example_run \
  --round_index 1 \
  --checkpoint_dir experiments/runs/my_task/example_run/round_1/model \
  --model_path /teamspace/studios/this_studio/model/qwen3-0.6b \
  --data_path experiments/inputs/my_task/data.jsonl \
  --split_ids_path experiments/runs/my_task/example_run/cgsd_split_ids.json \
  --start_server \
  --base_url http://127.0.0.1:18021/v1 \
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
python scripts/check_ast_integrity.py
python -m unittest discover -s tests -v
```
