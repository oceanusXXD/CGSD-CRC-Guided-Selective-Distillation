# CGSD-CRC-Guided-Selective-Distillation

本仓库实现和整理 **CGSD：CRC 引导的选择性蒸馏**。

目标任务是二分类过滤：给定 `query` 和 `document`，判断文档是否满足查询条件。CGSD 让小模型先判断；当 CRC 校准认为小模型不够可靠时，再把样本交给教师模型或后续蒸馏流程处理。

## 仓库内容

- 核心算法：[algorithms/cgsd.py](algorithms/cgsd.py)
- CGSD 分阶段命令行脚本：[scripts/](scripts)
- 参数高效微调基线：[scripts/train.py](scripts/train.py)、[scripts/train_prefix_tuning.py](scripts/train_prefix_tuning.py)
- 实验说明：[experiments/](experiments)
- 本地 JSONL 数据：[datasets/](datasets)

`outputs/`、模型权重、向量输出和教师接口输出不会提交到仓库。IMDb 的 JSONL 数据使用 Git LFS 管理。

## 环境

```bash
pip install -r requirements.txt
```

默认本地模型路径：

```text
model/qwen3-0.6b
```

也可以在脚本里用 `--model_path` 指定其他路径。

## 数据格式

输入数据使用 JSONL，每行一个样本：

```json
{"id":"q1__d2","query":"这篇文档是否相关？","document":"文档正文...","groundtruth":1}
```

字段含义：

- `id`：稳定且唯一的样本 ID
- `query`：过滤条件或查询文本
- `document`：待判断文本
- `groundtruth`：二分类标签，`1` 表示满足，`0` 表示不满足

当前仓库里的数据还包含对齐用字段：`query_id`、`document_id`、`review_id`、`parsed_answer`，部分文件还包含 `parsed_confidence`。训练和 CGSD 脚本默认只读取 `id/query/document/groundtruth`。

## 当前数据

| 文件 | 行数 | `groundtruth=0` | `groundtruth=1` |
| --- | ---: | ---: | ---: |
| `datasets/query_id_1.jsonl` | 9,297 | 8,716 | 581 |
| `datasets/query_id_2.jsonl` | 9,297 | 3,538 | 5,759 |
| `datasets/query_id_3.jsonl` | 9,297 | 6,903 | 2,394 |
| `datasets/twitter_hate_query_id_1.jsonl` | 24,783 | 4,163 | 20,620 |
| `datasets/imdb_query_id_1.jsonl` | 49,990 | 24,999 | 24,991 |
| `datasets/imdb_query_id_2.jsonl` | 49,990 | 36,247 | 13,743 |
| `datasets/imdb_query_id_3.jsonl` | 49,990 | 47,382 | 2,608 |

配套的 `*.metadata.json` 记录了原始来源、query 过滤条件、标签分布和超过 2048 token 后被移除的样本数。IMDb 三个 JSONL 文件由 Git LFS 管理，首次克隆后需要确认 LFS 文件已拉取。

CGSD 还需要覆盖全部样本 ID 的向量文件：

```json
{"id":"q1__d2","embedding":[0.12,-0.03,0.44]}
```

当前仓库没有提交向量文件，需要在实验前按同一批 `id` 生成。教师标签文件可选；如果不提供，离线流程会用 `groundtruth` 作为教师标签替代。

## 快速运行 CGSD

```bash
export DATA=datasets/query_id_1.jsonl
export EMB=datasets/query_id_1.embeddings.jsonl
export MODEL=model/qwen3-0.6b
export OUT=outputs/cgsd_q1

python scripts/cgsd_prepare.py \
  --data_path "$DATA" \
  --embeddings_path "$EMB" \
  --output_dir "$OUT" \
  --n_calibration 200 \
  --seed 1

python scripts/cgsd_predict.py \
  --output_dir "$OUT" \
  --round_index 0 \
  --model_path "$MODEL" \
  --data_path "$DATA"

python scripts/cgsd_calibrate.py \
  --output_dir "$OUT" \
  --round_index 0 \
  --temperature 15 \
  --alpha 0.07

python scripts/cgsd_select.py \
  --output_dir "$OUT" \
  --round_index 0 \
  --embeddings_path "$EMB" \
  --budget 250

python scripts/cgsd_train_round.py \
  --output_dir "$OUT" \
  --round_index 1 \
  --model_path "$MODEL" \
  --data_path "$DATA"
```

训练下一轮后，需要重新执行预测和校准，再按实验要求继续选择样本或生成最终结果。

正式实验建议每轮都对完整 `D_cal + U_pool` 重新预测；只对延迟集合重推理只能作为工程近似，前提是已经确认接受集合没有被 LoRA 训练破坏。

## 实验入口

详细实验要求写在各自文件夹的 README 中：

- [01 核心有效性](experiments/01_core_effectiveness)
- [02 选择策略对比](experiments/02_selection_comparison)
- [03 标注预算曲线](experiments/03_budget_curve)
- [04 消融实验](experiments/04_ablation)
- [05 端到端系统对比](experiments/05_end_to_end_comparison)
- [06 CRC 保证验证](experiments/06_crc_guarantee)
- [07 三角关系验证](experiments/07_triangle_relation)
- [08 LROBench 实验](experiments/08_lrobench)

## 基础校验

```bash
python -m unittest tests.test_layer_contracts
python scripts/cgsd_prepare.py --help
python scripts/cgsd_calibrate.py --help
```

完整训练和评估需要本地模型与 GPU 资源。
