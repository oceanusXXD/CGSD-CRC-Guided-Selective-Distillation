# CGSD 实验执行总览

这个目录把 `CGSD_experiment_plan.md` 里的实验拆成可执行说明。当前代码的 CGSD 主链路已经是 CLI 驱动：每个 stage 只读本 stage 的 CLI 参数和输入 artifact，不依赖 `cgsd_config.json` 或 `cgsd_state.json`。

代码和文档一致性审计见 [DOC_CODE_AUDIT.md](/teamspace/studios/this_studio/LLM_layer_test/experiments/DOC_CODE_AUDIT.md)。

## 当前代码支持情况

| 实验 | 当前代码状态 | 主要缺口 |
| --- | --- | --- |
| 01 核心有效性 | 可跑完整 CGSD 闭环 | 需要外部准备 FEVER/LROBench JSONL、embedding、可选 teacher 文件 |
| 02 数据选择对比 | DBDS 可直接跑，其他策略需手工生成训练行 | 没有 Random/Uncertainty/k-Center baseline 的独立 CLI |
| 03 标注预算曲线 | DBDS 预算扫描可跑 | 没有自动聚合和画图脚本 |
| 04 消融 | delta、轮次、LoRA rank、teacher beta、easy anchor 可跑 | band 比例消融未暴露 CLI 参数 |
| 05 端到端对比 | CGSD 和本地 LoRA 可跑 | Full GPT-5、4B cascade、二次分流需要外部结果 |
| 06 CRC 保证验证 | 多 seed、多 alpha 可跑 | 没有自动 20 次聚合脚本；严格保证需要独立最终校准集 |
| 07 三角关系 | 可由 03 和 06 的结果汇总 | 没有成本曲线/图表脚本 |
| 08 LROBench | per-query 独立模式可跑 | cross-query LoRA 参数平均/迁移未实现 |

## 统一数据前置

1. 把实验数据转换为 JSONL，每行至少包含 `id`、`query`、`document`、`groundtruth`。
2. `id` 必须稳定且全局唯一；重复运行、embedding、teacher 文件、split 文件都用这个 `id` 对齐。
3. `groundtruth` 必须能归一成二分类 `1/0`；`yes/true/1` 会映射为 `1`，`no/false/0` 会映射为 `0`。
4. 如果原数据不是这些字段名，优先在 CLI 里传 `--query_field`、`--document_field`、`--label_field`，不要改代码。
5. FEVER 可预处理成：`query=claim`，`document=evidence 或候选 Wikipedia 文本`，`groundtruth=支持/相关为 1，不支持/不相关为 0`。
6. LROBench 可预处理成：每个 query 一个 JSONL，`query=筛选条件`，`document=行文本或序列化后的 row`，`groundtruth=该 row 是否满足筛选条件`。

最小数据行示例：

```json
{"id":"fever_000001","query":"The claim text","document":"Candidate evidence text","groundtruth":1}
```

## Embedding 输入格式

1. `cgsd_prepare.py` 和 `cgsd_select.py` 都需要 `--embeddings_path`。
2. embedding 必须覆盖数据文件里的所有 `id`，默认维度是 `1024`；维度不同就显式传 `--embedding_dim`。
3. 当前仓库只读取预计算 embedding，不负责调用 embedding 模型生成向量。
4. 支持三种格式：

JSONL：

```json
{"id":"fever_000001","embedding":[0.12,-0.03,0.44]}
```

JSON object：

```json
{"fever_000001":[0.12,-0.03,0.44]}
```

NPZ：每个 key 是样本 `id`，value 是一维向量。

## Teacher 输入格式

1. 真实 teacher/API 结果通过 `cgsd_predict.py --teacher_labels_path` 传入。
2. 不传 `--teacher_labels_path` 时，代码用数据集 `groundtruth` 作为离线 teacher 替代。
3. 真实 teacher 文件至少需要 `id` 和 `teacher_label`；也可以只给 `teacher_logit_margin`，代码会用 margin 符号生成标签。
4. 可选字段 `teacher_confidence` 或 `teacher_logit_margin` 会进入 `sample_weight` 的 teacher 加权。
5. teacher 文件没有覆盖的 `id` 会回落到 `groundtruth` 替代；如果要统计真实 API 调用量，teacher 文件应覆盖所有会进入预测、校准和选择的样本。

Teacher JSONL 示例：

```json
{"id":"fever_000001","teacher_label":1,"teacher_confidence":0.93}
```

也支持 JSON list、`{"rows":[...]}`、`{"predictions":[...]}` 或 `{"id": 1}` 这种映射结构。

## Cache 和单步运行

每个 CGSD stage 都支持：

- `--cache_policy reuse`：默认值；完整输出已存在时复用，部分输出存在时报错。
- `--cache_policy overwrite`：明确重算并覆盖本 stage 输出。
- `--cache_policy fail`：任一输出已存在就失败，适合防止误覆盖。
- `--show_result`：只展示已有主结果，不执行计算、不写文件。

每个实验、每个 seed、每个预算点都应使用独立 `--output_dir`，避免不同实验的 cache 混在一起。

## 通用变量

下面变量在各实验 README 中复用，按实际路径改：

```bash
export MODEL=model/qwen3-0.6b
export DATA=datasets/fever.jsonl
export EMB=datasets/fever.embeddings.jsonl
export TEACHER=datasets/fever.teacher.jsonl
export OUT=outputs/cgsd_exp
```

如果没有真实 teacher 文件，删掉命令里的 `--teacher_labels_path "$TEACHER"`。

## 通用 CGSD 单轮顺序

1. `cgsd_prepare.py`：固定 `D_cal` 和 `U_pool`，校验 embedding 覆盖。
2. `cgsd_predict.py`：对 calibration 和 pool 全量 student 推理，并附加 teacher 或 groundtruth 替代标签。
3. `cgsd_calibrate.py`：用 calibration 预测校准 CRC，写出 pool accept/defer。
4. `cgsd_select.py`：从 defer 集选择训练样本，并额外选择 easy anchor。
5. `cgsd_train_round.py`：从 base model 重新训练下一轮 LoRA。
6. 重复 predict/calibrate/select/train，最后用 `cgsd_finalize.py` 生成部署决策。

严格实验默认固定温度 `--temperature 15`。代码仍保留 round0 温度扫描能力，但它不应作为严格 CRC 保证实验的默认口径。

## 指标口径

1. CRC 证明口径是 `round_summary.json` 里的 `crc.empirical_risk` 和 `crc.risk_bound`，损失为 `1{accept and wrong}`，分母是 calibration 总数。
2. `pool_summary.accept_error_rate` 是 accept 子集里的条件错误率，只是诊断指标，不是 CRC 证明里直接受 `alpha` 约束的量。
3. 当前默认流程会用同一个 `D_cal` 参与每轮 CRC、DBDS defer 集识别和停止判断；这足以跑工程实验，但严格最终 CRC 保证需要额外保留一个没有参与选择、温度、停止判断的最终校准集。
4. `--budget` 是 DBDS defer 样本预算；默认 `--easy_anchor_ratio 0.1` 会额外加入 anchor，所以 teacher 训练标签数通常是 `budget + floor(0.1 * budget)`。

## 结果文件

- `round_*/predict_usage.json`：student 调用行数和估算 token。
- `round_*/calibrate_usage.json`：CRC 校准行数、teacher/groundtruth 计数。
- `round_*/select_usage.json`：DBDS 选择数、embedding 使用量、teacher/groundtruth 计数。
- `round_*/train_usage.json`：训练 step、训练样本数、估算训练 token。
- `cgsd_summary.json`：最终 round、阈值、teacher 调用摘要。
- `deployment_decisions.jsonl`：最终部署时 student accept、teacher defer、训练标签复用决策。
