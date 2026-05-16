# CGSD 实验执行总览

这个目录把 `CGSD_experiment_plan.md` 里的实验拆成可执行说明。当前代码的 CGSD 主链路已经是 CLI 驱动：每个 stage 只读本 stage 的 CLI 参数和输入 artifact，不依赖 `cgsd_config.json` 或 `cgsd_state.json`。

代码和文档一致性审计见 [DOC_CODE_AUDIT.md](/teamspace/studios/this_studio/LLM_layer_test/experiments/DOC_CODE_AUDIT.md)。

## 当前代码支持情况

| 实验 | 当前代码状态 | 主要缺口 |
| --- | --- | --- |
| 01 核心有效性 | 可跑完整 CGSD 闭环；`experiments/bin/` 提供薄脚本 | FEVER embedding 还需补齐 |
| 02 数据选择对比 | DBDS 和 Random/Uncertainty/k-Center/Defer-Random 训练行都可脚本生成 | paired t-test 仍需外部统计 |
| 03 标注预算曲线 | DBDS 预算扫描和 CSV 聚合可跑 | 画图脚本未实现 |
| 04 消融 | delta、band 比例、轮次、LoRA rank、teacher beta、easy anchor 可跑 | 自动多配置调度未实现 |
| 05 端到端对比 | CGSD 和本地 LoRA 可跑 | Full GPT-5、4B zero-shot baseline、二次分流需要外部结果 |
| 06 CRC 保证验证 | 多 seed、多 alpha 可跑，并可汇总 violation rate | 严格保证需要独立最终校准集 |
| 07 三角关系 | 可由 03 和 06 的结果汇总，支持成本常量估算 | 图表脚本未实现 |
| 08 LROBench | 合并版可 smoke，per-query 输入可拆分 | cross-query LoRA 参数平均/迁移未实现 |

## 后续测试路线

当前已经实跑通过一条 LROBench smoke：`lrobench/exp1_seed1`，链路为 `round0 eval -> round0 select -> round1 train/eval -> finalize -> collect`。后续不要直接上全量大 bash，先按下面顺序把每类实验的最小闭环跑通，再扩 seed、预算和数据集。

### 0. 环境和数据冒烟

目的：确认输入、embedding、模型路径、缓存策略都对。

```bash
export DATASET=lrobench
export RUN_NAME=exp1_seed1
export MODEL=../model/qwen3-0.6b
export DIM=2560
export SEED=1
export ALPHA=0.07
export TEMP=15
export CACHE_POLICY=reuse
```

检查项：

1. `experiments/inputs/lrobench/data.jsonl` 有 `1162` 行。
2. `experiments/inputs/lrobench/embeddings.npy` shape 是 `(1162, 2560)`。
3. `MODEL` 指向真实本地 checkpoint；当前机器上是 `../model/qwen3-0.6b`。
4. FEVER 只检查数据，不跑正式 CGSD；还缺 `experiments/inputs/fever/embeddings.npy`。

### 1. LROBench 单配置闭环

目的：证明主代码链路可执行。这个阶段只跑一个 seed、一个 budget。

```bash
experiments/bin/cgsd_round0_select.sh

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python scripts/cgsd_train_round.py \
  --output_dir experiments/runs/lrobench/exp1_seed1 \
  --round_index 1 \
  --model_path "$MODEL" \
  --data_path experiments/inputs/lrobench/data.jsonl \
  --split_ids_path experiments/runs/lrobench/exp1_seed1/cgsd_split_ids.json \
  --train_rows_path experiments/runs/lrobench/exp1_seed1/cgsd_train_rows.jsonl \
  --lora_r 1 \
  --lora_target_modules qv \
  --lora_layer_scope all \
  --epochs 3 \
  --lr 2e-4 \
  --batch_size 1 \
  --gradient_accumulation_steps 16 \
  --eval_batch_size 4 \
  --max_length 512 \
  --cache_policy reuse

ROUND=1 experiments/bin/cgsd_eval_round.sh
ROUND=1 experiments/bin/cgsd_finalize.sh
```

L4 上不要用训练默认 `batch_size=16`，会 OOM；用上面的显存参数作为当前默认测试口径。完成标准：

1. `round_0/round_summary.json` 存在。
2. `round_0/selection_summary.json` 存在，`selected_budget` 等于设定 budget。
3. `round_1/model/model_config.json` 和 adapter 存在。
4. `round_1/round_summary.json` 存在。
5. `cgsd_summary.json` 和 `deployment_decisions.jsonl` 存在。

汇总：

```bash
experiments/bin/cgsd_collect_results.py \
  --runs experiments/runs/lrobench/exp1_seed1 \
  --output_csv experiments/runs/lrobench/exp1_seed1/results_summary.csv \
  --output_jsonl experiments/runs/lrobench/exp1_seed1/results_summary.jsonl \
  --crc_summary_csv experiments/runs/lrobench/exp1_seed1/crc_summary.csv
```

### 2. 实验 2 baseline 最小对比

目的：确认 Random、Uncertainty、k-Center、Defer-Random 的选样脚本和训练闭环可用。

先固定同一个 round0 来源：

```bash
export SOURCE_OUT=experiments/runs/lrobench/exp1_seed1
export BUDGET=250
```

每个策略单独建 run，不要混写：

```bash
RUN_NAME=exp2_random_seed1 STRATEGY=random experiments/bin/cgsd_baseline_rows.sh
RUN_NAME=exp2_uncertainty_seed1 STRATEGY=uncertainty experiments/bin/cgsd_baseline_rows.sh
RUN_NAME=exp2_kcenter_seed1 STRATEGY=k-center experiments/bin/cgsd_baseline_rows.sh
RUN_NAME=exp2_defer_random_seed1 STRATEGY=defer-random experiments/bin/cgsd_baseline_rows.sh
```

对每个 baseline run 都跑同样的 `round1 train -> eval -> finalize`。完成标准：

1. 每个 run 有自己的 `cgsd_train_rows.jsonl`。
2. 每个 run 有 `round_1/round_summary.json`。
3. 每个 run 有 `cgsd_summary.json`。
4. 汇总表能同时收集 DBDS 和 baseline：

```bash
experiments/bin/cgsd_collect_results.py \
  --runs 'experiments/runs/lrobench/exp1_seed1' 'experiments/runs/lrobench/exp2_*_seed1' \
  --output_csv experiments/runs/lrobench/exp2_selection_comparison_seed1.csv
```

注意：paired t-test 还没有内置，先产出每个 seed 的结果表，后续再做统计。

### 3. 多 seed 主实验

目的：得到实验 1 的稳定结果。建议先跑 `SEED=1,2,3`，稳定后再扩到 5 个 seed。

每个 seed 单独 run：

```bash
for SEED in 1 2 3; do
  export SEED
  export RUN_NAME=exp1_seed${SEED}
  experiments/bin/cgsd_round0_select.sh
  # train/eval/finalize 使用第 1 节的 L4 训练参数
done
```

完成标准：

1. 每个 seed 都有 `cgsd_summary.json`。
2. `cgsd_collect_results.py --runs 'experiments/runs/lrobench/exp1_seed*'` 能产出 CSV。
3. 报告均值/方差时使用 run-level CSV，不手抄单个 JSON。

### 4. 预算曲线和消融

目的：确认 `budget/delta/band ratio/teacher beta/easy anchor/LoRA rank` 的影响。每个配置必须使用独立 `RUN_NAME`。

预算曲线先跑：

```bash
for BUDGET in 0 50 100 250 500; do
  export RUN_NAME=exp3_budget${BUDGET}_seed1
  export BUDGET
  if [ "$BUDGET" = "0" ]; then
    experiments/bin/cgsd_round0_eval.sh
    ROUND=0 experiments/bin/cgsd_finalize.sh
  else
    experiments/bin/cgsd_round0_select.sh
    # train/eval/finalize 使用第 1 节的 L4 训练参数
  fi
done
```

消融先跑最小集合：

```bash
BAND_RATIOS=1,0,0 RUN_NAME=exp4_band_B_seed1 experiments/bin/cgsd_round0_select.sh
BAND_RATIOS=0,1,0 RUN_NAME=exp4_band_M_seed1 experiments/bin/cgsd_round0_select.sh
BAND_RATIOS=0,0,1 RUN_NAME=exp4_band_F_seed1 experiments/bin/cgsd_round0_select.sh
EASY_ANCHOR_RATIO=0 RUN_NAME=exp4_no_anchor_seed1 experiments/bin/cgsd_round0_select.sh
```

完成标准：

1. 每个配置都有独立 `selection_summary.json`，能看到实际 `effective_band_ratios` 或 anchor 设置。
2. 训练后都有 `round_1/round_summary.json` 和 `cgsd_summary.json`。
3. 用 `cgsd_collect_results.py` 汇总，而不是人工拼表。

### 5. CRC 多 alpha 验证

目的：检查不同 `alpha` 下 `wrong_accept_rate` 和 `crc.risk_bound` 是否按预期变化。

```bash
for ALPHA in 0.03 0.05 0.07 0.10; do
  export ALPHA
  export RUN_NAME=exp6_alpha${ALPHA}_seed1
  experiments/bin/cgsd_round0_eval.sh
done

experiments/bin/cgsd_collect_results.py \
  --runs 'experiments/runs/lrobench/exp6_alpha*_seed1' \
  --output_csv experiments/runs/lrobench/exp6_alpha_seed1.csv \
  --crc_summary_csv experiments/runs/lrobench/exp6_crc_seed1.csv
```

完成标准：

1. 每个 alpha 有 `round_0/round_summary.json`。
2. `crc_summary.csv` 里有 `violation_rate`。
3. 若要写 theorem-level final guarantee，必须另准备独立最终校准集；默认流程只能作为工程验证。

### 6. LROBench per-query 测试

目的：从 pooled smoke 过渡到严格 per-query 结果。

```bash
experiments/bin/cgsd_split_lrobench_inputs.py \
  --data_path experiments/inputs/lrobench/data.jsonl \
  --embeddings_path experiments/inputs/lrobench/embeddings.npy \
  --output_root experiments/inputs \
  --prefix lrobench
```

先挑 1 个 query 跑通：

```bash
export DATASET=lrobench_select100
export RUN_NAME=select100_seed1
export N_CALIBRATION=10
export BUDGET=10
export ANCHOR_COUNT=0
experiments/bin/cgsd_round0_select.sh
```

完成标准：

1. per-query 输入目录有 `data.jsonl`、`embeddings.npy`、`embeddings.ids.jsonl`。
2. 小样本下 `n_calibration + budget + anchor_count` 小于该 query 样本数。
3. 每个 query 单独 run；不同 query 的 split、embedding、LoRA checkpoint 不混用。

### 7. FEVER 正式运行前置

目的：补齐 FEVER 的唯一关键缺口。

完成 FEVER embedding 后，必须满足：

1. `experiments/inputs/fever/data.jsonl` 行数和 `embeddings.npy` 第一维一致。
2. `embeddings.ids.jsonl` 覆盖所有 FEVER `id`。
3. embedding 文本口径是固定 query 加上 `Claim/Evidence` document，不是只 embed claim 或只 embed evidence。
4. 先按第 1 节跑 `fever/exp1_seed1` smoke，再扩多 seed 和 baseline。

## 实验工作区约定

实验相关入口都放在 `experiments/` 下，便于查找：

```text
experiments/
  inputs/<dataset>/     # 实验输入入口，通常是指向本地数据资产的软链接
  runs/<dataset>/<run>/ # 每次实验的所有输出 artifact
  bin/                  # 薄 bash wrapper，只封装稳定 stage
```

默认 wrapper 变量：

```bash
export DATASET=lrobench
export RUN_NAME=exp1_seed1
export MODEL=model/qwen3-0.6b
export DIM=2560
export SEED=1
export ALPHA=0.07
export TEMP=15
```

默认输入输出会解析为：

```bash
DATA=experiments/inputs/$DATASET/data.jsonl
EMB=experiments/inputs/$DATASET/embeddings.npy
OUT=experiments/runs/$DATASET/$RUN_NAME
```

如果有真实 teacher 文件，放到 `experiments/inputs/<dataset>/teacher.jsonl` 后显式传：

```bash
export TEACHER=experiments/inputs/$DATASET/teacher.jsonl
```

当前已建立的输入入口：

```text
experiments/inputs/lrobench/data.jsonl
experiments/inputs/lrobench/embeddings.npy
experiments/inputs/fever/data.jsonl
```

FEVER 还缺 `experiments/inputs/fever/embeddings.npy`，所以 FEVER 正式 CGSD 需要先生成全量 embedding。

## 薄脚本

这些脚本只把稳定相邻 stage 包起来，不隐藏关键 checkpoint：

| 脚本 | 做什么 |
| --- | --- |
| `experiments/bin/cgsd_round0_eval.sh` | `prepare -> predict round0 -> calibrate round0`，不选训练样本 |
| `experiments/bin/cgsd_round0_select.sh` | `prepare -> predict round0 -> calibrate round0 -> select round0` |
| `experiments/bin/cgsd_train_round.sh` | 训练一个 LoRA round，例如 `ROUND=1` |
| `experiments/bin/cgsd_eval_round.sh` | `predict roundN -> calibrate roundN` |
| `experiments/bin/cgsd_select_round.sh` | 从已评估的 roundN 继续 selection |
| `experiments/bin/cgsd_finalize.sh` | 固定某一轮生成部署决策 |
| `experiments/bin/cgsd_baseline_rows.sh` | 为实验 2 生成 baseline `cgsd_train_rows.jsonl` |
| `experiments/bin/cgsd_collect_results.py` | 汇总 run 的 round summary、usage、CRC violation 和可选成本估算 |
| `experiments/bin/cgsd_split_lrobench_inputs.py` | 把合并版 LROBench 数据和 embedding 拆成 per-query 输入目录 |
| `experiments/bin/cgsd_run_exp1_default_3rounds.sh` | 可选 overnight wrapper，按 250/150/100 跑完实验 1 默认三轮 |

建议日常用前 5 个小脚本逐步跑；最后一个大脚本只在配置已经确认后使用。

## 统一数据前置

1. 把实验数据转换为 JSONL，每行至少包含 `id`、`query`、`document`、`groundtruth`。
2. `id` 必须稳定且全局唯一；重复运行、embedding、teacher 文件、split 文件都用这个 `id` 对齐。
3. `groundtruth` 必须能归一成二分类 `1/0`；`yes/true/1` 会映射为 `1`，`no/false/0` 会映射为 `0`。
4. 如果原数据不是这些字段名，优先在 CLI 里传 `--query_field`、`--document_field`、`--label_field`，不要改代码。
5. FEVER 可预处理成：`query=固定任务句`，`document=Claim + Evidence 完整载荷`，`groundtruth=支持/相关为 1，不支持/不相关为 0`。不要把 FEVER claim 单独塞进 `query`，否则和后续统一的 `query + document` completion 口径不一致。
6. LROBench 可预处理成：每个 query 一个 JSONL，`query=筛选条件`，`document=行文本或序列化后的 row`，`groundtruth=该 row 是否满足筛选条件`。

最小数据行示例：

```json
{"id":"fever_000001","query":"Does the evidence support the claim?","document":"Claim:\nThe claim text.\n\nEvidence:\nCandidate evidence text.","groundtruth":1}
```

## Embedding 输入格式

1. `cgsd_prepare.py` 和 `cgsd_select.py` 都需要 `--embeddings_path`。
2. embedding 必须覆盖数据文件里的所有 `id`；当前本地 Qwen3 embedding 是 `2560` 维，运行时传 `DIM=2560` 或 `--embedding_dim 2560`。
3. 正式实验默认读取预计算 embedding；生成 embedding 时要保证文本口径和 `query + document` 完全一致。
4. 支持四种格式：

JSONL：

```json
{"id":"fever_000001","embedding":[0.12,-0.03,0.44]}
```

JSON object：

```json
{"fever_000001":[0.12,-0.03,0.44]}
```

NPZ：每个 key 是样本 `id`，value 是一维向量。

NPY：二维矩阵，旁边必须有同名 id sidecar，例如 `embeddings.npy` 搭配 `embeddings.ids.jsonl`：

```json
{"id":"fever_000001"}
```

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

## 当前数据路径

当前仓库已经放入两份实验数据：

```bash
export FEVER_DATA=datasets/fever_cgsd.jsonl
export LRO_DATA=datasets/lrobench_cgsd.jsonl
```

FEVER JSONL schema：

```json
{"id":"fever_evidence_75397","query":"Does the evidence support the claim?","document":"Claim:\n...\n\nEvidence:\n...","groundtruth":1}
```

LROBench JSONL schema：

```json
{"id":"select100_row000:select100","query":"The provided birth date is later than the birth date of Alex Albon.","document":"forename: Jaime\nsurname: Alguersuari\ndob: 1990-03-23","groundtruth":0}
```

两份数据都已经统一成当前 CGSD 主链路默认读取的字段：`id/query/document/groundtruth`。FEVER 的 `query` 是固定任务句，`document` 中包含 Claim 和 Evidence；LROBench 的 `query` 是筛选条件，`document` 是 row 文本。

当前仓库已经放入 LROBench embedding：

```bash
export LRO_EMB=datasets/lrobench_cgsd.embeddings.npy
```

LROBench 运行时传：

```bash
--data_path "$LRO_DATA" --embeddings_path "$LRO_EMB" --embedding_dim 2560
```

当前仓库还没有放入 FEVER embedding。正式 FEVER CGSD 结果需要生成覆盖 `datasets/fever_cgsd.jsonl` 全量 `id` 的 pair embedding，生成文本口径必须和 JSONL 一致：固定 query 加上包含 Claim/Evidence 的 document。

在实验操作中优先使用 `experiments/inputs/...` 路径；上面的 `datasets/...` 是本地数据资产位置。

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
