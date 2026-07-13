# 方法说明

## 1. 这份代码在做什么

这个项目实现的是一条面向 LLM 二分类任务的“选择式训练”管线。它的目标不是单纯训练一个分类器，而是先让一个基础大语言模型作为 student 对样本做二分类判断，再用校准集估计哪些样本是模型比较不确定或更可能出错的区域，然后从候选池里挑出更有训练价值的样本，最后用这些样本对基础模型做 LoRA 微调。

整条链路可以理解成：

```text
原始数据
-> 统一成 query-document 二分类格式
-> 切分 guide / final / pool
-> 基座 LLM 做 round0 推理
-> 用 guide 校准 CRC 阈值
-> 给 pool 样本打 accept / defer 标记
-> 用 random、PCSS 或 CRC error-mass 选训练样本
-> 用选出的样本训练 LoRA
-> 用 LoRA 后的模型继续推理和评估
```

这里的核心思想是：利用模型自己的预测 margin、校准集上的错误分布，以及 accept/defer 区域的信息，把有限训练预算放到更可能改善模型的位置。

## 2. 任务形式和标签协议

项目把所有任务统一成 query-document 二分类问题。每条样本包含：

```json
{"id":"sample_1","query":"判断条件或问题","document":"待判断文本","groundtruth":1}
```

标签协议固定为：

- `1`：document 满足 query。
- `0`：document 不满足 query。

模型也被要求只输出 `1` 或 `0`。推理时不把自然语言答案作为主要结果，而是读取模型在第一个答案 token 上对 `1` 和 `0` 的 logit 或 logprob。核心分数定义为：

```text
score = logit_or_logprob("1") - logit_or_logprob("0")
```

因此：

- `score > 0` 时，预测为 `1`。
- `score <= 0` 时，预测为 `0`。
- `abs(score)` 越大，说明模型越偏向某一类。
- `abs(score)` 越小，说明模型越不确定。

这个协议贯穿训练、推理、CRC 校准、选样和评估，避免不同阶段对标签含义或模型输出解析方式产生漂移。

## 3. 数据切分：guide、final、pool

数据准备阶段会把全集切成三个互不重叠的集合：

- `guide`：校准集。主要用于 CRC 阈值校准、估计标签比例、估计错误分布。默认大小是 `1000`。
- `final`：最终保留集。用于独立评估或认证，不参与训练样本选择。默认大小是 `0`，具体实验中也可以设为 `200` 或其他固定规模。
- `pool`：候选池。后续选训练样本都从这里抽取。

默认切分方式是按真实标签做分层抽样，也就是 `split_strategy=stratified`。这样 guide 和 final 的正负样本比例会尽量贴近剩余数据的整体比例。也可以使用完全随机切分 `split_strategy=random`。

关键超参：

| 超参 | 默认值 | 含义 |
| --- | --- | --- |
| `n_guide` | `1000` | guide 校准集大小 |
| `n_final` | `0` | final 保留集大小 |
| `split_strategy` | `stratified` | 是否按标签分层切分 |
| `seed` | `42` | 切分随机种子 |

## 4. Round0 student 推理

round0 是基础模型还没有经过 LoRA 微调时的推理结果。代码通过 OpenAI-compatible vLLM 接口请求模型，并使用 raw completion，而不是 chat completion。原因是代码已经手写了模型需要的 chat 标记和 no-thinking block，如果再交给 chat completion 自动套模板，可能导致模型第一个输出 token 位置发生偏移，进而影响 `1/0` logprob 的读取。

推理 prompt 的语义是：

```text
你是一个精确的二分类器，只能回答 "1" 或 "0"。

Query: ...
Document: ...
如果 document 满足 query，返回 "1"；否则返回 "0"。
```

推理输出会记录：

- `score`：`1` 相对 `0` 的 logprob margin。
- `prediction`：由 `score > 0` 得到的二分类预测。
- `probability`：对 margin 做 sigmoid 后得到的正类概率。
- `zero_logit` / `one_logit`：实际记录的是 `0` 和 `1` 的 logprob 或 logit。
- `generated_text`：规范化后的 `0/1` 输出。

关键超参：

| 超参 | 默认值 | 含义 |
| --- | --- | --- |
| `model_path` | `model/qwen3-0.6b` | 基础模型路径 |
| `temperature` | `0.0` | 生成温度，默认确定性输出 |
| `max_tokens` | `1` | 只生成一个答案 token |
| `top_logprobs` | `20` | 取候选 token logprob，保证能读到 `0/1` |
| `parallel_requests` | `1024` | 并发请求数 |
| `request_retries` | `3` | 单样本失败重试次数 |
| `timeout` | `180` | 请求和服务等待超时 |
| `max_model_len` | `40960` | vLLM 最大上下文长度 |
| `max_num_seqs` | `4096` | vLLM 最大并发序列数 |
| `max_num_batched_tokens` | `524288` | vLLM 最大 batch token 数 |
| `gpu_memory_utilization` | `0.98` | vLLM 显存利用上限 |
| `enforce_eager` | `True` | vLLM eager 执行开关 |

实际实验中，为了显存稳定或吞吐控制，也常把 `parallel_requests`、`max_num_seqs`、`gpu_memory_utilization` 调低。

## 5. CRC：用校准集决定 accept / defer

CRC 是这条管线的核心判断机制之一。它的作用是根据 guide 集上的 student 预测表现，学习一个阈值 `lambda_hat`，然后把样本分成：

- `accept`：模型足够自信，可以直接接受当前预测。
- `defer`：模型不够自信，或者落在风险较高区域，需要后续重点处理。

代码先把原始 `score` 转成一个无符号 routing score：

```text
routing_score = sigmoid(abs(score) / temperature)
```

这个值范围大约在 `[0.5, 1.0]`：

- 接近 `0.5`：模型对 `0/1` 非常犹豫。
- 接近 `1.0`：模型对某一类非常自信。

CRC 会在一组候选阈值上搜索：

```text
lambda_grid = 0.50, 0.51, 0.52, ..., 1.00
```

对于每个阈值 `lambda`：

```text
accept = routing_score >= lambda
defer = routing_score < lambda
```

然后在 guide 集上计算一个风险上界：

```text
empirical_risk = wrong_accept_count / guide_count
risk_bound = guide_count / (guide_count + 1) * empirical_risk + 1 / (guide_count + 1)
```

注意这里的分母是整个 guide 集大小，不是 accept 样本数。也就是说，它控制的是“被接受且预测错误”的总体质量，而不是 accept 子集内部错误率。

算法选择第一个满足下面条件的阈值：

```text
risk_bound <= alpha
```

如果所有 `0.50` 到 `1.00` 的阈值都不满足，就设 `lambda_hat=1.01`，这会导致所有样本都进入 defer。

关键超参：

| 超参 | 默认值 | 含义 |
| --- | --- | --- |
| `alpha` | `0.1` | CRC 风险容忍水平 |
| `temperature` | `15.0` | 把 margin 映射成 routing score 的温度 |
| `lambda_grid` | `0.50` 到 `1.00`，步长 `0.01` | CRC 阈值搜索网格 |
| `selection_budget` | `0` | 可选，仅用于在 CRC summary 里额外预览 PCSS 预算 |

CRC 之后，每条 pool 样本会额外带上：

- `routing_score`
- `routing_temperature`
- `crc_decision`
- `defer`
- `decision_threshold`
- `tau_crc`

其中 `tau_crc` 是把 `lambda_hat` 映射回原始 margin 空间后的阈值：

```text
tau_crc = temperature * log(lambda_hat / (1 - lambda_hat))
```

## 6. 训练样本选择方法

CRC 之后，代码提供三种训练数据选择方式：random、PCSS、CRC error-mass。

核心选择流程的伪代码如下。它先完成 CRC 校准和 pool 路由，然后根据 `method` 切换到不同的选样策略：

```text
Input:
  guide_predictions        # guide 集上的 student 预测，含 id/label/score/prediction
  pool_predictions         # pool 集上的 student 预测，含 id/label/score/prediction
  budget                   # 本轮训练样本预算
  method                   # random / pcss / crc-error-mass
  alpha                    # CRC 风险水平
  temperature              # routing score 温度
  seed                     # 随机种子
  blocked_ids              # 已进入累计训练集的样本 id，避免重复选择

Step 1: 计算 guide 上的 routing score
  for row in guide_predictions:
      row.routing_score = sigmoid(abs(row.score) / temperature)

Step 2: CRC 阈值校准
  lambda_hat = 1.01
  for lambda in [0.50, 0.51, ..., 1.00]:
      accept_rows = []
      wrong_accept_count = 0

      for row in guide_predictions:
          if row.routing_score >= lambda:
              accept_rows.append(row)
              if row.prediction != row.label:
                  wrong_accept_count += 1

      empirical_risk = wrong_accept_count / len(guide_predictions)
      risk_bound = len(guide_predictions) / (len(guide_predictions) + 1) * empirical_risk
                 + 1 / (len(guide_predictions) + 1)

      if risk_bound <= alpha:
          lambda_hat = lambda
          break

Step 3: 给 guide 和 pool 打 accept/defer 标记
  routed_guide = []
  for row in guide_predictions:
      if lambda_hat > 1.0:
          row.decision = "defer"
      else if row.routing_score >= lambda_hat:
          row.decision = "accept"
      else:
          row.decision = "defer"
      routed_guide.append(row)

  routed_pool = []
  for row in pool_predictions:
      row.routing_score = sigmoid(abs(row.score) / temperature)

      if lambda_hat > 1.0:
          row.decision = "defer"
      else if row.routing_score >= lambda_hat:
          row.decision = "accept"
      else:
          row.decision = "defer"
      routed_pool.append(row)

Step 4: 移除已经训练过的样本
  candidate_pool = []
  for row in routed_pool:
      if row.id not in blocked_ids:
          candidate_pool.append(row)

Step 5: 按 method 选择训练样本
  if method == "random":
      selected = RANDOM_SELECT(candidate_pool, budget, seed)

  if method == "pcss":
      selected = PCSS_SELECT(
          routed_guide,
          candidate_pool,
          budget,
          temperature
      )

  if method == "crc-error-mass":
      selected = CRC_ERROR_MASS_SELECT(
          routed_guide,
          candidate_pool,
          budget,
          seed,
          accept_strategy,
          defer_strategy
      )

Output:
  selected_train_rows      # 本轮新增训练样本
  selection_summary        # 预算拆分、候选数量、实际选择数量、是否 shortfall
```

random 的伪代码最直接：

```text
function RANDOM_SELECT(candidate_pool, budget, seed):
    ids = unique ids from candidate_pool
    shuffle(ids, seed)
    return first min(budget, len(ids)) rows
```

PCSS 的伪代码如下：

```text
function PCSS_SELECT(guide_rows, pool_rows, budget, temperature):
    # 1. 用 guide 的真实标签估计目标正类比例
    guide_label1_count = count(row.label == 1 for row in guide_rows)
    p_hat_1 = guide_label1_count / len(guide_rows)

    # 2. 先按目标标签比例拆预算
    target_label1_budget = round_half_up(budget * p_hat_1)
    target_label0_budget = budget - target_label1_budget

    # 3. pool 中没有真实标签可用时，用 student prediction 当 proxy label
    label1_candidates = [row for row in pool_rows if row.prediction == 1]
    label0_candidates = [row for row in pool_rows if row.prediction == 0]

    # 4. 如果某一层候选不足，把剩余预算补给另一层
    B_label1 = min(target_label1_budget, len(label1_candidates))
    B_label0 = min(target_label0_budget, len(label0_candidates))
    remaining = min(budget, len(pool_rows)) - B_label1 - B_label0

    if remaining > 0:
        add_to_label0 = min(remaining, len(label0_candidates) - B_label0)
        B_label0 += add_to_label0
        remaining -= add_to_label0

    if remaining > 0:
        add_to_label1 = min(remaining, len(label1_candidates) - B_label1)
        B_label1 += add_to_label1

    # 5. 每个 proxy label 层内部选最不确定样本
    sort label0_candidates by (routing_score ascending, id ascending)
    sort label1_candidates by (routing_score ascending, id ascending)

    selected_label0 = first B_label0 rows from label0_candidates
    selected_label1 = first B_label1 rows from label1_candidates

    return selected_label1 + selected_label0
```

CRC error-mass 的伪代码如下：

```text
function CRC_ERROR_MASS_SELECT(
    guide_rows,
    pool_rows,
    budget,
    seed,
    accept_strategy,
    defer_strategy
):
    # 1. 统计 guide/pool 的 defer 比例和错误比例
    guide_defer_rows = [row for row in guide_rows if row.decision == "defer"]
    pool_defer_rows = [row for row in pool_rows if row.decision == "defer"]
    pool_accept_rows = [row for row in pool_rows if row.decision == "accept"]

    r_U = len(pool_defer_rows) / len(pool_rows)
    r_C = len(guide_defer_rows) / len(guide_rows)

    guide_error_count = count(row.prediction != row.label for row in guide_rows)
    guide_defer_error_count = count(row.prediction != row.label for row in guide_defer_rows)

    e_all = guide_error_count / len(guide_rows)

    # 2. 估计 defer 区域是否浓缩了错误
    if len(guide_defer_rows) == 0 or guide_error_count == 0 or e_all <= 0:
        e_defer = 0.0
        c_crc = 1.0
        eta_crc = 0.0
    else:
        e_defer = guide_defer_error_count / len(guide_defer_rows)
        c_crc = e_defer / e_all

        if c_crc <= 1.0 or r_C <= 0.0 or r_C >= 1.0:
            eta_crc = 0.0
        else:
            eta_crc = clamp(log(c_crc) / log(1 / r_C), 0.0, 1.0)

    # 3. 根据错误浓缩度分配 accept/defer 预算
    s_defer = clamp(r_U + eta_crc * (1 - r_U)^2, 0.0, 1.0)
    s_accept = 1.0 - s_defer

    B_defer = round_half_up(budget * s_defer)
    B_accept = budget - B_defer

    B_defer = min(B_defer, len(pool_defer_rows))
    B_accept = min(B_accept, len(pool_accept_rows))

    # 4. 两侧内部选择。默认 random，也可以 high-confidence
    selected_accept = SELECT_SIDE(pool_accept_rows, B_accept, seed, accept_strategy)
    selected_defer = SELECT_SIDE(pool_defer_rows, B_defer, seed + 1, defer_strategy)

    return selected_accept + selected_defer

function SELECT_SIDE(rows, k, seed, strategy):
    if strategy == "random":
        shuffle(rows, seed)
        return first k rows

    if strategy == "high-confidence":
        sort rows by (routing_score descending, id ascending)
        return first k rows
```

### 6.1 random

random 是最简单的 baseline。它从 pool 候选池中均匀随机抽取 `budget` 条样本，不主动使用 accept/defer 信息，也不控制标签比例。

它仍然读取 CRC 后的 pool 文件，因为这个文件里已经包含完整 metadata、score、prediction 和 defer 标记，但 random 自身不根据这些字段排序。

关键超参：

| 超参 | 默认值 | 含义 |
| --- | --- | --- |
| `budget` | 必填 | 本轮要选多少训练样本 |
| `seed` | `42` | 随机种子，实际会加上 `round_index` |

### 6.2 PCSS（主方法）

PCSS 可以理解为 Prior-Corrective Stratified Selection，即“先校正标签先验，再在层内挑不确定样本”。

它分两步：

第一步，用 guide 的真实标签估计目标正类比例：

```text
p_hat_1 = guide_label1_count / guide_count
```

然后把训练预算拆成 proxy label 0 和 proxy label 1 两部分：

```text
target_label1_budget = round_half_up(budget * p_hat_1)
target_label0_budget = budget - target_label1_budget
```

这里的 proxy label 不是人工标签，而是 student 当前预测：

```text
proxy_label = prediction
```

第二步，在 pool 中按 proxy label 分层。每一层内部按 `routing_score` 从小到大排序，优先选择模型最不确定的样本：

```text
越小的 routing_score -> 越接近决策边界 -> 越优先被选
```

因此 PCSS 同时做两件事：

1. 用 guide 的真实标签比例约束训练集的整体类别结构，避免 student 自身预测偏置导致训练集类别坍缩。
2. 在每个 proxy label 层内部优先选择模型不确定样本，把预算放到决策边界附近。

如果某个 proxy label 层候选样本不够，代码会把剩余预算补到另一层，尽量填满可行预算。

关键超参：

| 超参 | 默认值 | 含义 |
| --- | --- | --- |
| `budget` | 必填 | 总训练样本预算 |
| `seed` | `42` | 主要用于去重和稳定流程，PCSS 排序本身是确定性的 |
| `temperature` | 来自 CRC summary | 决定 routing score 的尺度 |
| `alpha` | 来自 CRC summary | 记录在 PCSS plan 中，保持与 CRC 口径一致 |
| `lambda_hat` | 来自 CRC summary | 记录在 PCSS plan 中，保持与 CRC 口径一致 |

### 6.3 CRC error-mass（对照方法）

CRC error-mass 的目标是根据 guide 上的错误分布，估计错误是否集中在 defer 区域。如果 defer 区域确实聚集了更多错误，就给 defer 侧更多训练预算。

代码会统计：

```text
r_U = pool_defer_count / pool_total
r_C = guide_defer_count / guide_count
e_all = guide_error_count / guide_count
e_defer = guide_defer_error_count / guide_defer_count
```

其中：

- `r_U`：pool 中 defer 区域比例。
- `r_C`：guide 中 defer 区域比例。
- `e_all`：guide 上整体错误率。
- `e_defer`：guide 的 defer 区域错误率。

然后计算错误浓缩度：

```text
c_crc = e_defer / e_all
```

如果 defer 区域错误率比整体更高，`c_crc` 会大于 1，说明 defer 区域确实更“脏”。接着代码把这个浓缩度转成一个预算偏移系数：

```text
eta_crc = log(c_crc) / log(1 / r_C)
```

并限制在 `[0, 1]`。如果 guide 中没有 defer 样本、没有错误，或者浓缩度不大于 1，则 `eta_crc=0`。

最终 defer 侧预算比例为：

```text
s_defer = r_U + eta_crc * (1 - r_U)^2
s_accept = 1 - s_defer
```

再得到预算：

```text
B_defer = round_half_up(budget * s_defer)
B_accept = budget - B_defer
```

直观理解是：

- 如果 defer 区域只是普通区域，预算按 pool 中 defer 占比来分。
- 如果 defer 区域明显聚集错误，就额外向 defer 侧倾斜。
- 倾斜幅度受 `eta_crc` 控制，并且不会超过总预算。

accept 和 defer 两侧内部支持两种选择策略：

- `random`：侧内随机抽样。
- `high-confidence`：侧内按 `routing_score` 从高到低选，也就是优先选模型最自信的样本。

默认两侧都是 `random`。

关键超参：

| 超参 | 默认值 | 含义 |
| --- | --- | --- |
| `budget` | 必填 | 总训练样本预算 |
| `accept_strategy` | `random` | accept 侧内部选择策略 |
| `defer_strategy` | `random` | defer 侧内部选择策略 |
| `seed` | `42` | 随机种子，defer 侧会使用 `seed + 1` |
| `temperature` | 来自 CRC summary | routing score 温度 |
| `alpha` | 来自 CRC summary | CRC 风险水平 |
| `lambda_hat` | 来自 CRC summary | CRC 校准得到的阈值 |

## 7. Teacher label 和训练标签

选出的训练行可以直接使用数据里的 `groundtruth`，也可以额外接入 teacher 文件。teacher 文件可以提供：

- `teacher_label`
- `teacher_confidence`
- `teacher_logit_margin`

如果提供了 teacher label，训练行的 `label` 和 `groundtruth` 会被 teacher label 覆盖。如果没有 teacher 文件，代码会把原始 `groundtruth` 当作离线实验里的 teacher 替代标签，并记录来源。

也就是说，这套管线既可以用于已有标注数据的实验，也可以扩展到“先向更强 teacher 请求标签，再用这些标签训练 student”的主动学习式流程。

## 8. LoRA 训练

训练阶段使用被选中的累计训练集，对基础 Causal LM 做 LoRA SFT。训练目标不是让模型生成长文本解释，而是只监督二分类答案 token：`1` 或 `0`。

训练样本会构造成 chat_binary 格式：

```text
system: 只能回答 1 或 0
user: Query + Document + 返回规则
assistant: 1<|im_end|> 或 0<|im_end|>
```

loss 只计算 assistant 答案部分，prompt 部分全部 mask 为 `-100`。这可以让训练直接优化模型在分类答案 token 上的概率，而不是浪费 loss 在复现输入 prompt 上。

模型结构方面：

- 基础模型默认是 `model/qwen3-0.6b`。
- 训练模式固定使用 LoRA。
- 默认 LoRA 目标模块是 attention + MLP。
- 基座模型权重冻结，只训练 LoRA 参数。

LoRA 可选目标模块组：

| 名称 | 模块 |
| --- | --- |
| `qv` | `q_proj`, `v_proj` |
| `qkvo` / `attention` | `q_proj`, `k_proj`, `v_proj`, `o_proj` |
| `mlp` | `gate_proj`, `up_proj`, `down_proj` |
| `attention_mlp` | attention 四个投影 + MLP 三个投影 |

LoRA 可选层范围：

| 名称 | 含义 |
| --- | --- |
| `last1` | 只训练最后一层 |
| `last4` | 训练最后四层 |
| `all` | 所有层都挂 LoRA |

训练关键默认超参：

| 超参 | 默认值 | 含义 |
| --- | --- | --- |
| `epochs` | `3` | 训练轮数 |
| `lr` | `2e-4` | 学习率 |
| `weight_decay` | `0.01` | AdamW 权重衰减 |
| `batch_size` | `4` | 单步 batch size |
| `gradient_accumulation_steps` | `4` | 梯度累积步数 |
| `max_grad_norm` | `1.0` | 梯度裁剪阈值 |
| `warmup_ratio` | `0.1` | warmup 占总优化步数比例 |
| `scheduler_type` | `cosine` | 训练入口固定使用 cosine scheduler |
| `max_length` | `512` | 训练最大序列长度 |
| `threshold` | `0.0` | 评估时 score 转 prediction 的阈值 |
| `torch_dtype` | `auto` | 模型加载 dtype |
| `tf32` | `True` | CUDA 上启用 TF32 |
| `num_workers` | `2` | DataLoader worker 数 |
| `prefetch_factor` | `2` | DataLoader 预取 |
| `pad_to_multiple_of` | `8` | padding 对齐 |
| `cache_tokenization` | `True` | 是否缓存 tokenization |
| `pin_memory` | `True` | CUDA 训练时 pin memory |
| `lora_r` | `1` | LoRA rank |
| `lora_alpha` | `16` | LoRA alpha |
| `lora_dropout` | `0.05` | LoRA dropout |
| `lora_target_modules` | `attention_mlp` | LoRA 目标模块组 |
| `lora_layer_scope` | `all` | LoRA 作用层范围 |
| `balance_train_classes` | `False` | 是否按类别平衡 loss 权重 |
| `seed` | `42` | 训练随机种子 |

代码里保留了 early stopping 参数，但当前训练入口不使用 guide 做 early stopping，也不使用 guide 做最佳 epoch 选择。guide 被当作保留校准集合，不参与训练和模型选择。

如果打开 `balance_train_classes`，代码会按训练集中 0/1 类别频数计算类别权重：

```text
class_weight[label] = total_count / (2 * count(label))
```

然后把类别权重映射到 `0/1` token 的 loss 权重上。

## 9. 本地推理和 vLLM 推理

LoRA 训练完成后，有两种推理方式。

第一种是本地 PyTorch 推理。它直接加载基础模型和 LoRA adapter，在 prompt 最后一个非 padding 位置读取下一 token logits，再计算：

```text
score = logit("1") - logit("0")
```

本地推理适合小规模验证或调试。

本地推理关键超参：

| 超参 | 默认值 | 含义 |
| --- | --- | --- |
| `max_length` | `2048` | 最大输入长度 |
| `batch_size` | `64` | batch size |
| `max_tokens_per_batch` | `8192` | 动态 batch token 上限 |
| `threshold` | `0.0` | 二分类阈值 |
| `split_name` | `all` | 可选 `all/guide/final/pool` |

第二种是 vLLM 推理。它可以加载 LoRA adapter 并通过 OpenAI-compatible completion 接口跑大规模评估。评估口径和 round0 一样，仍然只生成一个 token，并读取 `1/0` 的 logprob margin。

vLLM 推理适合大规模评估和复现实验结果。

## 10. Embedding 阶段

代码中保留了 embedding 构建和校验能力。当前 random、PCSS、CRC error-mass 三种主选样方法不强制依赖 embedding，但 embedding 可以用于：

- 验证数据覆盖。
- 支持后续扩展的相似度、邻域或 difficulty 方法。
- 为外部分析或可视化提供向量工件。

embedding 文本格式是：

```text
Query:
{query}

Document:
{document}
```

构建方式支持：

- `transformers`：本地加载 embedding 模型。
- `vllm`：用 vLLM pooling 模式生成 embedding。

如果 `mode=document`，每条样本直接作为一个文本输入 embedding 模型。如果 `mode=chunk`，document 会按句子切块，每个 chunk 单独 embedding，最后对多个 chunk 的向量做均值池化并归一化。

embedding 关键超参：

| 超参 | 默认值 | 含义 |
| --- | --- | --- |
| `backend` | `transformers` | embedding 后端 |
| `request_batch_size` | `16` | 请求 batch size |
| `flush_rows` | `256` | 写入磁盘的行批量 |
| `max_length` | `4096` | embedding 模型最大长度 |
| `torch_dtype` | `bfloat16` | embedding 模型 dtype |
| `tensor_parallel_size` | `1` | vLLM tensor parallel |
| `gpu_memory_utilization` | `0.92` | vLLM 显存利用上限 |
| `mode` | `document` | 整文档或 chunk 模式 |
| `target_chars` | `3000` | chunk 目标字符数 |
| `overlap_chars` | `300` | chunk 重叠字符数 |

embedding 输出会写出：

- `embeddings.npy`
- `embeddings.ids.jsonl`
- `embeddings.meta.json`

并支持断点续跑：如果已经写了一部分 id sidecar，下次会从已完成前缀后继续。

## 11. 缓存和复跑策略

大部分 stage 都支持统一的缓存策略：

| 策略 | 含义 |
| --- | --- |
| `reuse` | 默认策略。输出完整存在时直接复用 |
| `overwrite` | 忽略已有结果，重新生成 |
| `fail` | 如果输出已存在就报错 |

如果某个 stage 的部分输出存在、部分输出缺失，默认会认为这是不完整缓存并报错，要求显式 overwrite 或清理后重跑。vLLM 推理额外支持 partial 文件，用于长时间推理中断后的断点恢复。

## 12. 输出工件

一次标准实验通常会产生这些文件：

```text
split_ids.json
prepare_usage.json

round_0/
  all_student_predictions.jsonl
  guide_student_predictions.jsonl
  final_student_predictions.jsonl
  pool_student_predictions.jsonl
  guide_crc_predictions.jsonl
  pool_crc_predictions.jsonl
  crc_summary.json
  selected_train_rows.jsonl
  selection_summary.json

train_rows.jsonl

round_1/
  model/
    adapter/
    model_config.json
  training_rows_used.jsonl
  train_label_snapshot.json
  training_round_summary.json
```

其中：

- `split_ids.json` 定义 guide/final/pool。
- `*_student_predictions.jsonl` 是 student 原始预测。
- `*_crc_predictions.jsonl` 是加上 routing score 和 defer 决策后的预测。
- `selection_summary.json` 记录本轮选样方法、预算拆分和实际选中的 id。
- `train_rows.jsonl` 是累计训练集。
- `round_1/model` 是 LoRA checkpoint。

## 13. 整体算法总结

这套方法可以概括为一个“LLM 自诊断驱动的数据选择与轻量微调框架”：

1. 先把不同任务统一成 `query + document -> 0/1`。
2. 用基础 LLM 产生初始预测，并读取 `1` 和 `0` 的 logprob margin。
3. 用 guide 校准 CRC 阈值，把 pool 分成 accept 和 defer。
4. 根据训练预算选择样本：
   - random：均匀随机 baseline。
   - PCSS：匹配 guide 标签先验，并优先选择不确定样本。
   - CRC error-mass：根据错误是否集中在 defer 区域来分配 accept/defer 预算。
5. 用选出的样本做 LoRA SFT，只监督 `1/0` 答案 token。
6. 用 LoRA 后模型继续推理、评估，必要时进入下一轮。
