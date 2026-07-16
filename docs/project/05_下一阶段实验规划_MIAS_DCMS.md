# MIAS/DCMS 下一阶段实验规划

> 日期：2026-07-14
> 性质：执行规划，不是实验结果报告。
> 适用范围：AG News 多分类闭环、TREC 复现、HelpSteer2 DPO 扩样、历史二分类来源恢复。

---

## 1. 当前判断

现在还不能说“单分类、多分类和 DPO 都做完了”。新主线真正完成运行的只有两个数据集：AG News 和 HelpSteer2-Preference。历史二分类有 4 个数据源、7 个 settings 的汇总证据，但没有恢复原始样本级文件，也没有在当前代码树重跑。

下一阶段不应立刻铺满所有数据集和方法。当前最需要回答的是三个问题：

1. AG News 上已经观察到的采集构成差异，是否会传到下游分类性能；
2. 改变模型的 class intercept，能否在固定数据池上改变类别采集率；
3. HelpSteer2 的小型 CPU pilot 能否扩展到足以比较方法的样本量和 seed 数。

执行优先级据此固定为：

$$
\text{AG News 下游与因果闭环}
\rightarrow
\text{TREC 机制复现}
\rightarrow
\text{HelpSteer2 DPO 扩样}
\rightarrow
\text{第二数据集与完整论文矩阵}.
$$

历史二分类暂不追加计算，直到样本级来源恢复。

---

## 2. 已有证据与尚缺证据

### 2.1 数据集口径

| 模块 | 当前数据集 | 已完成内容 | 不能据此声称的内容 |
|---|---|---|---|
| 历史二分类 | IMDb、FEVER、Codebase、TwitterHate | 4 个数据源、7 个 settings 的旧汇总与来源清单 | 当前代码重跑、样本级 MIAS 审计、因果结论 |
| 多分类 | AG News | 固定 split、全 active pool 真模型评分、4 种方法的自然采集诊断 | 下游 LoRA 效果、DCMS 效果、class-intercept 因果关系、多 seed 与多模型复现 |
| DPO | HelpSteer2-Preference | Random 与 ActiveDPO+DCMS 各 1 个完整 CPU pilot | 方法优劣、稳定的下游提升、六方法主表 |
| 尚未运行 | TREC、TL;DR | 无新实验 | 任何结果性结论 |

按“在当前代码树真正运行”计，共 2 个数据集；若只统计历史二分类清单，则项目曾涉及 6 个数据集。两种口径不得混写。

### 2.2 AG News 已测结果

固定划分为 400 条 seed、5,000 条 selector-safe active pool、5,000 条 test。Qwen3-0.6B 已在 CPU float32 下完成全部 5,000 条 active 样本评分。现有采集诊断使用 budgets 100、500、1,000。

| 方法 | TV@100 | TV@500 | TV@1,000 | 相对 Random 95% envelope |
|---|---:|---:|---:|---|
| Random | 0.0814 | 0.0294 | 0.0234 | 未超过 |
| Entropy | 0.3046 | 0.2866 | 0.2416 | 三个预算都超过 |
| BADGE | 0.2022 | 0.1900 | 0.2000 | 三个预算都超过 |
| GALAXY | 0.1152 | 0.1132 | 0.1452 | 500、1,000 超过；100 未超过 |

Entropy 分数与类别之间的 adjusted uncertainty coefficient 为 0.2143，permutation $p=0.001$。这支持“selector score 与类别有关”，但还没有证明下游性能受影响。

结果来源：`experiments/runs/multiclass/ag_news/diagnostics/classification_diagnostics.json`。

### 2.3 HelpSteer2 CPU pilot 已测结果

当前 pilot 使用 20 条 selection pool、4 条 active budget、4 条 held-out，seed 为 1。

| 方法 | Preference Acc. | AULC | Worst-group | Raw judge win | Length-controlled win | Acquisition TV |
|---|---:|---:|---:|---:|---:|---:|
| Random | 0.50 | 0.50 | 0.00 | 0.50 | 0.333 | 0.60 |
| ActiveDPO+DCMS | 0.50 | 0.50 | 0.00 | 0.50 | 0.333 | 0.50 |

ActiveDPO+DCMS 的 utility retained 为 1.0，selected slack 为 0.5，最大约束违反为 0.4。两个方法的下游指标相同，held-out 只有 4 条，因此这一结果只能证明执行链路可运行，不能用于判断方法优劣。

完整 HelpSteer2 selector-safe 数据已经准备好：256 条 seed、8,000 条 active pool、256 条 held-out、165 条 test。下一轮应使用这份固定 split，不继续扩大 20 条样本的 pilot。

结果来源：`experiments/reports/dpo_pilot/current/`。

### 2.4 当前计算条件

当前机器有 4 个逻辑 CPU、2 个物理核心、15 GiB 内存，没有可用 CUDA GPU。CPU 只支持到 AVX2，本轮模型推理中 float32 快于 bfloat16，因此：

- CPU 任务默认使用 float32；
- 缓存 logits、selection、审计和小规模 pilot 可继续在当前机器运行；
- 多 seed LoRA 和全量 DPO 矩阵在启动前必须先做耗时基准；
- 若全量 DPO 评分预计超过 72 小时 wall time，不得静默缩小数据池，应转到 GPU 或单独冻结一份缩小协议。

---

## 3. 第一优先级：补齐 AG News 证据链

### 3.1 A0：冻结输入并完成样本级审计

先将现有 scored pool、oracle store、split、方法、预算和 seed 写入同一份 protocol manifest。每个方法和预算必须能够生成：

$$
\{id,y_i,p_i,u_i,A_i,method,budget,seed\}.
$$

其中真实标签只允许在 selection 完成后从 oracle store 进入诊断与训练。需要执行：

1. selected id 唯一性与 pool 覆盖检查；
2. active、seed、test 三个 split 的交叉检查；
3. selector 输入字段的 hidden-label 泄漏检查；
4. propensity identity 审计；
5. 预算、训练 token、selector compute 的统一记录。

确定性 top-$k$ 方法若直接记录 $p_i=A_i$，只能作为成员记录，不能解释为跨模型或跨 seed 的总体采集概率。总体 propensity 必须来自重复 seed、模型扰动或预先声明的随机选择机制。

**A0 通过条件：**所有 12 个现有 acquisition settings 都能回溯到样本级记录；零重复 id；零 split overlap；零隐藏标签泄漏；审计脚本无 issue。

### 3.2 A1：先跑一个预算的下游训练

第一批固定 budget 500，比较 Random、Entropy、BADGE。选择 500 的原因是 Entropy 与 BADGE 的采集差异都已超过 Random envelope，同时训练样本量比 budget 100 更适合观察下游变化。

建议配置如下；以下是待冻结设置，不是已有结果。

| 项目 | 建议值 |
|---|---|
| 数据划分 | 沿用 400 seed / 5,000 active / 5,000 test |
| Active budget | 500；总监督训练集为 400 seed + 500 acquired |
| 方法 | Random、Entropy、BADGE |
| 模型 | 先沿用 Qwen3-0.6B 与同一 LoRA 配置 |
| Pilot training seed | 1 |
| 主要指标 | Accuracy、Macro-F1、Worst-class F1、AULC |
| 诊断指标 | 每类 F1、采集率 $\rho_k$、TV、maximum propensity ratio |
| 成本指标 | oracle calls、seed labels、training tokens、selector compute |

这一阶段共 3 个训练 run。三种方法必须共享初始化、训练步数、token budget、test set 和评测代码。

**A1 通过条件：**3 个 run 全部完成；训练 token 差异符合预注册公平规则；每类指标齐全；失败 run 保留并记录原因。

**扩种子触发条件：**Entropy 或 BADGE 相对 Random 在 Macro-F1 / Worst-class F1 上出现至少 1.0 个百分点的绝对差异，或任一类别 F1 出现至少 2.0 个百分点差异。该阈值只用于决定是否继续用算力，不等于统计显著性。

若没有达到触发条件，不直接铺开预算矩阵。先完成 A3 的 class-intercept 干预，再判断是否需要在 budget 100 或 1,000 上补一个敏感性点。

### 3.3 A2：将中等预算扩到 3 seeds，再冻结核心 5 seeds

若 A1 达到触发条件，先补 training seeds 2、3：

- Random × 3 seeds；
- Entropy × 3 seeds；
- BADGE × 3 seeds。

含 A1 在内共 9 个训练 run。比较时使用 paired seed delta、bootstrap 95% CI 和 paired permutation test。

达到以下条件后，才补 seeds 4、5：

1. 至少 2/3 seeds 的采集构成差异方向一致；
2. 至少 2/3 seeds 的主要下游差异方向一致；
3. 没有由训练 token、early stop 或失败 run 造成的方法间配置漂移。

最终论文若把该结果列为核心结论，budget 500 的核心方法必须达到 5 seeds。3 seeds 只用于筛选与扩展结果。

### 3.4 A3：运行 class-intercept 因果干预

使用缓存 logits，不重新推理模型。先在 active pool 上计算 logits 的预干预标准差 $s_\ell$，再对 AG News 的 4 个类别分别注入 class-specific intercept：

$$
\ell^{(a,k)}(x)=\ell(x)+a\,s_\ell e_k,
\qquad
a\in\{-2,-1,-0.5,0,0.5,1,2\}.
$$

标准差、单位和强度列表必须在查看干预后的 selected labels 前冻结。下文仍用 $\alpha=a\,s_\ell$ 表示实际加入 logits 的值。

对 Random、Entropy、BADGE 使用同一 pool、同一 budget 500 和同一 tie-breaking 规则，共计：

$$
4\ \text{classes}\times 7\ \alpha\text{s}\times 3\ \text{methods}=84
$$

个 selection evaluation。它们不是 84 个模型训练。

每个点保存 selected ids、每类采集率、TV、score 分布和随机对照。必须先验证 $\alpha=0$ 与自然采集结果逐 id 一致。响应曲线报告完整 7 个点、Spearman 系数、slope CI 和是否存在方向反转，不允许只保留符合预期的类别或强度。

**A3 通过条件：**$\alpha=0$ 完全复现；Random 对 intercept 不响应；至少一个 active selector 的类别采集率随 $\alpha$ 发生可重复变化；所有强度点均有记录。

若 $\alpha=0$ 不能复现，停止后续训练并先修复 score 或 tie-breaking 漂移。

### 3.5 A4：在中等预算加入 DCMS

只有 A0-A3 通过后，才比较：

- Entropy vs. Entropy+DCMS；
- BADGE vs. BADGE+DCMS。

DCMS 的 soft group estimator 只能使用 400 条 seed 标签，不能读取 5,000 条 active pool 的真实类别。沿用冻结配置中的 `main_kappa=0.05` 和 slack grid：

$$
\{0,0.01,0.02,0.05,0.1,0.2,0.5\}.
$$

在 budget 500、3 seeds 下，新增 6 个 `+DCMS` 训练 run。除下游分类指标外，还必须报告 utility retained、constraint violation、selected distribution prediction error、soft-group calibration 和 interval coverage。

**A4 通过条件：**两个 wrapper 都满足冻结的 utility-retention 规则；约束违反有实际下降；不存在 active true label 泄漏；下游结果按 paired seed 完整报告。DCMS 若只改善构成而未改善下游性能，应按原结果表述，不改写成性能提升。

若 budget 500 被列为论文核心结果，再为 Random、Entropy、BADGE、Entropy+DCMS、BADGE+DCMS 补 seeds 4、5，共新增 10 个 run，使五个核心方法都达到 5 seeds。

### 3.6 A5：预算与 GALAXY 扩展

预算 100 和 1,000 的训练只在 A2/A4 稳定后启动。核心候选矩阵为：

$$
5\ \text{methods}
\times 2\ \text{additional budgets}
\times 3\ \text{seeds}
=30\ \text{runs},
$$

其中 5 个方法为 Random、Entropy、BADGE、Entropy+DCMS、BADGE+DCMS。GALAXY 先保留为 acquisition baseline；若中等预算下游结果表明其有独立解释价值，再增加 3 budgets × 3 seeds = 9 个扩展 run。

---

## 4. 第二优先级：在 TREC 复现机制

TREC 的所有设置均为建议值，尚未运行。建议沿用官方 500 条 test，将 5,452 条 train 划分为：

| Split | 建议数量 | 用途 |
|---|---:|---|
| Seed | 300 | 初始模型与 group estimator |
| Active pool | 4,500 | selector-safe 候选池 |
| Calibration / validation | 652 | 训练选择与校准，不进入 active pool |
| Official test | 500 | 独立最终评测 |

按 AG News 的 active-pool 比例，建议 budgets 为 100、450、900，约对应 2%、10%、20%。冻结 split 前先报告 6 个 coarse classes 的数量，并检查 rare class 在 seed 和 validation 中是否有足够样本。

### 4.1 T1：acquisition-only 复现

先用一个模型 seed 完成全 active pool 评分，运行 Random、Entropy、BADGE、GALAXY × 3 budgets，共 12 个 acquisition settings。比较：

- TV 和每类 enrichment；
- Random 95% envelope；
- score 与类别依赖；
- AG News 与 TREC 的偏移方向是否一致；
- rare classes 是否出现系统性欠采集。

若至少一个 active selector 在两个预算上超过 Random envelope，再增加两个模型/训练 seeds 检查稳定性。若没有超过，只报告跨数据集不复现，不继续训练完整 TREC 矩阵。

### 4.2 T2：TREC 下游与 DCMS

中等 budget 450 上先比较 Random、Entropy、BADGE，3 seeds 共 9 个训练 run。若 AG News 上只有一个 active selector达到稳定条件，TREC 可缩为 Random 加该 selector，共 6 个 run，但必须在 run matrix 冻结前记录原因。

自然选择结果通过后，再加入对应的 `+DCMS` wrapper。若 Entropy 与 BADGE 都保留，则新增 6 个训练 run。

### 4.3 T3：TREC class-intercept

沿用 7 个 $\alpha$、Random 对照和两个 active selectors。完整设计为：

$$
6\ \text{classes}\times 7\ \alpha\text{s}\times 3\ \text{methods}=126
$$

个 selection evaluation。仍使用缓存 logits，不计作模型训练。

TREC 的目的不是重复 AG News 的数值，而是检验以下关系能否复现：模型分数依赖改变后，类别采集率是否随之改变；这种改变是否超过随机采样波动；DCMS 是否在保留 utility 的同时减小构成偏移。

---

## 5. 第三优先级：扩大 HelpSteer2 DPO

### 5.1 D0：全量输入与算力门禁

使用现有固定 split：256 seed / 8,000 active / 256 held-out / 165 test。先完成以下共享产物：

1. 由 256 条 seed 训练并注册共享初始 policy checkpoint；
2. 对 8,000 条 active pool 生成 policy/reference log-probs；
3. 对 256 条 held-out 生成独立 evaluation log-probs；
4. 生成 prompt embedding、冻结 cluster 和 DCMS soft-group 输入；
5. 运行 label-isolation、swap、score 非退化和 preflight 审计。

正式评分前先用 128 个 pair 做端到端耗时与内存基准。若线性外推超过 72 小时 wall time，停止当前机器上的全量评分并迁移到 GPU。不得用 pilot 的 20 条 selection pool 代替主 split，也不得在看到结果后改变 pool 大小。

### 5.2 D1：3-seed 两方法扩样

第一轮只比较 Random 与 ActiveDPO+DCMS：

| 项目 | 建议值 |
|---|---|
| Active budget | 256 |
| Training set | 256 seed + 256 revealed active pairs |
| Held-out | 256 pairs |
| Methods | Random、ActiveDPO+DCMS |
| Seeds | 1、2、3 |
| Run count | 6 |

每个 run 使用相同初始 policy、training token budget、update steps 和评测输入。主要指标为 preference accuracy、worst-group、length-controlled win rate、capability regression、AULC；选择指标为 acquisition TV、maximum propensity ratio、utility retained 和 constraint violation。

**进入六方法主矩阵的条件：**

1. 6 个 run 全部完成，或失败 run 有明确记录；
2. 每个 held-out 主要 group 有足够样本；不足时只能按预注册规则合并 group；
3. ActiveDPO+DCMS 在 3 个 seeds 中都满足冻结的 utility-retention 规则；
4. acquisition composition 的方向至少在 2/3 seeds 一致；
5. 训练与评测没有配置漂移，run-record collection 和 preflight 均通过。

这一门禁不要求 ActiveDPO+DCMS 必须优于 Random。若下游差异接近零，但选择构成稳定不同，仍可进入机制分析；论文中不得写成性能提升。

### 5.3 D2：补齐六方法五种子主矩阵

D1 通过后，固定一个数据集、一个模型、一个 budget，运行：

$$
6\ \text{methods}\times 5\ \text{seeds}=30\ \text{runs}.
$$

六种方法为 Random、Reward Margin、APL、ActiveDPO、APL+DCMS、ActiveDPO+DCMS。D1 已完成其中 6 个，因此增量最多为 24 个 run。

核心结果使用 5 seeds；方法差异采用 paired seed comparison，报告 bootstrap CI 和 permutation $p$ 值。只有这一矩阵完成并通过 Gate 8-10 后，才考虑第二个 budget 或 TL;DR。

### 5.4 D3：长度干预与 DCMS 归因

在缓存 log-probs 上运行：

$$
\gamma\in\{-2,-1,-0.5,0,0.5,1,2\}.
$$

先只观察 selector score、selected-set overlap、长度分布、prompt cluster 分布和 acquisition TV。$\gamma=0$ 必须逐 id 复现原始 selector。若响应稳定，再选择预注册的负、零、正三个点进行 DPO 更新；不得根据下游指标挑选干预强度。

DCMS 的归因必须使用 ActiveDPO vs. ActiveDPO+DCMS、APL vs. APL+DCMS 的成对比较。Random vs. ActiveDPO+DCMS 只能用于主结果参照，不能单独证明收益来自 DCMS。

---

## 6. 历史二分类的处理规则

二分类当前状态为 `not_recovered`。下一阶段不重跑 IMDb、FEVER、Codebase 或 TwitterHate，也不从汇总 CSV 反推样本级结论。

只有满足以下条件才解除阻塞：

1. 原始 sample rows 或可验证的上游数据恢复；
2. sample id、label、prediction/logit 能建立一一映射；
3. 数据版本、路径、SHA256 和处理脚本有记录；
4. 至少一个代表 setting 能生成完整的样本级 acquisition record；
5. 新旧汇总指标可以在声明误差内对齐。

在此之前，缺失的历史样本级记录不能用于主线结论或附录 provenance。

---

## 7. 分阶段运行数量

下表区分模型训练和缓存分数上的 selection evaluation，避免把廉价干预点误写成训练规模。

| 阶段 | 模型训练 run | Selection evaluation | 启动条件 |
|---|---:|---:|---|
| AG News A0 审计 | 0 | 12 个已有 setting 的重建审计 | 立即 |
| AG News A1 pilot | 3 | 0 | A0 通过 |
| AG News A2 扩到 3 seeds | +6 | 0 | A1 达触发条件 |
| AG News A3 intercept | 0 | 84 | A0 通过，可与 A1 相邻执行 |
| AG News A4 DCMS | +6 | 若干 slack 点 | A2、A3 通过 |
| AG News 额外 budgets | +30 | 已有自然选择结果可复用 | 中等预算结果稳定 |
| AG News GALAXY 扩展 | +9 | 0 | 有独立解释价值 |
| AG News 中等预算补足 5 seeds | +10 | 0 | A2、A4 的 3-seed 结果稳定 |
| TREC acquisition pilot | 1；复现稳定性时再 +2 | 12；扩 seed 时再 +24 | AG News 最小闭环通过 |
| TREC 下游自然方法 | 9 | 0 | acquisition 复现 |
| TREC DCMS | +6 | 若干 slack 点 | 下游自然方法完成 |
| TREC intercept | 0 | 126 | $\alpha=0$ 复现检查通过 |
| HelpSteer2 D0 | 1 个共享初始 policy | 128-pair 基准及全量共享评分 | AG News 第一批完成后 |
| HelpSteer2 D1 | 6 | 共享 scoring/selection | D0 算力与输入门禁通过 |
| HelpSteer2 D2 | 最多 +24 | 共享 scoring/selection | D1 稳定 |
| HelpSteer2 length intervention | 0 或预注册 3 个训练点 | 7 个 $\gamma$ 点 | $\gamma=0$ 复现 |

第一批实际执行上限固定为：AG News A0、A1、A3，加 HelpSteer2 的 128-pair 耗时基准。完成这批后再重新估算算力，不直接启动后续大矩阵。

---

## 8. 每阶段必须保存的产物

每个 dataset × model × method × budget × seed 至少保存：

- 冻结配置与 `config_hash`；
- split manifest 与输入文件 hash；
- selector scores、selected ids、membership/propensity；
- reveal 记录和监督预算；
- training summary、checkpoint/adapter manifest；
- evaluation metrics 与每组计数；
- cost report；
- run record；
- failure reason；
- 生成表格和图所依赖的输入清单。

每个阶段结束后运行：

1. sample-level selection audit；
2. budget fairness audit；
3. execution status audit；
4. run-record collection；
5. paired metric comparison；
6. claim-to-evidence audit。

没有 run-level 记录的结果不进入论文表格；只有图片、日志截图或汇总均值的实验不视为完成。

---

## 9. 停止与继续规则

### 立即停止当前阶段

- hidden label 出现在 selector 输入；
- split overlap 或 sample id 无法回溯；
- $\alpha=0$ / $\gamma=0$ 不能复现原始选择；
- 跨方法 training token、初始化或 evaluation 输入发生未声明漂移；
- DCMS 使用 active true labels 构造 group membership；
- 失败 run 被删除或没有 failure reason。

### 继续扩展

- 自然采集差异超过 Random envelope，并在多个 seeds 上方向一致；
- 干预曲线在固定 pool 上产生可重复的 composition response；
- 下游差异达到工程触发阈值，或构成差异本身对机制分析有稳定价值；
- DCMS 满足 utility-retention 规则且实际减小约束违反；
- 计算成本在冻结预算内。

### 允许得到零结果

以下结论都可以接受，但必须保留完整记录：

- AG News 有明显 acquisition shift，但下游性能近似不变；
- class intercept 改变 score，却没有稳定改变 selected distribution；
- TREC 不复现 AG News 的偏移方向；
- DCMS 改善覆盖，但不改善平均性能；
- DPO 在扩大 held-out 后仍没有稳定方法差异。

零结果应改变论文主张范围，而不是通过换 seed、换预算或删方法来回避。

---

## 10. 完成定义

下一阶段只有在以下条件同时满足时才算完成：

1. AG News 有自然选择、下游训练、class-intercept、DCMS 和样本级审计的闭环结果；
2. AG News 核心结论达到 5 seeds，扩展结论至少 3 seeds；
3. TREC 至少完成 acquisition 复现，并依据预注册门禁决定是否完成下游与 DCMS；
4. HelpSteer2 使用 256/8,000/256/165 固定 split 完成 3-seed 门禁；
5. 若 D1 通过，完成六方法 × 五种子的 30-run DPO 主矩阵；
6. 所有监督调用、训练 token、judge calls 和 selector compute 均可审计；
7. 论文中的每条结果性主张都能指向实际 artifact；
8. 二分类若仍未恢复，只保留为历史证据，不进入新的样本级或因果结论。

当前最近的执行目标不是“把所有矩阵跑完”，而是用最少的新增训练先判断 AG News 的采集构成差异是否会传到下游，并验证该差异能否被 class intercept 主动改变。这个问题回答清楚后，TREC 和 DPO 扩展才有明确依据。
