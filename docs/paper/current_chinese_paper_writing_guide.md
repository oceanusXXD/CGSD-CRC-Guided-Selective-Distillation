# MIAS/DCMS 当前论文中文写作指南

> 版本：2026-07-16
>
> 用途：根据当前代码、冻结协议、测试结果和已生成的 pilot 产物，组织一篇可投稿的中文论文初稿。本文不是结果报告，也不替代最终 run record。所有方括号内容都必须在正式实验完成后替换为可追溯的结果或引用。

## 1. 先确定论文写什么

### 1.1 推荐定位

论文的中心问题应写成：

> 当一个模型使用自身的分数决定哪些样本获得昂贵监督时，模型状态是否会改变监督数据的分布；如果会，能否在保留样本效用的同时限制这种传导？

论文的核心对象是一个闭环：

\[
\text{selector state}
\rightarrow
\text{acquisition propensity}
\rightarrow
\text{supervision composition}
\rightarrow
\text{downstream behavior}.
\]

建议将 **MIAS (Model-Induced Acquisition Shift)** 写成需要被测量和干预的现象，将 **DCMS (Distribution-Constrained Model-Guided Selection)** 写成作用于任意基础选择器的约束层。不要把 DCMS 写成“消除模型偏差”的算法；更准确的说法是“限制已观测、已声明群组上的采集分布偏移”。

### 1.2 推荐标题

当前最稳妥的英文标题：

> **Model-Induced Acquisition Shift in Model-Guided Supervision Selection**

对应中文标题：

> **模型引导监督选择中的模型诱导采集偏移**

如果多分类干预和 DPO 主实验都完成，可以使用更具体的标题：

> **When Models Choose Their Own Feedback: Measuring and Constraining Model-Induced Acquisition Shift**

中文可写为：

> **当模型选择自己的反馈：模型诱导采集偏移的测量与约束**

只有在 Active Preference Acquisition 证据确实完成后，才把 `Active Preference Learning` 或 `DPO` 放进标题。当前 CPU pilot 还不足以支撑这种标题限定。

### 1.3 一句话贡献

完成正式实验后，论文可以围绕下面这句话展开：

> 我们提出一个固定池、隐藏标签、可干预的实验协议来测量模型诱导采集偏移，并提出 DCMS 作为基础模型引导选择器之上的分布约束层，在保留选择效用的同时控制可观测监督覆盖的偏移。

这句话没有声称所有主动学习都会产生偏移，也没有声称覆盖匹配必然提升下游性能，适合当前证据边界。

## 2. 当前证据边界

写作时必须把证据分为三层。

### 2.1 已由代码和测试确认的内容

- MIAS 选择发生在训练前，先根据 seed-only surrogate 对完整候选池打分，再揭示被选样本的 oracle 标签。
- 二分类使用带 intercept 的 logistic surrogate；多分类使用 softmax surrogate；DPO 使用 response-aware 的 antisymmetric difference feature。
- seed 被确定性划分为 fit、calibration、meta-validation 三部分，目标比例为 60/20/20。
- 基础效用是期望正向 validation influence 除以标注成本，候选标签在选择 API 中不可见。
- DCMS 使用完整候选池、soft group membership、utility-retention 规则、slack grid、rounding seed 和 propensity 记录。
- DPO 选择中，A/B 方向和 tie/non-tie 可训练性分开建模；tie 概率用于降低选择效用，不用于强制 DCMS 选择 tie。
- DPO 训练会记录实际处理的 pair 数和 input token 数，token budget 不再依赖估算的词数。
- `295 passed`、`266 subtests passed` 的完整测试通过；配置和 DPO execution manifest 可以生成。

### 2.2 当前 pilot 已观察、但不能写成主结果的内容

当前 HelpSteer2 CPU pilot 的初始 seed 只有 4 条，其中 1 条是 tie，真正可用于 DPO 的 non-tie pair 只有 3 条。因而 fit/calibration/meta 分区都极小，temperature 和 bootstrap 会进入 `insufficient_data` 状态。

此前的 selector mechanics 运行曾得到：pool size 20、budget 4、MIAS+DCMS utility retained 约 0.9613、selected slack 0.1、最大约束违反约 0.0824。这些数字只能作为旧 pilot 的执行链路记录，不能作为最终算法性能，也不能与正式 Random/MIAS 训练结果比较；tie-aware 修复后还需要重新生成选择产物。

在同一个旧 pilot seed 下，MIAS+DCMS 选中的 4 条中出现过 1 条 tie，而 Random 样本恰好没有 tie。这说明“DPO 不能直接使用二分类 A/B 头”的问题是真实存在的，但它仍然只是机制诊断，不是下游性能证据。

### 2.3 投稿前必须补齐的证据

- 至少 20 条 non-tie seed pair，且 A/B 两个方向都出现；正式预检已经把它作为 MIAS DPO 的最低条件。
- 多分类 class-intercept 或等价 selector-state intervention，并且包含 Random-calibrated acquisition shift。
- 二分类和多分类的固定 split、训练结果、每类指标和重复 seed。
- DPO 的 Random、MIAS、MIAS+DCMS 主矩阵，以及 GradientDPO 作为 ablation 的独立矩阵。
- 相同 active-label budget、相同 train-token budget、相同初始化和独立 held-out evaluator。
- 每个 claim 对应的 run record、selection summary、training summary、evaluation metrics 和 cost report。

## 3. 论文主线结构

建议正文按照研究问题组织，而不是按照数据集流水账组织。

### 3.1 Introduction：为什么模型选择监督值得研究

第一段：说明现代数据标注、蒸馏和偏好获取都在使用模型决定“哪些样本值得标”。监督成本越高，选择器越影响最终训练分布。

第二段：指出现有工作通常把 selection utility 和 downstream quality 放在一起看，却没有明确检查模型状态是否改变了不同群组获得监督的 propensity。

第三段：提出 MIAS 的闭环，说明本文不把普通 non-IID 采样直接称为 MIAS，而是要求固定池、随机校准和 selector-state intervention。

第四段：介绍 DCMS。强调它是 outer constraint layer：不替换基础 score，而是在 utility retention 的条件下约束 observable group moments。

结尾只列三项贡献：

1. 一个可证伪的 MIAS 测量与干预协议；
2. 一个支持 hard/soft/robust group 的 DCMS 选择层；
3. 在已经完成实验门禁的任务上，对 acquisition shift、训练构成和下游行为的成对评估。

不要在 Introduction 中提前写“显著提升”“普遍有效”或“消除偏差”。这些词必须由最终 run records 支持。

### 3.2 Problem Setup：固定池和标签隔离

定义候选池：

\[
\mathcal U=\{z_i\}_{i=1}^{N},
\qquad S\subseteq\mathcal U,
\qquad |S|=B.
\]

定义基础选择器得分 $u_i=f_\theta(z_i)$，并明确 oracle label $y_i$ 在 selection 前不可见。选择后才执行：

\[
\mathcal L_S=\{(z_i,y_i):i\in S\}.
\]

把 selector、oracle、target model、evaluator 四个角色分别定义。特别说明：seed label 既用于 surrogate fitting，也进入最终训练集，因此计入总监督预算。

### 3.3 MIAS：现象定义和识别条件

令 $g_i$ 表示候选的可观测或估计 group membership。可以定义 selected-set moment shift：

\[
\Delta_g(S)=
\frac{1}{|S|}\sum_{i\in S}g_i
-\frac{1}{|\mathcal U|}\sum_{i\in\mathcal U}g_i.
\]

同时报告 acquisition TV：

\[
\operatorname{TV}(S,\mathcal U)
=\frac12\sum_g|\hat p_S(g)-\hat p_\mathcal U(g)|.
\]

MIAS 的识别至少需要三步：

1. **dependence**：score 与 group/attribute 存在可重复关联；
2. **shift**：选择结果超出同预算 Random envelope；
3. **intervention**：改变 selector state（例如 class intercept、length coefficient、verbalizer/order）后，propensity 或 selected composition 随之改变。

只有第一步时，应写成“存在选择器依赖”；只有前两步时，应写成“观察到采集偏移”；三步完成后才使用“MIAS 传导”或有限范围内的因果语言。

### 3.4 MIAS surrogate 和 influence utility

在 seed 的 fit split 上拟合 surrogate $\phi$，在 calibration split 上选择 L2 和 temperature，在 meta-validation split 上计算验证梯度：

\[
g_{\mathrm{meta}}
=\nabla_\phi
\frac{1}{|M|}\sum_{j\in M}
\ell(\phi;x_j,y_j).
\]

对候选 $x_i$ 和可能标签 $y$，定义训练梯度：

\[
g_i(y)=\nabla_\phi\ell(\phi;x_i,y).
\]

本文使用期望正向 validation influence：

\[
I_i
=\sum_y p_\phi(y\mid x_i)
\max\left(0,
g_{\mathrm{meta}}^\top g_i(y)
\right).
\]

如果标注成本为 $c_i>0$，最终基础效用为：

\[
u_i^{\mathrm{MIAS}}=\frac{I_i}{c_i}.
\]

正文必须解释三点：标签是对候选标签的 posterior expectation，不是读取 oracle；正号截断表示只保留预测为正向验证影响的部分；成本归一化必须与实验的 budget 口径一致。

### 3.5 三类任务如何写

#### 二分类

使用带 intercept 的 logistic head：

\[
p(y=1\mid x)=\sigma(w^\top x+b).
\]

写作重点是：二分类是 MIAS 最简单的受控形式，用于展示类别或长度属性上的 acquisition shift；不要把历史二分类结果直接泛化到 DPO。

#### 多分类

使用一个 K-class softmax head：

\[
p(y=k\mid x)=
\frac{\exp(w_k^\top x+b_k)}
{\sum_{r=1}^{K}\exp(w_r^\top x+b_r)}.
\]

应报告 macro-F1、worst-class F1、accuracy 和 AULC，而不是只报告总体 accuracy。多分类固定 split 要保证 seed 能覆盖已有类别；class-intercept intervention 是主要机制证据。

#### DPO / preference

对两个回答分别得到冻结 representation $h_A,h_B$，构造顺序反对称特征：

\[
d_i=[h_A-h_B;\;\operatorname{signed\_length\_gap}].
\]

交换 A/B 后 $d_i$ 应变为 $-d_i$。方向 head 只使用 A/B non-tie seed。由于 tie pair 不能直接用于 DPO，另建顺序不变的 trainability feature：

\[
r_i=|d_i|,
\]

用它估计 $p_i^{\mathrm{non\text{-}tie}}$，再将方向 utility 乘以该概率：

\[
u_i^{\mathrm{DPO}}
=p_i^{\mathrm{non\text{-}tie}}
u_i^{\mathrm{direction}}.
\]

正文要明确：tie 不被当作 A/B 标签；seed feature 文件要覆盖所有 initially revealed pairs，包括 tie；DPO 训练阶段仍只使用揭示后的 non-tie rows。

### 3.6 DCMS：外层分布约束

令 $q_i\in[0,1]$ 为连续 inclusion propensity，$m_{ig}$ 为候选对 group $g$ 的 membership，$\tau_g$ 为完整 pool 的 target moment。DCMS 求解：

\[
\max_{q}
\sum_i q_i\tilde u_i
\]

约束包括：

\[
\sum_iq_i=B,
\qquad
\left|
\frac{1}{B}\sum_iq_im_{ig}-\tau_g
\right|\le \epsilon_g.
\]

其中 $\tilde u_i$ 可以是 rank-normalized base utility。实际实现还需要说明：

- slack 从预声明 grid 中选择；
- 只有达到 utility-retention threshold 的 slack 才可用；
- bootstrap posterior bounds 可作为 robust lower/upper membership；
- 连续解随后进行 seeded rounding；
- 记录 continuous moments、rounded moments、robust moments、propensities、selection indicator、solver status 和 rounding seed。

DCMS 的目标不是让 selected set 与 pool 完全相同，而是报告 utility-retention 与 coverage deviation 的 frontier。

## 4. 实验章节应该怎么写

### 4.1 统一协议

正文先用一段写共同控制条件：

- fixed candidate pool 和 fixed split；
- hidden oracle labels；
- shared seed initialization；
- equal active-label budget；
- equal training-token budget；
- same tokenizer、optimizer、learning rate、DPO beta 和 max length；
- independent held-out evaluator；
- 五个 training seeds 用于正式主结果；
- seed、oracle calls、judge calls、selector compute、training tokens 分开报告。

不要把“相同样本数量”写成“相同训练成本”，DPO 必须同时报告 pair count 和实际 input tokens。

### 4.2 任务矩阵

| 任务 | 正文角色 | 主方法 | 主要指标 |
|---|---|---|---|
| Binary | 先导证据/附录 | Random、MIAS、MIAS+DCMS（若特征和 seed 足够） | TV、enrichment、worst-class 指标、AULC |
| Multiclass | 受控干预 | Random、Entropy、BADGE、GALAXY、MIAS、MIAS+DCMS | Accuracy、Macro-F1、Worst-class F1、AULC |
| Preference/DPO | 主任务 | Random、MIAS、MIAS+DCMS | preference accuracy、length-controlled win rate、worst-group、capability regression、AULC |
| DPO ablation | 消融 | GradientDPO、GradientDPO+DCMS、Reward Margin、APL、ActiveDPO 系列 | 同上，并报告额外 selector cost |

冻结配置中的 `preference` 是主方法列表；旧 Reward Margin/APL/ActiveDPO 归入 preference ablations，不要在正文中把它们与主方法混写。

### 4.3 DPO 特别注意

DPO 实验必须把以下文件链路写清楚：

\[
\text{selector-safe pool}
\rightarrow
\text{MIAS feature merge}
\rightarrow
\text{selection}
\rightarrow
\text{oracle reveal}
\rightarrow
\text{DPO train rows}
\rightarrow
\text{held-out evaluation}.
\]

任何 active-pool oracle label、tie label 或 preference strength 在 selection 前出现，都会破坏标签隔离。训练结果必须引用 `training_summary.json` 中的 `processed_input_tokens`，不能重新用文本词数估算成本。

## 5. Results 章节写作模板

每个小节遵循“claim → evidence → mechanism → limitation”四步。

### 5.1 RQ1：是否存在采集偏移

推荐句式：

> 在固定候选池和相同 label budget 下，`[selector]` 的 selected-set TV 为 `[x]`，而同预算 Random envelope 的 95% 上界为 `[y]`。该结果说明 `[具体属性]` 与选择器得分存在可重复关联。由于本实验尚未改变 selector state，因此我们将其解释为 selection dependence，而不是因果传导。

### 5.2 RQ2：selector state 是否驱动偏移

推荐句式：

> 当 class intercept/length coefficient 从 `[negative]` 扫描到 `[zero]` 再到 `[positive]` 时，目标群组的 acquisition propensity 呈 `[单调/非单调]` 变化，且变化超出 Random-calibrated envelope。该干预支持 selector state 是采集偏移的一个可操作来源，但不证明所有模型或所有群组都具有相同响应。

### 5.3 RQ3：监督构成是否传导到下游

推荐句式：

> 在 matched utility 和相同训练 token budget 下，改变 selected-set composition 后，`[group metric]` 变化 `[x]`，而总体 `[metric]` 变化为 `[y]`。这表明监督构成与下游群组行为之间存在可观测传导。由于实验采用固定 pool 和有限 seed，结论仅适用于声明的任务、模型和群组。

### 5.4 RQ4：DCMS 的 utility-coverage trade-off

推荐句式：

> 与基础选择器相比，DCMS 将 coverage deviation 从 `[x]` 降至 `[y]`，utility retained 为 `[z]`，额外 selector compute 为 `[t]`。因此 DCMS 改变的是 utility-coverage frontier，而不是简单地把基础排序替换为 Random。若下游指标未提升，应如实报告为 coverage 控制与下游收益之间的 trade-off。

### 5.5 当前 pilot 应该怎么写

当前只能写：

> CPU pilot 验证了固定池、特征合并、MIAS/DCMS selection、oracle reveal 和 DPO training accounting 的执行链路。由于初始 seed 仅含 3 条 non-tie pair，surrogate 的 calibration/bootstrap 处于 insufficient-data 状态，因此该 pilot 不用于 MIAS 与 Random 的性能比较。

当前不能写：

- “MIAS 提升了 DPO preference accuracy”；
- “MIAS+DCMS 优于 Random”；
- “DCMS 解决了 DPO 的偏好偏差”；
- “DPO 训练已经证明算法有效”；
- “在所有任务上泛化”。

## 6. 表格和图应该提前设计

### 6.1 主表

**Table 1：选择器与采集偏移**

列：Task、Method、Budget、Acquisition TV、Max Propensity Ratio、Selected Distribution Error、Selector Compute。

**Table 2：下游训练结果**

列：Task、Method、Seed、Train Tokens、Accuracy/Macro-F1 或 Preference Accuracy、Worst Group、Capability Regression、AULC。

**Table 3：DCMS trade-off**

列：Base Utility、DCMS Utility Retained、Coverage Deviation、Robust Violation、Selected Slack、Solver Status。

**Table 4：消融**

列：No gate、No robust interval、Fixed slack、Moment-matched Random、GradientDPO、MIAS+DCMS。

不要把 binary、multiclass、DPO 的 accuracy 放在同一平均列中；它们的 oracle、训练目标和评估含义不同。

### 6.2 主图

1. **反馈闭环图**：selector state、candidate pool、oracle reveal、training、evaluation，以及 DCMS 介入位置。
2. **Intervention response curve**：横轴为 intercept/length coefficient，纵轴为 group propensity 或 selected moment，叠加 Random envelope。
3. **Utility-coverage frontier**：横轴 coverage deviation，纵轴 utility retained，区分基础选择器和 `+DCMS`。
4. **DPO tie gate 图**：显示方向 posterior、non-tie probability、最终 DPO utility 的关系；只在正式 seed 足够时放入正文，否则放方法附录。
5. **下游 group metric 图**：只显示完成五个 seed 和独立评测的任务。

每张图都要在 caption 中写清 pool、budget、seed aggregation、Random calibration 和 error bar 含义。

## 7. Abstract 写作模板

正式结果完成前，不要填写 `[ ]` 中的结果词。推荐结构如下：

> 模型引导的数据选择正在成为主动标注、知识蒸馏和偏好优化中的常见环节，但选择器本身可能改变哪些数据获得监督，以及不同群组获得监督的概率。本文研究这种模型诱导采集偏移（Model-Induced Acquisition Shift, MIAS），并提出分布约束模型引导选择（Distribution-Constrained Model-Guided Selection, DCMS）。MIAS 使用固定候选池、隐藏 oracle 标签和 selector-state intervention，区分 score-group dependence、selected-set shift 与下游行为传导；DCMS 在基础 acquisition utility 之上约束 observable 或 soft group moments，并记录 utility retention、propensity 和 robust constraint violation。我们在 `[已完成任务]` 上进行 `[固定 split/预算/seed]` 实验。结果显示 `[仅填入已冻结 run record 支持的结果]`，同时 selector compute、oracle calls 和实际训练 token 被单独报告。上述结果表明 `[限定范围内的结论]`，但不意味着 DCMS 消除了模型偏差或使覆盖匹配成为普适最优训练分布。

如果正式实验尚未完成，可以使用短版摘要用于内部讨论，但必须把“结果显示”改成“我们设计了实验以检验”。

## 8. Related Work 应该怎么组织

不要按论文逐篇罗列，按技术问题写四段：

1. active learning 与 sampling bias：已有工作如何讨论 non-IID、label bias 和 acquisition function；本文增加 selector-state intervention 和 downstream transmission。
2. data selection / selective distillation：已有工作优化训练效用或不确定性；本文把 selected-set composition 作为显式对象。
3. distribution correction / fairness-aware sampling：已有方法做 reweighting、coverage 或 group constraints；本文使用 model-guided utility 与 DCMS 的外层约束组合。
4. preference data selection / DPO：已有方法使用 reward margin、policy-reference gap 或 gradient utility；本文额外处理 A/B order symmetry、tie gate、label isolation 和 train-token accounting。

所有引用在补入前都要程序化核对标题、作者、年份和 venue。当前仓库没有冻结的 related-work bibliography，不能凭记忆填 DOI 或实验数字。

## 9. Limitations 必须主动写

- MIAS 只能测量已声明的 observable/estimated groups，不能直接覆盖未观测的 latent groups。
- seed 太小时，calibration、bootstrap 和 meta-gradient 都可能不稳定；当前 DPO CPU pilot 正是这种情况。
- DPO tie gate 依赖已揭示 seed 的 tie/non-tie 分布，不能把一个小 seed 的 gate 当作真实全池标签模型。
- fixed-pool 结果不自动推广到 streaming acquisition、不同模型家族或不同 oracle。
- DCMS 的 coverage target 是统计约束，不等于最优训练分布，也不等于公平性的完整定义。
- held-out preference evaluator、judge prompt、A/B order 和 response length 都可能影响结论，必须在主表和附录中保留审计信息。
- selector compute 和训练 token 仍然是实际资源成本；只报告 label count 会高估方法的经济性。

## 10. 从现在到投稿的写作顺序

1. 冻结 datasets、splits、methods、budget、seed list 和 evaluator。
2. 先生成 selection summary、reveal summary、training summary、evaluation metrics 和 cost report。
3. 由 run records 自动生成 Table 1--3 和图 2--5，禁止手工抄结果。
4. 先写 Methods 和 Experimental Protocol，再写 Results。
5. 以四个 RQ 为骨架写 Results，每段都绑定 artifact path 或 run id。
6. 最后写 Introduction、Abstract 和标题，根据最强的已完成证据缩小或扩大叙事范围。
7. 补入经核验的 Related Work 和 Limitations。
8. 做一次 claim audit：每个数字、因果动词、比较句和“显著/鲁棒/泛化”都必须能指向数据、统计检验或明确的限制条件。

## 11. 投稿前最终检查

### 证据

- [ ] 主结果的每一行都有 run id 和配置 hash。
- [ ] 训练前 selection 没有读取 active-pool hidden labels。
- [ ] DPO seed 至少 20 条 non-tie pair，A/B 两个方向都出现。
- [ ] Random、MIAS、MIAS+DCMS 使用相同 label/token budget。
- [ ] 所有正式结果有多个 training seeds 和 uncertainty interval。
- [ ] tie、A/B order、length、source、prompt cluster 都有审计。

### 表述

- [ ] 没有把 pilot 写成主结果。
- [ ] 没有把 dependence 写成 causality。
- [ ] 没有把 coverage matching 写成 bias elimination。
- [ ] 没有把总 accuracy 代替 macro/worst-group 指标。
- [ ] 没有使用未经引用核验的文献或数字。

### 复现

- [ ] 论文参数与 freeze config 一致。
- [ ] feature artifact 的 ID 覆盖、维度、标签安全性通过 preflight。
- [ ] DPO training summary 使用实际 processed tokens。
- [ ] 选择器保存 propensity、rounded moments、robust moments、slack 和 rounding seed。
- [ ] 代码、配置、run records 和图表之间可以相互追溯。

## 12. 当前应采用的结论

在正式多分类干预和 DPO 主矩阵完成前，论文应定位为：

> 一套用于测量模型引导监督选择偏移、识别其传导路径并施加分布约束的统一方法与实验协议；当前实现已经通过代码级和小规模执行链路验证，但跨任务性能结论仍待正式 run records 支持。

这是当前证据最稳妥的写法。后续实验若没有观察到下游提升，也可以保留论文价值：DCMS 仍可能在相同 utility 下减少 coverage deviation，或者揭示 coverage 控制与训练质量之间的真实 trade-off。
