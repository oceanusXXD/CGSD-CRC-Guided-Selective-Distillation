# 最终任务、算法与 AAAI 论文路径

> 项目名称：模型诱导采集偏移（Model-Induced Acquisition Shift, MIAS）与分布约束模型引导选择（Distribution-Constrained Model-Guided Selection, DCMS）  
> 推荐论文题目：**When Models Choose Their Own Feedback: Model-Induced Acquisition Shift in Active Preference Learning**  
> 投稿定位：AAAI-27 Main Technical Track  
> 主任务：Active Preference Acquisition for DPO  
> 受控验证任务：Multi-class Active Distillation  
> 先导证据：原二分类 Selective Distillation

---

## 目录

1. [最终研究定位](#1-最终研究定位)
2. [从原二分类论文到最终任务的变化](#2-从原二分类论文到最终任务的变化)
3. [统一任务定义](#3-统一任务定义)
4. [MIAS 的正式定义与识别条件](#4-mias-的正式定义与识别条件)
5. [核心算法 DCMS](#5-核心算法-dcms)
6. [理论内容与声明边界](#6-理论内容与声明边界)
7. [完整实验路径](#7-完整实验路径)
8. [Baseline 设计](#8-baseline-设计)
9. [指标与统计协议](#9-指标与统计协议)
10. [论文主图与主表](#10-论文主图与主表)
11. [AAAI 正文结构](#11-aaai-正文结构)
12. [最终贡献与允许的主张](#12-最终贡献与允许的主张)
13. [Go / No-Go 标准](#13-go--no-go-标准)
14. [参考文献与对位](#14-参考文献与对位)

---

# 1. 最终研究定位

## 1.1 一句话任务

本文研究：

> **当当前模型自己决定哪些未标注样本或偏好对值得获得昂贵监督时，模型已有的类别或属性依赖会不会改变不同数据区域获得监督的概率，并进一步影响后续蒸馏或 DPO；若会，怎样在保留主动选择信息价值的同时限制这种传导。**

论文的核心对象不是一般意义上的“主动学习产生非 IID 数据”。这一事实已经有完整文献。本文关注更具体、可干预、可证伪的闭环：

$$
\text{selector model tendency}
\rightarrow
\text{group-specific acquisition propensity}
\rightarrow
\text{acquired supervision coverage}
\rightarrow
\text{downstream group behavior}.
$$

只有当选择器模型的偏置被人为改变后，采集倾向和监督覆盖随之稳定改变，才能将该部分偏移识别为 **模型诱导采集偏移（MIAS）**。

## 1.2 论文的主次关系

最终论文采用以下结构，不再把三个任务并列处理：

| 模块 | 论文角色 | 必须回答的问题 |
|---|---|---|
| Active Preference Acquisition / DPO | 主任务 | 模型选择偏好反馈时，长度、来源、prompt 类型等属性覆盖是否被当前 policy 改写；是否影响对齐效果与能力保持 |
| 多分类 Active Distillation | 受控因果环境 | 在固定 pool 和真实标签下，改变 class intercept 是否会改变类别采集率及后续每类性能 |
| 原二分类任务 | 先导证据与附录 | 证明现象最早如何被观察到；保留 4 个数据源、7 个 predicates 的完整结果 |
| DCMS | 核心算法 | 在任意基础 acquisition score 外增加分布约束，限制模型偏置向监督覆盖传导 |

## 1.3 AAAI 适配理由

该工作同时连接：

- active learning；
- human-in-the-loop AI；
- LLM alignment；
- knowledge distillation；
- data selection；
- responsible and robust AI。

AAAI Main Track 明确接受理论、方法、算法、实证、整合性和批判性贡献，并偏好提出新问题、连接多个子领域、指出现有目标或假设缺陷的工作。本文的核心价值应由“新问题 + 因果机制 + 通用修正层”共同构成，不能只依赖 DCMS 的最终性能领先。

---

# 2. 从原二分类论文到最终任务的变化

## 2.1 原论文任务

原任务为二分类 selective distillation：

1. 给定未标注池 $\mathcal U=\{x_i\}$；
2. 小模型为每个样本计算预测概率或不确定性；
3. 在预算 $B$ 下挑选一部分样本交给教师标注；
4. 用教师标签微调学生模型；
5. 比较 Random、difficulty、typicality 和 diversity 等方法。

原论文观察到：不同真实类别在选择器上的不确定性分布不同，因此不确定性采样会改变被标注训练集的标签构成。

## 2.2 原有内容的保留方式

保留：

- IMDb、TwitterHate、FEVER、Codebase；
- 4 个数据源上的 7 个二分类 predicates；
- 所有已保存的 logits、uncertainty scores、selected indices、teacher labels；
- Qwen3-0.6B 与部分 Qwen3-1.7B 容量对照；
- Codebase q1 的训练构成控制实验；
- $\varepsilon_{\mathrm{ent}}$ 作为二分类特殊诊断；

需要修正：

- “7 个任务”统一改成“4 个数据源上的 7 个 predicates”；
- guide、seed、calibration、certification 的教师标签全部计入监督预算；
- $\varepsilon_{\mathrm{ent}}$ 依赖 seed / guide 标签，不再称为 zero-cost；
- 固定偏置采样器的期望 shift 不会按 $O(1/\sqrt n)$ 消失；随预算缩小的是随机波动；
- 二分类采样比例不一定符合 Macro-F1 或 worst-class 目标，因此不作为最终主算法。

## 2.3 新增内容

最终论文必须新增：

1. **多分类因果干预**：class-logit intercept、label order、verbalizer permutation；
2. **Active Preference Acquisition 主任务**：隐藏人类偏好标签、主动选择、揭示标签、DPO 更新、独立评测；
3. **MIAS propensity 审计**：不同群组被采集的概率；
4. **DCMS**：支持多属性、soft group、robust constraint 和 utility-retention；
5. **matched-utility composition intervention**：证明下游变化来自数据构成，而不是基础 utility 不同；
6. **公平预算与成本报告**：所有标签、judge calls、训练 token、selector compute 统一统计。

---

# 3. 统一任务定义

## 3.1 通用主动监督采集

在第 $t$ 轮，给定未标注候选池：

$$
\mathcal U_t=\{z_i\}_{i=1}^{N_t}.
$$

当前选择器模型 $f_{\theta_t}$ 为候选计算基础信息得分：

$$
u_i=u_{\theta_t}(z_i).
$$

在本轮预算 $B_t$ 下选择：

$$
S_t\subseteq\mathcal U_t,
\qquad |S_t|=B_t.
$$

oracle 返回监督标签后，更新目标模型：

$$
\theta_{t+1}
=
\operatorname{Train}
\left(
\theta_t,
\bigcup_{s\le t}\mathcal L_s
\right).
$$

总监督预算必须写为：

$$
B_{\mathrm{total}}=B_0+\sum_{t=1}^{T}B_t,
$$

其中 $B_0$ 为所有方法共享的均匀随机 seed。seed 用于初始训练、group estimation 和诊断，同时进入最终训练集，不能从预算中排除。

## 3.2 多分类任务

候选为：

$$
z_i=x_i,
$$

oracle 标签为：

$$
y_i\in\{1,\ldots,K\}.
$$

选择器可使用 Entropy、Margin、BALD、BADGE 等得分。真实类别在选择前不可见，只能在揭示标签后用于诊断和训练。

## 3.3 Active Preference Acquisition

候选为：

$$
z_i=(x_i,y_i^A,y_i^B),
$$

其中 $x_i$ 是 prompt，$y_i^A,y_i^B$ 是两个候选回答。人类偏好标签为：

$$
h_i\in\{A,B\}.
$$

选择阶段必须隐藏：

- chosen / rejected；
- preference strength；
- 人工 justification；
- 任何由真实偏好标签派生的特征。

选择器只能使用：

- prompt 和两个 response；
- 当前 policy / reward model 的输出；
- 标注前可计算属性，如长度、来源、prompt cluster；
- 当前轮前已经合法获得的监督数据。

固定池实验中，数据集已有的人类标签被视为 oracle，只有在样本被选中后才揭示。

## 3.4 四个角色必须分开

| 角色 | 功能 | 约束 |
|---|---|---|
| Selector | 决定哪些候选获得标签 | 不得读取隐藏标签 |
| Oracle | 返回分类或偏好监督 | 成本计入预算 |
| Target model | 用累计监督训练 | 所有方法使用相同训练 recipe |
| Evaluator | 测量最终行为 | 尽量与 oracle、selector 来源解耦 |

---

# 4. MIAS 的正式定义与识别条件

## 4.1 群组与属性

为每个候选定义标注前可计算或可估计的群组向量：

$$
a_i\in[0,1]^M.
$$

偏好任务可包括：

- length-gap bins；
- response source pair；
- prompt embedding cluster；
- domain / safety / helpfulness cluster；
- A/B position；
- 预先指定的交互项，例如 length $\times$ source。

多分类可包括：

- cross-fitted 类别后验；
- 域、来源、长度；
- embedding cluster。

## 4.2 采集倾向

令 $A_i\in\{0,1\}$ 表示候选是否被选。选择倾向为：

$$
\pi_\theta(z_i)=P(A_i=1\mid z_i,f_\theta).
$$

群组 $g$ 的平均采集率：

$$
\rho_g(\theta)
=
\mathbb E[\pi_\theta(Z)\mid G=g].
$$

## 4.3 Propensity transmission identity

被选数据中的群组分布满足：

$$
P_S(G=g)
=
\frac{P_U(G=g)\rho_g(\theta)}
{\sum_hP_U(G=h)\rho_h(\theta)}.
$$

该恒等式说明：pool 即使完全均衡，只要不同群组的 $\rho_g$ 不同，被标注数据就会改变构成。

## 4.4 三个量必须分开

### 模型输出倾向

$$
B_{\mathrm{prior}}
=TV(P(\hat Y),P(Y)).
$$

### 选择—群组耦合

可使用：

$$
C_{\mathrm{sel}}
=
\max_g
\left|
\log\frac{\rho_g}{\bar\rho}
\right|.
$$

也可以报告 $I(u;G)$ 或最大 propensity ratio。

### 最终采集偏移

$$
D_{\mathrm{acq}}
=D(P_S(G),P_U(G)).
$$

模型输出倾向只有通过选择—群组耦合，才会传到最终采集偏移。

## 4.5 Model-induced component

令 $I_\alpha$ 表示只改变选择器偏置强度的干预：

$$
\theta^{(\alpha)}=I_\alpha(\theta).
$$

定义响应曲线：

$$
R_g(\alpha)=\rho_g(\theta^{(\alpha)}).
$$

认定存在 MIAS 需要同时满足：

1. pool、oracle 标签、预算和训练配置保持不变；
2. $R_g(\alpha)$ 随 $\alpha$ 稳定改变；
3. selected distribution 按同方向变化；
4. 变化在多个 seed 和至少两个模型家族中可复现；
5. 该变化与后续 group performance、输出行为或 capability coverage 有可验证关系。

## 4.6 多分类干预

对分类 logits 注入 class-specific intercept：

$$
\ell_\theta^{(\alpha,k)}(x)
=
\ell_\theta(x)+\alpha e_k.
$$

观察：

$$
\alpha
\rightarrow
\rho_k(\alpha)
\rightarrow
P_S(Y=k)
\rightarrow
F1_k.
$$

至少使用 5 个干预强度，包含负值、零点和正值。

## 4.7 DPO 属性干预

当前 policy 的隐式偏好间隔：

$$
m_\theta(z_i)
=
\beta\left[
\log\frac{\pi_\theta(y_i^A\mid x_i)}{\pi_{\mathrm{ref}}(y_i^A\mid x_i)}
-
\log\frac{\pi_\theta(y_i^B\mid x_i)}{\pi_{\mathrm{ref}}(y_i^B\mid x_i)}
\right].
$$

定义标准化长度差：

$$
c_i
=
\frac{\operatorname{len}(y_i^A)-\operatorname{len}(y_i^B)}
{\operatorname{len}(y_i^A)+\operatorname{len}(y_i^B)}.
$$

构造受控间隔：

$$
m_\theta^{(\gamma)}(z_i)
=m_\theta(z_i)+\gamma c_i.
$$

再将其输入 margin / APL selector，观察长度依赖强度如何改变被选 pair 的属性分布和 DPO 后输出行为。

该实验不能预设“不确定 pair 一定长度接近”。实际选中区域由质量信号、属性依赖和噪声共同决定。

---

# 5. 核心算法 DCMS

## 5.1 算法目标

DCMS 不重新发明 APL、ActiveDPO 或 BADGE 的信息得分。它作为一个外层 acquisition layer：

> 在保留基础 selector 大部分信息价值的条件下，约束被选 batch 在预先指定属性上的覆盖，减少模型偏置对监督数据构成的过度传导。

形式上：

$$
\text{Base Selector}
\quad\longrightarrow\quad
u_i
\quad\longrightarrow\quad
\text{DCMS}
\quad\longrightarrow\quad
S_t.
$$

## 5.2 输入

第 $t$ 轮输入：

- 未标注池 $\mathcal U_t$；
- 基础 acquisition score $u_i$；
- 本轮预算 $B_t$；
- soft group vector $\hat a_i$；
- membership interval $[\ell_i,r_i]$；
- 目标 moments $\tau$；
- 最大允许 utility loss $\kappa$；
- 熵正则系数 $\eta$；
- rounding failure probability $\delta$。

## 5.3 基础 utility 标准化

不同 selector 的分数尺度不同。每轮使用 rank normalization：

$$
\bar u_i
=
\frac{\operatorname{rank}(u_i)-1}{N_t-1}.
$$

如果基础 selector 本身进行 sequential batch diversity 更新，则每选中一个样本后重新计算其内部 utility；DCMS 只增加外层约束，不删除原方法的多样性机制。

## 5.4 Soft group membership

### 多分类

真实类别在标注前未知。使用随机 seed 训练与 selector 解耦的 cross-fitted group estimator：

$$
\hat a_{ik}
=
\widehat P(Y=k\mid x_i).
$$

推荐从以下简单方法中固定一个主设置：

- multinomial linear probe；
- vector scaling；
- 小型 calibrated ensemble。

输入可以包括冻结 embedding 和 selector logits，但不能使用未揭示的真实标签，也不能把 selector 的硬预测直接当作真实类别配额。

### 偏好任务

长度、来源、A/B 位置属于直接可观察属性：

$$
\ell_{ig}=r_{ig}=a_{ig}.
$$

prompt cluster 可由冻结 encoder 聚类获得。若聚类不稳定，可用 bootstrap cluster assignment 形成软 membership。

## 5.5 目标 moments

目标分布必须与评测目标一致。

### 部署平均性能

$$
\tau_g
=
\frac1N\sum_{i=1}^Na_{ig}.
$$

### Macro-F1 / worst-class

类别目标可设为：

$$
\tau_k=\frac1K.
$$

域、来源和长度属性仍默认保持 pool moments。

### Active Preference Acquisition

默认保持未标注 pool 的预标注属性分布：

$$
\tau=\mathbb E_{z\sim P_U}[a(z)].
$$

其含义是允许 selector 在每个属性区域内部挑高价值 pair，同时限制其大幅删除某些区域。

## 5.6 Robust moment constraints

若真实 membership 满足：

$$
\ell_{ig}
\le a_{ig}^{\star}
\le r_{ig},
$$

DCMS 使用：

$$
\frac1{B_t}\sum_iq_i r_{ig}
\le
\tau_g+\epsilon_g,
$$

$$
\frac1{B_t}\sum_iq_i\ell_{ig}
\ge
\tau_g-\epsilon_g.
$$

对于直接可观察属性，上下界相同；对于多分类 soft groups，上下界吸收 group estimator 的不确定性。

## 5.7 主优化问题

令 $q_i\in[0,1]$ 为期望 inclusion weight。求解：

$$
\max_q
\sum_iq_i\bar u_i
+
\eta\sum_i
\left[-q_i\log q_i-(1-q_i)\log(1-q_i)\right]
$$

满足：

$$
\sum_iq_i=B_t,
$$

以及全部 robust moment constraints。

熵项的作用是：

- 避免 propensity 退化到极少数点；
- 保持适度探索；
- 支持 dependent rounding；
- 产生可审计的连续 inclusion propensity。

## 5.8 Utility-retention-driven slack

主动采样的非 IID 性可能有利，因此 DCMS 不强制完全匹配目标。

先求无约束解：

$$
q^{(0)}
=
\arg\max_q
\sum_iq_i\bar u_i+\eta H(q),
\qquad
\sum_iq_i=B_t.
$$

定义基础 utility：

$$
U_0=\sum_iq_i^{(0)}\bar u_i.
$$

在预先固定的 slack 网格 $\mathcal E$ 上求解 DCMS，选择最严格但仍保留足够 utility 的解：

$$
\epsilon_t^\star
=
\min\left\{
\epsilon\in\mathcal E:
\sum_iq_i^{(\epsilon)}\bar u_i
\ge(1-\kappa)U_0
\right\}.
$$

主设置固定：

$$
\kappa=0.05.
$$

即最多牺牲 5% 的基础 acquisition utility。$\kappa$ 不根据测试集性能调参；只在消融中报告 $0.02/0.05/0.10$。

## 5.9 离散化与输出

使用 dependent rounding 或 pipage rounding 得到严格大小为 $B_t$ 的集合：

$$
S_t=\operatorname{Round}(q^\star,B_t).
$$

每轮必须保存：

- 连续 propensity $q_i^\star$；
- 最终 selection indicator；
- 预期与实际 moment；
- 约束前后 utility；
- 选中的 sample ids；
- rounding seed。

## 5.10 完整算法流程

### 初始化

1. 从 pool 均匀随机抽取共享 seed $\mathcal L_0$；
2. 揭示 seed 标签并计入总预算；
3. 训练初始分类器或初始 DPO policy；
4. 用 cross-fitting 训练 group estimator；
5. 固定属性定义、目标 moments、slack grid 和评测协议。

### 每轮主动采集

1. 计算基础 selector score；
2. 计算标注前属性和 soft group interval；
3. 计算无约束 utility $U_0$；
4. 依次求解 slack grid；
5. 选择满足 utility-retention 的最严格可行解；
6. dependent rounding 得到 $S_t$；
7. 向 oracle 查询所选样本；
8. 加入累计训练集；
9. 用统一 recipe 更新模型；
10. 记录 propensity、coverage、utility、成本和下游指标。

# 6. 理论内容与声明边界

正文只保留四项清晰结果。

## 6.1 命题 1：采集分布传导恒等式

$$
P_S(G=g)
=
\frac{P_U(G=g)\rho_g}
{\sum_hP_U(G=h)\rho_h}.
$$

用途：证明 group-specific propensity 精确决定 selected distribution。

## 6.2 命题 2：模型 score 分布导致采集率差异

在只有 cardinality constraint 的熵正则选择中：

$$
q_i^{(0)}
=\sigma\left(\frac{\bar u_i-\nu}{\eta}\right),
$$

因此：

$$
\rho_g
=
\mathbb E\left[
\sigma\left(\frac{\bar U-\nu}{\eta}\right)
\middle|G=g
\right].
$$

用途：说明模型在不同群组上产生不同 score 分布时，采集率差异是系统性的。

## 6.3 命题 3：soft membership 误差下的覆盖界

若：

$$
\max_{i,g}|\hat a_{ig}-a_{ig}^\star|\le\xi,
$$

则连续解真实 moment 的偏差至多为：

$$
\epsilon_g+2\xi.
$$

使用 robust intervals 时，估计误差已经进入约束。

## 6.4 命题 4：rounding 后有限样本偏差

在 dependent rounding 的负相关条件下，以至少 $1-\delta$ 的概率：

$$
\left|
\frac1B\sum_{i\in S}a_{ig}
-
\frac1B\sum_iq_i a_{ig}
\right|
\le
\sqrt{\frac{\log(2M/\delta)}{2B}}.
$$

## 6.5 下游性能声明边界

不证明“匹配 pool 分布必然提升训练后性能”。固定模型的风险差可以由 TV 控制，但训练后参数改变需要额外假设。下游作用通过 matched-utility composition intervention 验证。

---

# 7. 完整实验路径

# 7.1 原二分类重审计

## 目的

确认原始现象在修正预算和统计口径后仍成立，并将其改写为 MIAS 的先导证据。

## 数据

- IMDb；
- TwitterHate；
- FEVER；
- Codebase；
- 共 4 个数据源、7 个 predicates。

## 必须重新计算

对每个 sample 保存或恢复：

$$
\{id,y_i,p_i,u_i,A_i,method,budget,seed\}.
$$

计算：

- $B_{\mathrm{prior}}$；
- 每类 $\rho_g$；
- propensity ratio；
- $D_{\mathrm{acq}}$；
- selected label distribution；
- per-class F1；
- $\varepsilon_{\mathrm{ent}}$；
- 所有教师标签的真实总预算。

## 正文保留

- 一个最直观的二分类置信度分布例子；
- 一行汇总结果；
- Codebase q1 构成干预作为早期证据。

其余完整表进入附录。

# 7.2 多分类受控因果实验

## 主数据集

| 数据集 | 类数 | 作用 |
|---|---:|---|
| AG News | 4 | 类别较均衡，排除 pool imbalance |
| TREC | 6 | 类别语义和样本量更异质，验证自然场景 |

附录可加入 DBPedia-14 或 Emotion，但不作为首要开发依赖。

## 模型

至少两个家族：

- 一个 Qwen 系列 1.5B--2B instruct model；
- 一个 Gemma 或 Llama 系列 2B--3B instruct model。

主实验前固定具体 checkpoint，不在结果出来后切换模型寻找显著性。

## 关键实验

### MC-1：自然采集差异

运行 Random、Entropy、BADGE、GALAXY，测量每类 $\rho_k$ 和 selected distribution。

### MC-2：class-intercept intervention

至少 5 个 $\alpha$，保持 pool 和标签不变，画：

$$
\alpha\rightarrow\rho_k\rightarrow D_{\mathrm{acq}}.
$$

### MC-3：label representation sensitivity

- label order permutation；
- label verbalizer permutation。

用于判断生成式分类器的类别倾向是否会改变采集分布。

### MC-4：下游学习

在相同总预算下训练，报告：

- Accuracy；
- Macro-F1；
- worst-class F1；
- per-class F1；
- AULC。

### MC-5：DCMS

比较：

- Entropy；
- Entropy + DCMS；
- BADGE；
- BADGE + DCMS。

验证 DCMS 是通用 wrapper，不依赖某一种 uncertainty score。

# 7.3 Active Preference Acquisition 主实验

## 主数据集

| 数据集 | 主要用途 | 关键属性 |
|---|---|---|
| HelpSteer2-Preference | 多属性人类偏好与固定池因果分析 | 长度、prompt cluster、文本属性、标注后 preference strength |
| TL;DR human comparisons | 长度与摘要质量关系清晰 | 长度差、source、post cluster |

HH-RLHF 作为附录外部验证。

## 固定池构造

1. 保留原始 prompt 和 response pair；
2. 隐藏 chosen/rejected；
3. 随机交换 A/B；
4. 删除任何泄漏标签的字段；
5. 划分 seed pool、active pool、held-out test；
6. 所有 selector 使用完全相同的 pool ids。

## 初始 policy

若 $\pi_\theta=\pi_{\mathrm{ref}}$，DPO 隐式 reward gap 接近零，无法直接排序。因此所有方法先共享一个均匀随机 seed，训练初始 DPO policy，再开始 active acquisition。

## 主实验流程

1. 当前 policy 对未标注 pair 打分；
2. selector 选择 batch；
3. 揭示人类偏好标签；
4. 将新数据加入累计训练集；
5. 使用相同 DPO recipe 更新；
6. 重复多轮；
7. 在独立测试集和生成评测上比较。

## 主属性

- length-gap bins；
- response source；
- prompt embedding cluster；
- A/B position；
- length $\times$ prompt cluster 交互。

preference strength 只作标注后诊断。

## DPO 因果干预

### PF-1：length coefficient sweep

改变 $\gamma$，测量：

- 各 length-gap bin 的 propensity；
- selected length distribution；
- held-out group accuracy；
- DPO 后生成长度；
- length-controlled win rate。

### PF-2：selector replacement

保持 pool 和训练 recipe 不变，更换 Qwen / 非 Qwen selector，检验不同模型是否选出不同属性覆盖。

### PF-3：A/B swap stability

同一 pair 交换 A/B，选择得分应满足预期对称性或在统计上稳定。若 score 大幅变化，必须单独报告位置偏置。

# 7.4 三个决定论文成败的因果实验

## C1：Bias intervention

- 多分类：class intercept $\alpha$；
- DPO：length coefficient $\gamma$。

需要看到稳定剂量反应，不能只比较“有干预 / 无干预”两个点。

## C2：Matched-utility coverage intervention

从同一 pool 构造：

- 原始 active batch；
- 基础 utility 基本匹配，但群组 moments 更接近目标的 batch。

严格控制：

- 标签数量；
- 平均 utility；
- score 分位数；
- prompt 去重率；
- 训练 token；
- 训练步数。

该实验用于证明 coverage 本身具有下游作用。

## C3：Composition intervention

从同一批已标注候选中，重采样出不同群组构成的训练集。保持样本来源、标签质量和训练配置一致，只改变构成，观察 group performance 和输出行为。

# 7.5 Online DPO 外部验证

固定池人类标签实验是主证据。Online DPO 生成新回答会同时改变候选池和选择器，因此因果解释更弱，只作为外部有效性实验：

- Random vs APL vs APL + DCMS；
- 使用公开可复现管线；
- 同时报告 win rate 和 general capability regression；
- 若资源不足，可放附录，不影响主结论。

---

# 8. Baseline 设计

## 8.1 DPO 主表 Baselines

| 方法 | 必要性 | 作用 |
|---|---|---|
| Random | 必须 | 最强且最公平的无模型选择基线 |
| Reward Margin | 必须 | 最简单的模型引导偏好选择 |
| APL | 必须 | 经典 active preference learning |
| ActiveDPO | 必须 | 理论化、梯度驱动的强主动选择方法 |
| APL + DCMS | 必须 | 验证 DCMS 可包装 uncertainty / preference utility |
| ActiveDPO + DCMS | 必须 | 验证 DCMS 可包装 gradient utility |

## 8.2 DPO 消融或附录 Baselines

| 方法 | 位置 | 说明 |
|---|---|---|
| Moment-matched Random | 关键消融 | 区分 coverage 贡献与 active utility 贡献 |
| Gradient-normalized ActiveDPO | 附录或主消融 | 排除原始梯度范数长度偏置 |
| IPW-only | 消融 | 判断训练阶段 propensity correction 能否替代选择约束 |
| ARM-FI | 条件 baseline | 仅在公开实现完整且可公平复现时加入 |
| PFP / Diverse-NS adapted comparison | Related Work / 条件实验 | 原方法并非同一 active acquisition 设置，不强行作为完全等价主 baseline |

## 8.3 多分类主表 Baselines

| 方法 | 必要性 | 作用 |
|---|---|---|
| Random | 必须 | 无偏基准 |
| Entropy | 必须 | 最直接 uncertainty sampling |
| BADGE | 必须 | uncertainty + gradient diversity |
| GALAXY | 必须 | 类别平衡 active learning 的直接强对照 |
| Entropy + DCMS | 必须 | 简单 utility + 分布约束 |
| BADGE + DCMS | 必须 | 复杂 utility + 分布约束 |

Margin、BALD 和 CoreSet 放入附录。

## 8.4 Baseline 公平性

所有方法必须共享：

- 相同随机 seed 数据；
- 相同总标签预算；
- 相同 active rounds 和 batch sizes；
- 相同候选池；
- 相同训练 token、steps、LoRA rank 和优化器；
- 相同 evaluator；
- 相同 model initialization；
- 相同随机 seed 集合。

选择器额外计算成本单独报告，不可与监督标签成本混为一项。

---

# 9. 指标与统计协议

## 9.1 选择阶段指标

- group acquisition rate $\rho_g$；
- maximum propensity ratio；
- acquisition TV / JS；
- embedding MMD；
- base utility retained；
- A/B swap instability；
- oracle calls；
- selector compute。

## 9.2 多分类下游指标

- Accuracy；
- Macro-F1；
- worst-class F1；
- per-class F1；
- AULC。

## 9.3 DPO 固定 pair 指标

- held-out human preference accuracy；
- worst-group preference accuracy；
- group gap；
- implicit reward calibration；
- length-gap / source / prompt-cluster 分组结果。

## 9.4 DPO 生成指标

- independent judge win rate；
- length-controlled win rate；
- response length / verbosity shift；
- helpfulness / safety 分项；
- general capability regression。

## 9.5 统计要求

1. 核心结果 5 seeds；扩展结果至少 3 seeds；
2. selection distribution 使用 bootstrap 95% CI；
3. 方法差异使用 paired seed comparison；
4. 干预曲线报告 Spearman 单调性和 slope CI；
5. 同时报告 effect size 与置信区间；
6. 不能只报告显著性；
7. 主属性、预算、指标和 baseline 在主实验前冻结；
8. judge 评测必须同步报告 length-controlled 版本；
9. 平均性能、worst-group、capability preservation 必须同时报告。

---

# 10. 论文主图与主表

AAAI 正文只有 7 页，正文建议使用 3 图 3 表，不再增加同质内容。

## Fig. 1：问题与算法总览

### 内容

左侧：

$$
\text{selector tendency}
\rightarrow
\rho_g
\rightarrow
P_S(G)
\rightarrow
\text{downstream behavior}.
$$

右侧：基础 utility 经 DCMS coverage trust region 得到 batch。

### 证明什么

- 论文研究的是闭环反馈采集，不是普通类别不平衡；
- DCMS 作用在标注前 acquisition stage。

## Fig. 2：Bias intervention response curves

### 两个 panel

- 多分类：class intercept vs target-class acquisition rate / acquisition TV；
- DPO：length coefficient vs selected length-gap moments。

### 证明什么

- selector bias 对采集倾向具有因果作用；
- MIAS 不是对自然非 IID 现象的重新命名。

## Fig. 3：Matched-utility composition intervention

### 内容

横轴：coverage deviation；
纵轴：worst-group 或 capability change；
点大小：base acquisition utility；
不同 marker：classification / DPO。

### 证明什么

- 在 utility 相近时，数据构成本身会改变下游行为；
- coverage 是机制变量，而非无关统计量。

## Table 1：MIAS 普遍性

| Task | Model | Selector | Propensity disparity | Acquisition TV | Downstream group gap |
|---|---|---|---:|---:|---:|

覆盖：

- 2 个偏好数据集；
- 2 个多分类数据集；
- 2 个模型家族。

### 证明什么

现象跨任务和模型存在，但允许不同 setting 的方向和强度不同。

## Table 2：DPO 主结果

| Method | Avg pref. acc. | Worst-group | LC win rate | Capability | AULC | Acq. TV | Utility retained | Cost |
|---|---:|---:|---:|---:|---:|---:|---:|---:|

方法：Random、Reward Margin、APL、ActiveDPO、APL+DCMS、ActiveDPO+DCMS。

### 证明什么

DCMS 是否改善 utility--coverage--performance 的综合权衡，而不是仅降低 TV。

## Table 3：关键消融

正文保留：

- w/o robust interval；
- fixed $\epsilon$；
- w/o entropy；
- moment-matched Random；
- IPW only。

### 证明什么

算法收益来自哪些组件，coverage control 是否可被简单训练校正替代。

## 附录图表

- 原二分类全部 7 predicates；
- 所有预算 learning curves；
- Margin、BALD、CoreSet；
- full per-group metrics；
- $\kappa$、$\eta$、cluster 数敏感性；
- online DPO；
- 模型与 LoRA rank 切换；
- 所有 seed-level 数据。

---

# 11. AAAI 正文结构

AAAI-27 Main Track 限制 7 页主内容，总长度最多 9 页，第 8--9 页只允许参考文献。关键证据不能依赖补充材料。

| 章节 | 页数建议 | 核心内容 |
|---|---:|---|
| 1 Introduction | 0.8 | Active DPO running example、MIAS、贡献 |
| 2 Problem and Mechanism | 1.0 | 统一任务、propensity identity、干预定义 |
| 3 DCMS | 1.2 | soft group、robust moments、adaptive slack、rounding |
| 4 Experimental Setup | 0.7 | 2 个 DPO 数据、2 个多分类数据、baseline、指标 |
| 5 Results | 2.5 | 因果曲线、主表、composition intervention、消融 |
| 6 Related Work and Limitations | 0.6 | AL bias、active DPO、feature preservation、限制 |
| 7 Conclusion | 0.2 | 结论 |

## 11.1 Introduction 的开场案例

使用同一个偏好 pair pool：

- Qwen selector 选择出的 batch 在长度和 prompt cluster 上出现某种覆盖；
- 非 Qwen selector 选择出另一种覆盖；
- 两者获得相同数量的人类标签，但后续 DPO 的 group behavior 不同。

该例子必须使用真实实验结果，不能在结果未完成前写成确定事实。

## 11.2 Related Work 的四条线

1. statistical bias in active learning；
2. class-balanced / label-shift active learning；
3. active preference acquisition：APL、ActiveDPO、ARM-FI；
4. preference feature / length preservation：PFP、Diverse-NS、DPO length bias。

定位句应强调：现有工作通常研究通用 active bias、改进 acquisition score 或训练后 feature preservation；本文研究选择器自身偏置如何在标注前改变 acquisition propensity，并用可控干预识别该路径。

---

# 12. 最终贡献与允许的主张

## 12.1 推荐贡献写法

### 贡献 1：问题与机制

> We formulate model-induced acquisition shift in active preference learning and derive how selector-dependent group acquisition propensities determine the composition of acquired supervision.

### 贡献 2：因果证据

> We causally manipulate class and preference-attribute dependence while holding the candidate pool and oracle labels fixed, tracing selector bias through acquisition coverage to downstream group behavior.

### 贡献 3：算法

> We introduce DCMS, a model-agnostic acquisition layer that preserves most of an underlying selector's information utility while enforcing uncertainty-aware distribution constraints over pre-label attributes.

### 贡献 4：实证解释

> We show that acquisition coverage helps explain regimes in which active DPO fails to reliably outperform random sampling, and that controlling this pathway improves the utility--coverage frontier.

## 12.2 可以使用的主张

- selector 会产生 group-specific acquisition propensities；
- propensity 精确决定 selected group distribution；
- 操纵 selector bias 会改变 acquisition shift；
- 某些 coverage shift 对 group learning 或 capability preservation 有不利影响；
- DCMS 在指定 setting 中改善 utility--coverage frontier；
- DCMS 可以包装不同基础 acquisition score。

## 12.3 禁止使用的主张

- 首次发现 active learning 有 sampling bias；
- 所有非 IID active sampling 都有害；
- 匹配 pool 分布必然最优；
- DCMS 无条件提高下游性能；
- 多分类中模型不爱判的类别一定被漏掉；
- DPO 中不确定 pair 一定长度接近；
- 3 seeds 足以证明 performance floor；
- 不计 seed / guide labels 的低监督预算；
- 用标注后属性指导标注前选择。

---

# 13. Go / No-Go 标准

## 13.1 现象进入论文的最低标准

- 至少 2 个 DPO setting 和 2 个多分类 setting 中，bias intervention 显著改变 group propensity；
- propensity identity 能准确复原 selected distribution；
- 至少两个模型家族复现方向一致或机制一致的响应。

## 13.2 下游机制标准

- matched-utility composition intervention 产生稳定的 group performance 或 capability 差异；
- 结果不能完全由标签噪声、训练 token 或 score utility 差异解释；
- 至少一个人类偏好数据集上成立。

## 13.3 DCMS 作为主算法的标准

- 在 APL 和 ActiveDPO 中至少两个基础 selector 上降低 acquisition shift；
- utility retained 达到预设阈值；
- worst-group、AULC 或 capability preservation 至少一项稳定改善；
- 平均性能没有明显且统计显著的退化；
- robust constraints 在实际离散 batch 上确实成立。

## 13.4 失败后的论文路径

| 结果 | 论文路径 |
|---|---|
| DPO 与多分类均有因果偏移，DCMS 有效 | 完整 AAAI：MIAS + DCMS |
| 因果偏移成立，但 DCMS 无稳定下游收益 | 机制 / 诊断论文，DCMS 降为分析工具 |
| 多分类成立，DPO 不成立 | 收缩为 active distillation / active learning 机制论文，不做跨范式主张 |
| DPO 有属性偏移，但没有下游影响 | 论文只能主张 acquisition coverage effect，不能主张 harmful failure mode |
| 自然相关存在，但干预无剂量反应 | 不得命名为 model-induced；重新检查 score 定义和 confound |

---

# 14. 参考文献与对位

1. Farquhar, Gal, and Rainforth. *On Statistical Bias in Active Learning: How and When to Fix It*. ICLR 2021.  
2. Zhao et al. *Active Learning under Label Shift*. AISTATS 2021.  
3. Zhang, Katz-Samuels, and Nowak. *GALAXY: Graph-based Active Learning at the Extreme*. ICML 2022.  
4. Rahmati et al. *Understanding Uncertainty-based Active Learning Under Model Mismatch*. 2024.  
5. Muldrew et al. *Active Preference Learning for Large Language Models*. ICML 2024.  
6. Shen, Sun, and Ton. *Active Reward Modeling: Adaptive Preference Labeling for Large Language Model Alignment*. ICML 2025.  
7. Lin et al. *ActiveDPO: Active Direct Preference Optimization for Sample-Efficient Alignment*. 2025/2026.  
8. Oh et al. *Random Is Hard to Beat: Active Selection in Online DPO with Modern LLMs*. ICLR 2026 Workshop.  
9. Kim et al. *Debiasing Online Preference Learning via Preference Feature Preservation*. Findings of ACL 2025.  
10. Deshpande et al. *Diverse, not Short: A Length-Controlled Data Selection Strategy*. EMNLP 2025.  
11. Wang et al. *HelpSteer2-Preference: Complementing Ratings with Preferences*. 2024.  
12. Stiennon et al. *Learning to Summarize from Human Feedback*. NeurIPS 2020.  
13. Bai et al. *Training a Helpful and Harmless Assistant with Reinforcement Learning from Human Feedback*. 2022.  
14. AAAI. *AAAI-27 Main Technical Track Call*. 2026.

---

## 最终收敛

这篇论文的核心不能写成“原二分类扩展到多分类和 DPO”。最终任务应被表述为：

> **在 active preference acquisition 中识别模型诱导采集偏移，并使用多分类作为可控因果环境、原二分类作为先导证据；随后通过 DCMS 在保留基础信息效用的同时限制监督覆盖被当前模型偏置改写。**

论文是否成立，取决于三条证据链是否同时完成：

1. selector bias 能因果改变 group propensity；
2. coverage composition 能因果改变下游 group behavior；
3. DCMS 能在 utility 基本匹配时改善 coverage 与下游权衡。
