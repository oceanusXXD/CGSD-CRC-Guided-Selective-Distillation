# MIAS 与 DCMS 开发及实验执行文档

> 用途：指导 AI 或研究人员逐阶段完成现象验证、算法实现和论文实验。  
> 原则：以研究问题和证据链为中心，不构建复杂平台，不提前追求大规模工程优化。  
> 配套文档：`01_最终任务_算法与AAAI论文路径_MIAS_DCMS.md`、`03_开发与实验验收清单_MIAS_DCMS.md`

---

## 目录

1. [执行目标与边界](#1-执行目标与边界)
2. [统一实验对象与记录格式](#2-统一实验对象与记录格式)
3. [阶段 D0：冻结任务与实验协议](#3-阶段-d0冻结任务与实验协议)
4. [阶段 D1：原二分类结果重审计](#4-阶段-d1原二分类结果重审计)
5. [阶段 D2：多分类 MIAS 因果验证](#5-阶段-d2多分类-mias-因果验证)
6. [阶段 D3：偏好固定池与初始 DPO](#6-阶段-d3偏好固定池与初始-dpo)
7. [阶段 D4：Active Preference Baselines](#7-阶段-d4active-preference-baselines)
8. [阶段 D5：DPO 端 MIAS 干预](#8-阶段-d5dpo-端-mias-干预)
9. [阶段 D6：DCMS 实现与算法验证](#9-阶段-d6dcms-实现与算法验证)
10. [阶段 D7：下游因果与主结果](#10-阶段-d7下游因果与主结果)
11. [阶段 D8：统计、图表与论文结果冻结](#11-阶段-d8统计图表与论文结果冻结)
12. [实验公平性与复现要求](#12-实验公平性与复现要求)
13. [失败诊断与收缩路径](#13-失败诊断与收缩路径)

---

# 1. 执行目标与边界

## 1.1 最终要验证的三条链

开发工作只围绕以下三条链展开：

### 链 A：模型偏置到采集偏移

$$
\text{bias intervention}
\rightarrow
\rho_g
\rightarrow
D_{\mathrm{acq}}.
$$

### 链 B：采集构成到下游行为

$$
P_S(G)
\rightarrow
\text{group performance / output behavior}.
$$

### 链 C：DCMS 修正

$$
\text{same base utility}
+
\text{coverage constraints}
\rightarrow
\text{better utility--coverage trade-off}.
$$

任何新增模块如果不能服务这三条链，不进入首轮开发。

## 1.2 不做的内容

首轮不需要：

- 构建通用 Web 服务；
- 复杂数据库或任务调度平台；
- 自动搜索大量数据集；
- 自动调参系统；
- 复杂可视化 dashboard；
- 为所有 active learning 方法统一重写框架；
- 在线人工标注系统；
- 大规模 model zoo。

开发产物应能：

- 固定候选池；
- 计算选择分数；
- 运行干预；
- 求解 DCMS；
- 训练分类器或 DPO policy；
- 生成可审计的结果表。

## 1.3 每个阶段的完成定义

每个阶段必须同时产生：

1. 可复现的配置；
2. sample-level 结果；
3. 汇总表；
4. 一段结果解释；
5. 对应验收清单已打勾；
6. 明确的继续、修改或停止结论。

“代码运行成功”不等于阶段完成。

---

# 2. 统一实验对象与记录格式

## 2.1 最小记录单元

每个候选样本或偏好 pair 至少记录：

| 字段 | 含义 |
|---|---|
| sample_id | 在原始 pool 中的稳定唯一编号 |
| split | seed / active_pool / validation / test |
| round | 主动采集轮次 |
| method | Random / APL / ActiveDPO / DCMS 等 |
| model | selector / target model checkpoint |
| seed | 随机种子 |
| base_score | 原始 acquisition score |
| normalized_score | rank-normalized utility |
| q_propensity | DCMS 或随机化选择的连续 inclusion propensity |
| selected | 是否最终被选 |
| observable_groups | 长度、来源、cluster 等标注前属性 |
| oracle_label | 被合法揭示后的分类或偏好标签 |
| train_tokens | 该样本实际贡献的训练 token |

多分类额外记录 logits、predicted class、true class。偏好任务额外记录 A/B 顺序、response lengths、policy log-prob、reference log-prob、implicit margin。

## 2.2 输出层级

### Sample-level

用于复查选择过程和重新计算任何统计量。

### Run-level

每个 dataset × model × method × budget × seed 一行，包含：

- selection metrics；
- training metrics；
- evaluation metrics；
- cost metrics；
- 配置哈希或版本标识。

### Paper-level

由 run-level 结果自动聚合为论文表格。最终数值不得手工录入。

## 2.3 角色隔离

必须分开保存：

- selector score；
- oracle label；
- evaluator score。

任何 active selector 不得从文件名、字段名或缓存中读取 hidden labels。

---

# 3. 阶段 D0：冻结任务与实验协议

## 3.1 目标

在运行主实验前固定研究定义，防止根据结果更换属性、指标或成功标准。

## 3.2 必须冻结的内容

### 数据

- 多分类主数据：AG News、TREC；
- 偏好主数据：HelpSteer2-Preference、TL;DR human comparisons；
- 原二分类：4 数据源、7 predicates；
- HH-RLHF、DBPedia / Emotion 只作附录候选。

### 模型

- 一个 Qwen 系列 1.5B--2B instruct model；
- 一个非 Qwen 2B--3B instruct model；
- 所有方法共享 reference model 和初始化。

### 属性

DPO 主属性固定为：

1. length-gap bins；
2. response source；
3. prompt embedding cluster；
4. A/B position；
5. length × prompt cluster interaction。

### Baselines

DPO 主表：Random、Reward Margin、APL、ActiveDPO、APL+DCMS、ActiveDPO+DCMS。  
多分类主表：Random、Entropy、BADGE、GALAXY、Entropy+DCMS、BADGE+DCMS。

### 主指标

- acquisition TV；
- maximum propensity ratio；
- utility retained；
- Macro / worst-group；
- DPO preference accuracy；
- length-controlled win rate；
- capability regression；
- AULC。

## 3.3 产出

- 一份冻结配置表；
- 数据 split 清单；
- 模型 checkpoint 清单；
- baseline 与指标清单；
- seed 列表；
- 主实验不允许修改项。

## 3.4 继续条件

只有当所有方法可以使用同一 pool、同一 seed 数据和同一预算定义时，才能进入 D1。

---

# 4. 阶段 D1：原二分类结果重审计

## 4.1 研究目标

回答：在修正预算、公平性和统计定义后，原二分类现象是否仍存在？

## 4.2 输入

- 原始 logits；
- selected indices；
- teacher labels；
- train / test metrics；
- guide、calibration、certification 标签记录；
- 所有 budget 和 seed。

若某些历史结果缺少 sample ids 或 logits，应先重跑最关键的代表 setting，而不是根据汇总表推算。

## 4.3 执行步骤

### D1.1 恢复样本级日志

为每个 setting 建立：

$$
\{id,y,p,u,A,method,budget,seed\}.
$$

检查 selected ids 是否能回溯到原始 pool。

### D1.2 修正监督预算

统一计算：

$$
B_{\mathrm{total}}
=B_{\mathrm{seed}}+B_{\mathrm{guide}}+B_{\mathrm{active}}.
$$

certification 若不参与训练，列为独立 evaluation / certification cost，不能隐藏。

### D1.3 重新计算机制量

- predicted class prior；
- class-conditional score distributions；
- $\rho_0,\rho_1$；
- propensity ratio；
- selected label shift；
- acquisition TV；
- $\varepsilon_{\mathrm{ent}}$。

### D1.4 验证 propensity identity

用估计的 $\rho_g$ 预测 selected label distribution，并与实际分布比较。差异应仅来自有限样本和采样实现。

### D1.5 重算下游结果

统一：

- Accuracy；
- Macro-F1；
- per-class F1；
- worst-class F1。

不得混用 Positive-F1 和 Macro-F1 形成主结论。

## 4.4 最低产出

1. `binary_audit_sample_level`；
2. 4 数据源 / 7 predicates 汇总表；
3. 监督成本修正表；
4. $\varepsilon_{\mathrm{ent}}$、propensity disparity、acquisition TV 的关系图；
5. 一段结论：哪些 setting 强、哪些弱、哪些与原叙述不一致。

## 4.5 判定

### 通过

- 至少多个独立数据源存在明显 class-specific propensity；
- selected distribution 可由 propensity identity 解释；
- 原始结果在公平预算下仍有代表性。

### 需要修改

- 现象只集中在单一数据源；
- guide 标签导致预算比较完全失真；
- 结论依赖 Positive-F1 指标切换。

D1 不要求重新证明 DCMS，只负责把原工作变成可信的先导证据。

---

# 5. 阶段 D2：多分类 MIAS 因果验证

## 5.1 研究目标

在一个最干净的可控环境中证明：改变选择器类别倾向，会改变不同真实类别被采集的概率。

## 5.2 数据准备

### AG News

- 保持类别均衡或报告精确 pool prior；
- 固定 pool；
- 固定 seed split；
- 确保 label verbalizer 不泄漏测试统计。

### TREC

- 报告每类样本量；
- 使用相同的 active budget 比例；
- 记录 rare classes 的最小可采集数量。

## 5.3 初始模型

1. 从 pool 均匀随机抽取 seed；
2. 训练初始分类器；
3. 保存每个样本的完整 class logits；
4. 检查基础 accuracy 和 per-class calibration；
5. seed 进入所有方法最终训练集。

## 5.4 自然选择实验

运行：

- Random；
- Entropy；
- BADGE；
- GALAXY。

只做选择统计也必须记录：

- 每类 score 分布；
- 每类采集率；
- selected class distribution；
- acquisition TV；
- score 与 true class 的 mutual information。

此阶段不预设弱类被低采集。结果可能是困难类被过量采集、过度自信错误类被低采集，或不同数据集方向不同。

## 5.5 Class-intercept intervention

对目标类别 $k$：

$$
\ell^{(\alpha,k)}(x)=\ell(x)+\alpha e_k.
$$

建议固定 5--7 个强度，例如对称覆盖负、零、正区间。具体数值依据 logits 标准差预先归一化，不能按结果选取。

每个 $\alpha$：

1. 重新计算 selector score；
2. 使用相同 batch budget；
3. 计算 $\rho_k(\alpha)$；
4. 计算 acquisition TV；
5. 保存 selected ids；
6. 不在第一轮立即训练所有模型，先检查响应曲线。

## 5.6 Label order 与 verbalizer

对生成式分类器：

- 置换标签展示顺序；
- 使用语义等价 verbalizers；
- 检查预测先验、score ranking 和采集率变化。

该实验用于判断现象是否来自任务真实难度，还是 label representation 对小模型造成的选择偏置。

## 5.7 下游训练

只对通过因果门槛的 setting 运行完整训练：

- Random；
- Entropy；
- BADGE；
- GALAXY；
- Entropy+DCMS；
- BADGE+DCMS。

预算至少覆盖低、中、高三个点。所有方法使用相同总标签数和训练 token。

## 5.8 关键输出

- class intercept response curves；
- per-class acquisition rates；
- selected distribution prediction error；
- Macro-F1 / worst-class learning curves；
- DCMS utility retained 与 coverage deviation。

## 5.9 继续条件

至少两个 dataset × model setting 满足：

- $\alpha$ 与目标群组采集率存在稳定单调关系；
- selected distribution 随之改变；
- 方向在多个 seed 上稳定。

若自然采样存在偏移但干预无响应，不得将其归为 MIAS，需要检查 score 定义或 confound。

---

# 6. 阶段 D3：偏好固定池与初始 DPO

## 6.1 研究目标

建立没有标签泄漏、所有方法可公平使用的 active preference acquisition 环境。

## 6.2 固定池选择

主数据：

- HelpSteer2-Preference；
- TL;DR human comparisons。

每个数据集划分：

- shared random seed set；
- active unlabeled pool；
- held-out pairwise test；
- generation evaluation prompts。

不同 split 不能共享完全相同 prompt，避免 prompt memorization。

## 6.3 标签隐藏

在 active pool 文件中删除或隔离：

- chosen / rejected；
- preference label；
- preference strength；
- annotator explanation；
- 任何直接表示胜负的排序字段。

oracle label 单独存放，只有 selection 完成后通过 sample_id 读取。

## 6.4 A/B 随机交换

每个 pair 使用可复现随机种子决定是否交换 A/B。交换后同步更新 oracle label，但 selector 文件中仍不可见。

必须保存原始顺序和实验顺序，以便做 position-bias 分析。

## 6.5 初始 policy

1. 所有方法共享随机 seed；
2. 用 seed 训练初始 DPO policy；
3. 保存 policy 和 reference 对所有 active pool response 的 log-probs；
4. 验证隐式 reward margin 不是全零；
5. 验证训练后 preference accuracy 高于随机，但没有饱和。

如果初始 policy 已接近饱和，active selection 缺乏空间，应降低 seed 比例或更换难度适中的模型 / 数据 split，而不是事后更换指标。

## 6.6 属性构造

### 直接属性

- response A / B token length；
- normalized length gap；
- response source；
- A/B position。

### Prompt clusters

1. 使用冻结 encoder；
2. 只在 active pool prompt 上聚类；
3. cluster 数在主实验前固定；
4. 检查 cluster 是否极端失衡；
5. 保存 cluster prototype 和 assignment。

不使用人工关键词规则定义“风格”。

## 6.7 输出

- 无标签 active pool；
- 单独 oracle label store；
- split manifest；
- A/B swap manifest；
- 初始 policy checkpoint；
- 全 pool log-probs 和属性表；
- 数据泄漏检查报告。

---

# 7. 阶段 D4：Active Preference Baselines

## 7.1 研究目标

先正确复现已有 selector，再研究其 acquisition shift。不能用自定义近似版本替代论文方法后直接得出结论。

## 7.2 Random

- uniform sampling without replacement；
- 与所有方法使用相同 batch size；
- 多个 seed；
- 作为成本和性能基准。

## 7.3 Reward Margin

根据当前 policy 的 implicit reward gap 构造得分。必须明确选择的是：

- 小绝对 margin；
- 大 margin；
- 或某种 uncertainty transform。

不得把不同定义混在同一个方法名下。

## 7.4 APL

根据原方法实现其 prompt entropy / preference criterion。需要记录每一阶段筛选前后的 pool 大小和最终 score，避免只保留最终 selected ids。

## 7.5 ActiveDPO

实现或复现：

- raw gradient score；
- 相对已有数据的新颖性 / information component；
- gradient normalization 版本。

如果完整算法无法复现，应清楚标记为“fixed-pool adaptation”，不能称为原论文完全复现。

## 7.6 Baseline sanity checks

每个 selector 必须通过：

1. 同一输入多次计算结果可复现；
2. 不读取 oracle label；
3. A/B 交换后变化符合公式或被单独记录；
4. score 不全部相同；
5. score 与文本长度的相关性被报告；
6. 选择数量严格等于预算；
7. selected ids 无重复。

## 7.7 第一轮输出

先只做 acquisition audit，不急于完整 DPO：

- Random / Margin / APL / ActiveDPO 的属性分布；
- propensity ratio；
- acquisition TV；
- prompt cluster coverage；
- score-length correlation；
- selector compute。

通过审计后再进入 D5。

---

# 8. 阶段 D5：DPO 端 MIAS 干预

## 8.1 研究目标

证明 preference acquisition 的属性偏移由 selector model 的可控依赖驱动，而不是候选池本身造成。

## 8.2 Length coefficient intervention

构造：

$$
m^{(\gamma)}(z)=m(z)+\gamma c(z).
$$

使用固定 $\gamma$ 网格，至少包含负、零、正值。

每个 $\gamma$：

- 重新计算选择得分；
- 保持 pool、预算和 oracle 标签不变；
- 计算各 length-gap bin 的 propensity；
- 计算 prompt cluster 与 source 的联动变化；
- 保存 selected ids。

## 8.3 Selector model replacement

保持数据与训练 recipe 不变，用两个模型家族作为 selector。比较：

- score rank correlation；
- selected-set overlap；
- attribute TV；
- group propensities。

模型替换实验不能与 target model 训练配置同时变化，否则无法归因。

## 8.4 A/B swap intervention

对相同 pair 的两种顺序分别计算 score：

$$
u(z_{A,B}),\quad u(z_{B,A}).
$$

报告：

- rank correlation；
- top-$B$ overlap；
- position-specific propensity；
- 交换前后 acquisition TV。

## 8.5 标注后诊断

选择完成并揭示标签后，允许分析：

- preference strength；
- chosen length；
- human rating differences。

这些变量只能解释结果，不能回流到同一轮 selector。

## 8.6 继续条件

至少两个偏好 setting 中：

- $\gamma$ 与 length-related propensity 呈稳定响应；
- 该响应不是由 A/B position 单独解释；
- 模型替换导致的 coverage 差异可复现；
- Random 没有同方向系统性响应。

若 DPO 端只有弱相关，没有因果响应，则不应继续用跨范式 MIAS 作为主标题。

---

# 9. 阶段 D6：DCMS 实现与算法验证

## 9.1 研究目标

验证 DCMS 是否正确求解定义的问题，而不是先看最终性能。

## 9.2 实现顺序

### D6.1 精确可观察属性版本

先在 DPO 长度、来源、prompt cluster 上实现。此时 $\ell=r=a$，便于验证约束和 rounding。

### D6.2 多分类 soft group 版本

训练 cross-fitted group estimator，输出：

- $\hat a_{ik}$；
- calibration metrics；
- bootstrap / ensemble interval。

### D6.3 Utility normalization

实现 rank normalization，并验证不同 base selector 的 score 经标准化后处于同一尺度。

### D6.4 无约束解

计算 $q^{(0)}$ 与 $U_0$。验证在约束关闭时，DCMS 与基础 rank utility 的选择行为一致或高度接近。

### D6.5 Slack selection

固定 $\kappa$ 和 $\epsilon$ 网格。记录每个候选 slack 的：

- feasibility；
- utility retained；
- expected moments；
- objective value。

### D6.6 Rounding

完成 dependent rounding，验证：

- batch size 精确；
- 无重复；
- 实际 moments 接近连续解；
- 多次 rounding 的偏差符合理论量级。

## 9.3 算法正确性实验

### A1：Synthetic feasibility

构造已知 group 和 score 的小数据，人工计算最优或近最优结果，检查 DCMS 是否满足约束。

### A2：No-constraint recovery

将 $\epsilon$ 设为足够宽，检查 DCMS 是否恢复基础 selector。

### A3：Exact group coverage

使用直接可观察属性，检查连续解和离散解是否满足目标 moments。

### A4：Soft group error

人为增加 group estimator 噪声，比较普通约束与 robust interval 的真实 coverage violation。

### A5：Utility--coverage frontier

改变 $\kappa$ 或 slack，画 utility retained 与 acquisition TV 的 Pareto 曲线。

## 9.4 DCMS 不应依赖的信号

- 隐藏 true class；
- hidden chosen/rejected；
- 标注后 preference strength；
- test-set performance；
- 根据最终结果选择的属性或阈值。

## 9.5 输出

- 每轮 $q_i$；
- continuous / rounded moments；
- utility retained；
- constraint violation；
- selected ids；
- solver status；
- synthetic 和真实数据正确性报告。

只有算法正确性通过后，才能运行下游大实验。

---

# 10. 阶段 D7：下游因果与主结果

## 10.1 多分类训练

对通过 D2 的 setting 运行：

- Random；
- Entropy；
- BADGE；
- GALAXY；
- Entropy+DCMS；
- BADGE+DCMS。

统一训练：

- seed；
- LoRA rank；
- batch size；
- epochs / steps；
- token budget；
- early stopping 规则。

输出 learning curves 和 per-class metrics。

## 10.2 DPO 主实验

对每个数据集、模型、预算运行：

- Random；
- Reward Margin；
- APL；
- ActiveDPO；
- APL+DCMS；
- ActiveDPO+DCMS。

训练使用累计数据或每轮统一更新方案，必须在所有方法间一致。

## 10.3 Matched-utility coverage intervention

### 构造原则

从同一 pool 中寻找两组 batch：

- 平均和分位数 utility 接近；
- coverage deviation 不同；
- 样本数相同；
- prompt duplication 相近；
- 总 token 相近。

### 训练

使用完全相同模型初始化和训练步骤。重点报告：

- worst-group preference accuracy；
- length-controlled win rate；
- capability regression；
- 输出长度变化。

## 10.4 Composition intervention

从同一已标注候选集合重采样不同构成。建议至少三个 level：低、中、高 coverage deviation。不要改变标签质量和样本难度分位数。

## 10.5 Moment-matched Random

构造满足 DCMS moments 但不使用 active utility 的 Random baseline。比较：

- Random；
- moment-matched Random；
- base active selector；
- active selector + DCMS。

解释：

- moment-matched Random 优于 Random：coverage 目标本身有益；
- active+DCMS 优于 moment-matched Random：信息 utility 仍有贡献；
- active+DCMS 只降低 TV 但不改善性能：算法只能主张 coverage control，不能主张性能修复。

## 10.6 主要结果判读

### 理想结果

DCMS 保留至少 95% utility，显著降低 acquisition shift，改善 worst-group / capability，平均性能不退化。

### 可接受结果

DCMS 平均性能相近，显著降低方差和 worst-group gap。仍可形成稳健性贡献。

### 不足结果

DCMS 只让分布更接近 pool，但平均和分组性能均无改善。此时应降低算法主张，重点转向 MIAS 机制诊断。

---

# 11. 阶段 D8：统计、图表与论文结果冻结

## 11.1 汇总前检查

- 所有主方法完成相同 seed；
- 没有只保留成功运行；
- 所有失败 run 有原因记录；
- 没有根据结果删除不利数据集；
- 预算和 token 统计完整；
- judge 版本固定。

## 11.2 统计

### Selection metrics

- bootstrap CI；
- permutation test；
- effect size。

### Intervention

- Spearman monotonicity；
- slope 与 95% CI；
- 响应曲线而非单点比较。

### Model performance

- paired seed comparison；
- mean ± std；
- 95% CI；
- AULC。

## 11.3 图表生成

按照论文文档固定的 3 图 3 表生成。每张图必须附带：

- 输入结果文件；
- 聚合规则；
- seed 数；
- error bar 定义；
- 是否包含失败 run。

## 11.4 结果冻结

生成：

- `results_manifest`；
- 主表 CSV；
- 附录表 CSV；
- 主图数据；
- 结论与证据映射表。

结果冻结后，只允许修复代码错误或增加明确标注的补充实验，不能替换主指标和主要 baseline。

---

# 12. 实验公平性与复现要求

## 12.1 监督预算

所有已揭示标签都计入：

- random seed；
- active labels；
- guide / calibration labels；
- 任何用于 group estimator 的标签。

纯测试标签不进入训练预算，但必须列为 evaluation resource。

## 12.2 训练公平

- 相同初始化；
- 相同训练 token；
- 相同 optimizer；
- 相同更新次数；
- 相同数据累计规则；
- 相同 prompt formatting；
- 相同 generation parameters。

## 12.3 选择公平

- 相同 active pool；
- 相同 batch budget；
- 相同轮数；
- 无 replacement 或 replacement 规则一致；
- 同一 pair 不能重复计费。

## 12.4 评测隔离

- oracle、selector、evaluator 尽量分开；
- human fixed labels 是主因果证据；
- LLM judge 只作生成外部验证；
- judge 结果同时报告 length-controlled 指标。

## 12.5 可复现材料

最低发布：

- 数据 split ids；
- selector scores；
- selected ids；
- 属性定义；
- 配置；
- 聚合脚本；
- 关键模型 checkpoint 或 adapter；
- 运行说明。

---

# 13. 失败诊断与收缩路径

## 13.1 多分类无稳定干预响应

检查顺序：

1. class intercept 是否足以改变预测分布；
2. selector score 是否真的使用干预后的 logits；
3. batch 太大是否掩盖差异；
4. group rate 估计是否用错分母；
5. 随机种子是否固定。

若确认无响应，应停止“模型偏置导致采集偏移”的多分类主张。

## 13.2 DPO 无稳定属性偏移

检查：

1. initial policy margin 是否近零；
2. A/B position 是否主导；
3. active pool 是否过于同质；
4. length coefficient 是否经过标准化；
5. APL / ActiveDPO 实现是否符合原定义。

若两个数据集和两个模型都无稳定响应，DPO 不再作为主任务。

## 13.3 DCMS 降低 TV 但性能不变

该结果可能表示：

- 原 shift 有利或无害；
- 属性选择不对应真实能力群组；
- base utility 已经覆盖充分；
- 下游评测不敏感。

此时保留 MIAS 机制结论，算法只主张可控 acquisition，不宣称性能修复。

## 13.4 DCMS 性能下降

依次检查：

- utility retained 是否达到阈值；
- target moments 是否与评测目标一致；
- constraints 是否过多；
- soft group estimator 是否错误；
- rounding 是否破坏连续解；
- 交互 moments 是否遗漏。

不得在测试集上反复修改 $\tau$ 直到性能变好。

## 13.5 最终收缩原则

证据优先级：

1. 可控干预；
2. propensity 机制；
3. matched-utility downstream effect；
4. DCMS 修正；
5. 扩展数据集和在线实验。

资源不足或结果不稳定时，先保留前四项，删除外围扩展，而不是保留大量弱相关实验。
