# MIAS 与 DCMS 开发及实验验收清单

> 用途：每完成一个开发模块或实验后，回到本文档逐项打勾。  
> 本文档检查“是否支持研究主张”，不只检查程序能否运行。  
> 使用方法：每次验收复制相应阶段，在 `[ ]` 中打勾，并填写证据位置、结论和未解决问题。

---

## 0. 验收记录模板

- 验收阶段：
- 验收日期：
- 执行者 / AI：
- 代码版本或 commit：
- 配置文件：
- 结果目录：
- 使用数据集：
- 使用模型：
- 随机种子：
- 验收结论：通过 / 条件通过 / 不通过
- 主要阻塞：
- 下一步动作：

---

## 目录

1. [总体验收原则](#1-总体验收原则)
2. [Gate 0：任务与协议冻结](#2-gate-0任务与协议冻结)
3. [Gate 1：原二分类重审计](#3-gate-1原二分类重审计)
4. [Gate 2：多分类基础环境](#4-gate-2多分类基础环境)
5. [Gate 3：多分类 MIAS 因果识别](#5-gate-3多分类-mias-因果识别)
6. [Gate 4：偏好固定池与标签隔离](#6-gate-4偏好固定池与标签隔离)
7. [Gate 5：Active Preference Baselines](#7-gate-5active-preference-baselines)
8. [Gate 6：DPO 端 MIAS 因果识别](#8-gate-6dpo-端-mias-因果识别)
9. [Gate 7：DCMS 算法正确性](#9-gate-7dcms-算法正确性)
10. [Gate 8：下游因果与主结果](#10-gate-8下游因果与主结果)
11. [Gate 9：统计与公平性](#11-gate-9统计与公平性)
12. [Gate 10：论文图表与主张](#12-gate-10论文图表与主张)
13. [最终 Go / No-Go](#13-最终-go--no-go)

---

# 1. 总体验收原则

## 1.1 硬性原则

- [ ] 任何 hidden label 在 selection 完成前都不可访问。
- [ ] 所有已获得的监督标签均计入总预算。
- [ ] 所有方法使用相同候选池、seed 数据和训练 recipe。
- [ ] 结果能回溯到 sample-level ids。
- [ ] 论文数值来自结构化结果聚合，不是手工填写。
- [ ] 每个主张都有对应实验或理论证据。
- [ ] 相关性结果不会被写成因果结论。
- [ ] 未通过因果干预的现象不会命名为 MIAS。
- [ ] DCMS 不会被描述为无条件提升性能。

## 1.2 阶段通过规则

一个 Gate 通过需要：

- 所有“必须项”打勾；
- 证据文件存在；
- 结论经过人工或独立 AI 复核；
- 没有未解释的关键异常；
- 下一阶段不会依赖错误或缺失结果。

## 1.3 禁止的验收方式

- [ ] 没有只看最终平均值而不看 sample-level 选择。
- [ ] 没有只看程序退出码为 0。
- [ ] 没有用单个 seed 通过主结论。
- [ ] 没有删除失败 run 后重新计算平均值。
- [ ] 没有在看到 test 结果后修改主属性或 target moments。
- [ ] 没有用“结果看起来合理”替代公式或统计检查。

---

# 2. Gate 0：任务与协议冻结

## 2.1 任务定义

- [ ] 主任务明确为 Active Preference Acquisition，而不是 offline DPO subset selection。
- [ ] 多分类明确为受控因果验证环境。
- [ ] 原二分类明确为先导证据和附录。
- [ ] MIAS 定义包含 selector bias intervention。
- [ ] DCMS 定义为基础 acquisition score 的外层约束，而不是新 score。

证据位置：  
结论：

## 2.2 数据与模型

- [ ] 主数据固定为 HelpSteer2-Preference、TL;DR、AG News、TREC。
- [ ] 原二分类列为 4 个数据源、7 个 predicates。
- [ ] 至少两个模型家族已经固定。
- [ ] 数据 split ids 已保存。
- [ ] 主实验前未根据初步结果更换数据集。

证据位置：  
结论：

## 2.3 Baselines 与指标

- [ ] DPO 主 baselines 已固定。
- [ ] 多分类主 baselines 已固定。
- [ ] Random 在所有实验中存在。
- [ ] moment-matched Random 被列为关键消融。
- [ ] 主指标同时包含平均性能、worst-group、coverage 和 cost。
- [ ] DPO judge 指标包含 length-controlled 版本。

证据位置：  
结论：

## 2.4 预算

- [ ] $B_{\mathrm{total}}=B_0+\sum_tB_t$ 已写入所有配置。
- [ ] seed labels 计入预算。
- [ ] group estimator 使用的标签计入预算。
- [ ] certification / evaluation labels 单独报告。
- [ ] 不存在“预算 50 但额外使用 1000 guide labels”的情况。

### Gate 0 判定

- [ ] 通过
- [ ] 条件通过
- [ ] 不通过

阻塞项：  
下一步：

---

# 3. Gate 1：原二分类重审计

## 3.1 数据完整性

- [ ] 每个 setting 有稳定 sample_id。
- [ ] logits、score、selected indicator 和 teacher label 可对齐。
- [ ] selected ids 可回溯到原 pool。
- [ ] 没有重复选中样本或重复计费。
- [ ] 缺失历史日志的 setting 已明确标注或重跑。

证据位置：

## 3.2 预算修正

- [ ] guide / seed / active labels 已重新统计。
- [ ] certification labels 未被错误计入训练预算或隐藏。
- [ ] 各方法的公平预算表已生成。
- [ ] 原主表中预算口径不一致的位置已列明。

证据位置：

## 3.3 机制统计

- [ ] 已计算预测 prior。
- [ ] 已计算每类 score distribution。
- [ ] 已计算 $\rho_0,\rho_1$。
- [ ] 已计算 propensity ratio。
- [ ] 已计算 acquisition TV。
- [ ] 已计算 $\varepsilon_{\mathrm{ent}}$。
- [ ] 已区分输出偏好、选择耦合和最终 shift。

证据位置：

## 3.4 Propensity identity

- [ ] 使用 $P_U(G)$ 和 $\rho_g$ 预测 selected distribution。
- [ ] 预测结果与实际 selected distribution 一致。
- [ ] 差异有有限样本解释。
- [ ] 公式实现通过人工小样本检查。

证据位置：

## 3.5 下游指标

- [ ] 主结果统一报告 Macro-F1、Accuracy、worst-class F1。
- [ ] Positive-F1 只作补充。
- [ ] 所有平均值从未四舍五入的 setting 结果计算。
- [ ] Codebase q1 等反例没有被隐藏。

## 3.6 结论检查

- [ ] “7 个任务”已改为“4 个数据源上的 7 个 predicates”。
- [ ] 未声称 $\varepsilon_{\mathrm{ent}}$ zero-cost。
- [ ] 未声称固定偏移按 $O(1/\sqrt n)$ 消失。
- [ ] 未将 PCSS 观察结果写成性能下界。

### Gate 1 判定

- [ ] 通过：原二分类可作为可信先导证据
- [ ] 条件通过：仅部分 setting 可用
- [ ] 不通过：原结果需大规模重跑

主要结论：  
异常：  
下一步：

---

# 4. Gate 2：多分类基础环境

## 4.1 数据

- [ ] AG News pool prior 已报告。
- [ ] TREC 每类样本数已报告。
- [ ] seed / active / test split 无重叠。
- [ ] pool 在所有方法间完全一致。
- [ ] label order 和 verbalizer 配置可追踪。

## 4.2 初始模型

- [ ] 初始模型只使用共享随机 seed 训练。
- [ ] seed 标签计入预算。
- [ ] 每个样本完整 logits 已保存。
- [ ] baseline accuracy 不为随机水平。
- [ ] 模型没有完全饱和。
- [ ] per-class calibration 已报告。

## 4.3 自然采集统计

- [ ] Random、Entropy、BADGE、GALAXY 均完成选择。
- [ ] 每类 score distribution 已保存。
- [ ] 每类采集率已计算。
- [ ] selected class distribution 已计算。
- [ ] acquisition TV 已计算。
- [ ] 没有预设困难类一定低采集。

## 4.4 复现性

- [ ] 同一配置重复运行 selected ids 一致或符合随机化设定。
- [ ] 不同 seed 的差异在预期范围内。
- [ ] 所有选择数量精确等于预算。
- [ ] 无 hidden true label 泄漏。

### Gate 2 判定

- [ ] 通过
- [ ] 条件通过
- [ ] 不通过

证据位置：  
下一步：

---

# 5. Gate 3：多分类 MIAS 因果识别

## 5.1 Class-intercept 干预实现

- [ ] 干预只改变 logits，不改变 pool、标签和模型参数。
- [ ] 至少 5 个 $\alpha$ 强度。
- [ ] $\alpha=0$ 与原始 selector 结果一致。
- [ ] 干预后的 score 确实重新计算。
- [ ] 目标类别和非目标类别均有采集率记录。

## 5.2 响应曲线

- [ ] 已画 $\alpha$ vs $\rho_k$。
- [ ] 已画 $\alpha$ vs acquisition TV。
- [ ] 已报告 Spearman 单调性。
- [ ] 已报告 slope 与 95% CI。
- [ ] 响应在多个 seed 上稳定。
- [ ] 至少两个 dataset × model setting 复现。

## 5.3 Propensity 传导

- [ ] 每个 $\alpha$ 的 selected distribution 可由 propensity identity 预测。
- [ ] 方向与实际一致。
- [ ] Random 不随 $\alpha$ 出现同样系统性变化。

## 5.4 Representation intervention

- [ ] label order permutation 完成。
- [ ] verbalizer permutation 完成。
- [ ] 其对预测 prior、score rank、acquisition shift 的影响已报告。
- [ ] 没有只选择最有利的 permutation。

## 5.5 多模型

- [ ] Qwen 家族完成。
- [ ] 非 Qwen 家族完成。
- [ ] 机制方向至少部分复现。
- [ ] 若方向不同，已有合理解释而非删除结果。

### Gate 3 关键判定

- [ ] **通过 MIAS 识别**：操纵 selector bias 可稳定改变 acquisition propensity
- [ ] **仅相关性成立**：自然差异存在，但干预不稳定
- [ ] **不成立**：干预不改变 propensity

证据位置：  
可写入论文的主张：  
禁止写入的主张：

---

# 6. Gate 4：偏好固定池与标签隔离

## 6.1 Split

- [ ] seed、active pool、held-out pair test 已划分。
- [ ] generation evaluation prompts 已单独划分。
- [ ] 相同 prompt 未跨关键 split 泄漏。
- [ ] 所有 selector 使用同一 active pool ids。

## 6.2 标签隐藏

- [ ] active selector 输入中没有 chosen / rejected。
- [ ] 没有 preference strength。
- [ ] 没有 justification。
- [ ] 没有从文件顺序或字段名推断标签。
- [ ] oracle labels 单独存储。
- [ ] selection 完成后才允许读取 oracle label。

## 6.3 A/B 交换

- [ ] A/B 交换由固定随机种子控制。
- [ ] 交换后 oracle label 同步更新。
- [ ] 原始顺序和实验顺序均保存。
- [ ] position-bias 分析所需字段齐全。

## 6.4 初始 DPO policy

- [ ] 所有方法共享完全相同的 seed。
- [ ] reference model 固定。
- [ ] 初始 policy 已训练并保存。
- [ ] active pool 的 policy / reference log-probs 已保存。
- [ ] implicit margin 不全为零。
- [ ] held-out preference accuracy 高于随机但未饱和。

## 6.5 属性

- [ ] length gap 定义固定。
- [ ] response source 定义固定。
- [ ] prompt encoder 固定。
- [ ] cluster 数固定。
- [ ] cluster assignment 已保存。
- [ ] 没有使用关键词规则临时定义风格。
- [ ] interaction moment 已预先指定。

### Gate 4 判定

- [ ] 通过：固定池可用于严格 active acquisition
- [ ] 条件通过：存在可控限制
- [ ] 不通过：标签泄漏或 split 问题

证据位置：  
下一步：

---

# 7. Gate 5：Active Preference Baselines

## 7.1 Random

- [ ] uniform without replacement 正确。
- [ ] batch size 与其他方法一致。
- [ ] 多 seed 完成。
- [ ] selected ids 无重复。

## 7.2 Reward Margin

- [ ] margin 公式明确。
- [ ] 选择大 margin 还是小 margin 已固定。
- [ ] 没有混用 certainty / uncertainty 定义。
- [ ] score 分布非退化。

## 7.3 APL

- [ ] prompt entropy 阶段实现与论文描述一致。
- [ ] preference criterion 实现明确。
- [ ] 两阶段筛选日志完整。
- [ ] APL 未读取隐藏标签。

## 7.4 ActiveDPO

- [ ] gradient score 已验证。
- [ ] novelty / information component 已实现或明确说明 adaptation。
- [ ] gradient normalization 版本可运行。
- [ ] 与原方法不同之处已记录。

## 7.5 通用 sanity checks

- [ ] 同一输入 score 可复现。
- [ ] score 不全部相同。
- [ ] score 与长度相关性已报告。
- [ ] A/B swap 前后 score 差异已报告。
- [ ] top-$B$ selected ids 可复核。
- [ ] selector compute 已记录。
- [ ] oracle calls 精确等于预算。

## 7.6 第一轮 acquisition audit

- [ ] 每个方法的 length distribution 已报告。
- [ ] source distribution 已报告。
- [ ] prompt cluster coverage 已报告。
- [ ] acquisition TV / JS 已报告。
- [ ] maximum propensity ratio 已报告。
- [ ] Random 作为参考分布存在。

### Gate 5 判定

- [ ] 通过：可进入 MIAS 干预
- [ ] 条件通过：某 baseline 只能作为 adaptation
- [ ] 不通过：selector 实现不可审计

证据位置：

---

# 8. Gate 6：DPO 端 MIAS 因果识别

## 8.1 Length coefficient 干预

- [ ] $c_i$ 已标准化。
- [ ] $\gamma$ 网格包含负、零、正值。
- [ ] $\gamma=0$ 与原始 score 一致。
- [ ] 每个 $\gamma$ 使用相同 pool 和预算。
- [ ] 每个 length bin 的 propensity 已计算。
- [ ] prompt cluster / source 联动变化已计算。

## 8.2 响应判定

- [ ] $\gamma$ vs length-related propensity 曲线已生成。
- [ ] 单调性和 slope CI 已报告。
- [ ] 多 seed 方向稳定。
- [ ] 至少两个 preference setting 复现。
- [ ] Random 未表现同方向系统性响应。

## 8.3 Selector replacement

- [ ] 两个模型家族作为 selector。
- [ ] target training recipe 保持不变。
- [ ] score rank correlation 已报告。
- [ ] selected-set overlap 已报告。
- [ ] attribute coverage 差异已报告。

## 8.4 A/B position

- [ ] 原顺序与交换顺序 score 已比较。
- [ ] position propensity 已报告。
- [ ] 位置偏置没有被错误解释为长度偏置。
- [ ] 必要时 position 进入 DCMS moments。

## 8.5 标注后分析

- [ ] preference strength 只在 selection 后使用。
- [ ] chosen length 只作诊断。
- [ ] 没有将标注后属性回流到同一轮 selector。

### Gate 6 关键判定

- [ ] **通过 DPO MIAS**：可控模型依赖稳定改变属性采集率
- [ ] **部分通过**：只在一个数据集或模型成立
- [ ] **不通过**：没有稳定响应

若不通过：

- [ ] 已停止“Active DPO 为主任务”的强主张。
- [ ] 已决定是否收缩为多分类 / distillation 论文。

证据位置：  
结论：

---

# 9. Gate 7：DCMS 算法正确性

## 9.1 输入与 utility

- [ ] base score 输入正确。
- [ ] rank normalization 正确。
- [ ] 不同 selector 的 utility 尺度可比较。
- [ ] $U_0$ 计算正确。
- [ ] $\kappa$ 在主实验前固定。

## 9.2 Group membership

### 可观察属性

- [ ] length / source / position membership 正确。
- [ ] prompt cluster membership 可复现。

### 多分类 soft groups

- [ ] group estimator 只用合法 seed labels。
- [ ] 使用 cross-fitting 或独立 split。
- [ ] calibration 已报告。
- [ ] interval 覆盖率已检查。
- [ ] 未使用 pool true labels 作为选择约束。

## 9.3 优化问题

- [ ] $0\le q_i\le1$。
- [ ] $\sum_iq_i=B_t$。
- [ ] robust upper constraints 正确。
- [ ] robust lower constraints 正确。
- [ ] entropy 项符号正确。
- [ ] solver status 被保存。
- [ ] infeasible 情况不会静默输出错误 batch。

## 9.4 Slack selection

- [ ] $\epsilon$ 网格预先固定。
- [ ] 每个 slack 的 feasibility 已记录。
- [ ] utility retained 已计算。
- [ ] 选择的是满足阈值的最严格可行解。
- [ ] 没有使用 test performance 选择 slack。

## 9.5 Rounding

- [ ] 最终 batch size 精确为 $B_t$。
- [ ] 无重复样本。
- [ ] rounding seed 保存。
- [ ] 实际 moment 与连续 moment 的差异已报告。
- [ ] 多次 rounding 偏差符合预期量级。

## 9.6 正确性实验

- [ ] Synthetic feasibility 通过。
- [ ] No-constraint recovery 通过。
- [ ] Exact group coverage 通过。
- [ ] Soft group error 实验完成。
- [ ] Robust interval 优于无 robust 版本或差异被解释。
- [ ] Utility--coverage Pareto 曲线已生成。

## 9.7 隐私与标签边界

- [ ] DCMS 未读取 true class。
- [ ] DCMS 未读取 chosen / rejected。
- [ ] DCMS 未读取 preference strength。
- [ ] DCMS 未使用 test metric 调参。

### Gate 7 判定

- [ ] 通过：算法定义与实现一致
- [ ] 条件通过：某扩展尚未完成
- [ ] 不通过：不能进入主训练

证据位置：  
已知限制：

---

# 10. Gate 8：下游因果与主结果

## 10.1 训练公平性

- [ ] 所有方法相同初始化。
- [ ] 相同训练 token。
- [ ] 相同 optimizer 和超参数。
- [ ] 相同 update steps。
- [ ] 相同数据累计规则。
- [ ] 相同 prompt formatting。
- [ ] 相同 generation parameters。

## 10.2 多分类主结果

- [ ] Random 完成。
- [ ] Entropy 完成。
- [ ] BADGE 完成。
- [ ] GALAXY 完成。
- [ ] Entropy+DCMS 完成。
- [ ] BADGE+DCMS 完成。
- [ ] Accuracy / Macro-F1 / worst-class / AULC 完整。
- [ ] 每类 acquisition rate 与每类 F1 可对应分析。

## 10.3 DPO 主结果

- [ ] Random 完成。
- [ ] Reward Margin 完成。
- [ ] APL 完成。
- [ ] ActiveDPO 完成。
- [ ] APL+DCMS 完成。
- [ ] ActiveDPO+DCMS 完成。
- [ ] held-out preference accuracy 完整。
- [ ] worst-group 完整。
- [ ] length-controlled win rate 完整。
- [ ] capability regression 完整。
- [ ] AULC 完整。

## 10.4 Matched-utility intervention

- [ ] 两组 batch 样本数相同。
- [ ] 平均 base utility 匹配。
- [ ] utility 分位数匹配。
- [ ] prompt duplication 相近。
- [ ] token 数相近。
- [ ] coverage deviation 明显不同。
- [ ] 下游训练配置完全一致。
- [ ] group behavior 差异可复现。

## 10.5 Composition intervention

- [ ] 从同一已标注候选集合构造。
- [ ] 至少三个 coverage level。
- [ ] 标签质量相同。
- [ ] 样本难度分位数受控。
- [ ] 训练 token 受控。
- [ ] 结果呈合理趋势或反例被保留。

## 10.6 Moment-matched Random

- [ ] moments 与 DCMS 接近。
- [ ] 不使用 active utility。
- [ ] 与 Random、active、active+DCMS 同表比较。
- [ ] 可以区分 coverage 和 utility 的贡献。

## 10.7 DCMS 成功条件

- [ ] 在至少两个 base selector 上降低 acquisition shift。
- [ ] utility retained 达到预设阈值。
- [ ] worst-group / AULC / capability 至少一项稳定改善。
- [ ] 平均性能无明显统计显著退化。
- [ ] 实际 rounded batch 满足约束。
- [ ] 结果跨两个模型家族至少部分复现。

### Gate 8 判定

- [ ] 完整算法贡献成立
- [ ] 只成立 coverage control，不成立性能修复
- [ ] 只成立 MIAS 机制，DCMS 不足
- [ ] 主结论不成立

证据位置：  
可写主张：  
必须降级的主张：

---

# 11. Gate 9：统计与公平性

## 11.1 Seed 与失败 run

- [ ] 核心结果 5 seeds。
- [ ] 扩展结果至少 3 seeds。
- [ ] 所有预定 seed 均有记录。
- [ ] 失败 run 未被静默删除。
- [ ] 失败原因已分类。

## 11.2 统计方法

- [ ] selection metrics 有 bootstrap 95% CI。
- [ ] distribution differences 有 permutation test 或等价检验。
- [ ] model comparison 使用 paired seeds。
- [ ] 报告 effect size。
- [ ] 报告置信区间。
- [ ] 干预曲线报告单调性和 slope。
- [ ] 没有只报告 p-value。

## 11.3 成本

- [ ] oracle label calls 完整。
- [ ] seed labels 完整。
- [ ] judge calls 完整。
- [ ] selector compute 完整。
- [ ] training tokens 完整。
- [ ] 成本比较口径一致。

## 11.4 评测独立性

- [ ] human fixed labels 是主要因果评测。
- [ ] selector 与 evaluator 尽量不同。
- [ ] judge 版本固定。
- [ ] judge prompt 固定。
- [ ] raw win rate 与 length-controlled win rate 同时报告。

### Gate 9 判定

- [ ] 通过
- [ ] 条件通过
- [ ] 不通过

证据位置：

---

# 12. Gate 10：论文图表与主张

## 12.1 Fig. 1：机制总览

- [ ] 图中区分 selector、oracle、target、evaluator。
- [ ] 图中展示 propensity 中介路径。
- [ ] 图中展示 DCMS 作用在标注前。
- [ ] 未暗示所有 shift 都有害。

## 12.2 Fig. 2：因果响应曲线

- [ ] 包含多分类 class-intercept panel。
- [ ] 包含 DPO length-coefficient panel。
- [ ] error bars 定义明确。
- [ ] 至少 5 个干预强度。
- [ ] 没有只展示成功数据集而隐藏失败 setting。

## 12.3 Fig. 3：Matched-utility composition

- [ ] 横轴为 coverage deviation。
- [ ] 纵轴为 group / capability change。
- [ ] utility 匹配信息可见。
- [ ] 图注说明控制变量。

## 12.4 Table 1：MIAS 普遍性

- [ ] 2 个 DPO 数据集。
- [ ] 2 个多分类数据集。
- [ ] 2 个模型家族。
- [ ] propensity disparity、TV 和 downstream gap 同时出现。
- [ ] 任务数量表述正确。

## 12.5 Table 2：DPO 主结果

- [ ] Random、Margin、APL、ActiveDPO、两种 DCMS 版本完整。
- [ ] 平均、worst-group、LC win rate、capability、AULC 完整。
- [ ] acquisition TV 和 utility retained 完整。
- [ ] cost 完整。
- [ ] 最佳值标记基于统计，不是单个均值。

## 12.6 Table 3：消融

- [ ] w/o robust intervals。
- [ ] fixed $\epsilon$。
- [ ] w/o entropy。
- [ ] moment-matched Random。
- [ ] IPW only。
- [ ] 其余消融进入附录。

## 12.7 论文主张审查

- [ ] 未写“首次发现 active learning sampling bias”。
- [ ] 未写“所有 active shift 都有害”。
- [ ] 未写“匹配 pool 必然最优”。
- [ ] 未写“DCMS 有下游无条件保证”。
- [ ] 未写“困难类一定被漏掉”。
- [ ] 未写“不确定 pair 一定长度接近”。
- [ ] 未写虚假的极低预算。
- [ ] 未把 correlation 写成 causation。
- [ ] Limitations 明确说明属性定义和 fixed-pool 范围。

## 12.8 AAAI 格式

- [ ] 正文技术内容不超过 7 页。
- [ ] 第 8--9 页只放参考文献。
- [ ] 关键证据没有只放 supplement。
- [ ] reproducibility checklist 已准备。
- [ ] 论文匿名化。
- [ ] 引用真实、可核查。

### Gate 10 判定

- [ ] 通过：可形成投稿稿件
- [ ] 条件通过：缺少非关键附录
- [ ] 不通过：主张与证据不匹配

---

# 13. 最终 Go / No-Go

## 13.1 完整 AAAI 方案 Go

以下全部满足才能以 “MIAS + DCMS in Active Preference Learning” 投稿：

- [ ] 多分类 bias intervention 成立。
- [ ] DPO bias intervention 成立。
- [ ] propensity identity 得到验证。
- [ ] matched-utility downstream effect 成立。
- [ ] DCMS 在至少两个基础 selector 上有效。
- [ ] utility retained 达标。
- [ ] worst-group / AULC / capability 有稳定改善。
- [ ] 两个模型家族复现。
- [ ] 预算和成本完全公平。
- [ ] 3 图 3 表可以在正文中独立支撑主张。

## 13.2 机制论文 Go

若以下成立，可以收缩为机制 / 诊断论文：

- [ ] 多分类或 DPO 干预稳定改变 propensity。
- [ ] selected distribution 可由 propensity 解释。
- [ ] downstream composition effect 成立。
- [ ] DCMS 性能收益不稳定。

此时：

- [ ] 标题不突出 DCMS。
- [ ] DCMS 降为 diagnostic correction / analysis baseline。
- [ ] 不宣称算法 SOTA。

## 13.3 多分类论文 Go

若 DPO 端不成立，但分类端完整成立：

- [ ] 删除跨范式主张。
- [ ] 主任务改为 model-induced acquisition shift in active distillation / classification。
- [ ] DPO 结果放 negative result 或 discussion。
- [ ] 保留 DCMS 的多分类版本。

## 13.4 No-Go

满足任一条件应停止当前主张：

- [ ] bias intervention 不改变 propensity。
- [ ] hidden label 泄漏无法排除。
- [ ] 预算无法公平重算。
- [ ] 下游差异完全由 utility / token / 标签质量解释。
- [ ] DCMS 约束在实际 rounded batch 中不成立。
- [ ] 结果只在单个 seed 或单个 predicate 出现。
- [ ] 论文必须依赖未审阅 supplement 才能证明核心结论。

## 13.5 最终验收结论

- 最终论文路径：
- 已通过 Gates：
- 未通过 Gates：
- 可写入摘要的贡献：
- 必须放入 Limitations 的内容：
- 必须删除的主张：
- 最终结果目录：
- 最终代码版本：
- 复核人：
- 日期：
