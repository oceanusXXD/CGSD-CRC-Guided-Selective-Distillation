# Archived AAAI Direction Notes: Selector-Induced Supervision Shift

> Status: research notes. The active manuscript blueprint is
> `docs/paper/aaai_manuscript_blueprint.md`.
>
> Terminology note: this historical direction uses SISS/DGA. The active code,
> protocol, and manuscript use MIAS/DCMS; do not mix the two naming systems in
> a submission.
>
> 目标：将已有二分类发现扩展为一个覆盖二分类、多分类和偏好学习的通用现象，并围绕统一诊断与统一纠正方法组织 AAAI 长文。
>
> 证据边界：二分类已有较完整的现象与机制证据；AG News、DBpedia-14 和 HelpSteer2 的全量模型实验尚待完成。本文档中的跨任务结论目前属于研究假设与实验计划，不应提前写成已验证事实。

## 1. 结论先行

当前 EMNLP 稿件不应直接扩写为 AAAI 长文。旧稿围绕 CRC、error-mass heuristics 和 PCSS 展开，容易被审稿人理解为一个只适用于二分类 selective distillation 的工程性修补。AAAI 版本应当重新定义问题：

> 当一个小模型依据自身预测置信度决定哪些未标注样本值得送给教师标注或蒸馏时，选择分数并不是任务中立的。模型已有的类别偏置、能力盲区和表面属性偏好会与置信度纠缠，使获得的监督数据在任务子群分布上发生系统性偏移。蒸馏训练随后把这种选择器偏置写入学生模型，造成不同任务子群之间的不均衡学习。

推荐将该现象命名为 **Selector-Induced Supervision Shift，SISS**。论文只设置一个主要方法名：**Distribution-Guarded Acquisition，DGA**。AED 可以保留为诊断协议名称，但不要与 DGA 并列包装成两个同等复杂的新算法。

论文主线应当是：

1. 在二分类中发现并解释 SISS；
2. 在多分类中证明它不是 yes/no 标签比例的偶发现象；
3. 在 DPO 中证明“任务子群”不必是离散类别，也可以是长度、prompt 类型、偏好方向和质量差等属性；
4. 用 DGA 将“跨层预算保护”和“层内信息选择”解耦，在限制异常监督分布偏移的同时保留困难样本的信息价值。

## 2. 推荐论文定位

### 2.1 推荐标题

首选：

**Uncertainty Is Not Neutral: Distribution-Guarded Data Acquisition for Model-Driven Distillation**

备选：

- **When Small Models Choose Their Own Supervision: Selector-Induced Distribution Shift in Distillation**
- **From Classification to Preference Alignment: Diagnosing and Correcting Selector-Induced Supervision Shift**
- **Who Chooses the Teacher Queries? Distribution Shift in Confidence-Driven Distillation**

首选标题强调两个 AAAI 友好的信息：这是一个通用机器学习问题，而不是单一 NLP 数据集现象；论文同时包含现象和可部署纠正方法。

### 2.2 一句话贡献

> We identify and characterize selector-induced supervision shift: confidence-driven acquisition can transmit a model's existing task-stratum biases into the supervision distribution. We then introduce distribution-guarded acquisition, which separates cross-stratum budget control from within-stratum informativeness optimization.

### 2.3 明确不再使用的主叙事

以下内容不应继续作为摘要、引言或贡献列表的中心：

- “CRC 同时负责诊断、选择和认证”的 dual-role 叙事；
- CRC-error-mass、NS-error-mass 和 Defer-kcenter 作为主要方法；
- “PCSS is the first method ...”一类未经充分文献支持的首创性表述；
- 将 pool 的真实类别比例直接宣称为唯一最优训练比例；
- 使用固定 1,000 条 guide labels，却不将其计入标注预算；
- 仅凭二分类 yes-rate 变化宣称通用分布偏移；
- 将观测性 2×2 对照直接称为 causal effect。

这些旧方法仍然有价值，但其角色应改为：用于发现现象的历史 probes、二分类机制实验或消融基线。

## 3. 核心问题定义

给定未标注池

\[
\mathcal U=\{x_i\}_{i=1}^{N},
\]

小模型选择器 \(f_\theta\)，模型驱动的 acquisition score \(u_i\)，选择算法 \(A\) 和预算 \(B\)。选择器产生

\[
S_B=A(\mathcal U,f_\theta,u,B).
\]

每个样本还属于一个或多个任务子群 \(G^{(m)}\)。在二分类中，子群可以是真实标签；在多分类中是类别；在 DPO 中可以是回答长度差、较长回答是否获胜、prompt cluster、偏好强度或质量差。

论文研究的因果链是：

\[
\text{selector state}
\rightarrow
\text{score--stratum dependence}
\rightarrow
\text{acquired supervision shift}
\rightarrow
\text{post-distillation stratum behavior shift}.
\]

这里的关键不是普通的训练集与测试集 domain shift，而是一个**内生的、由当前学生模型触发的监督分布变化**。同一个未标注池、同一个教师和同一个预算，仅仅改变小模型选择器或 acquisition score，就可能得到不同的任务子群构成。

### 3.1 SISS 的三个必要层次

论文不能只报告最终性能下降，而应依次测量：

1. **Score entanglement**：置信度排序是否暴露任务子群身份；
2. **Acquisition shift**：实际选择结果是否超出相同预算下 Random 的自然波动；
3. **Behavior transmission**：训练后的模型是否沿着监督分布偏移方向发生子群行为变化。

只有三层同时成立，才能形成完整的现象论证。

### 3.2 主张边界

可以主张：

> DGA limits the transmission of selector bias into the acquired supervision distribution while preserving within-stratum informativeness.

不能主张：

- DGA 消除了模型偏见；
- reference distribution 对所有任务都是最优训练分布；
- uncertainty sampling 总是有害；
- Random 总是优于困难样本选择；
- DGA 自动修复错误或失准的 uncertainty score。

## 4. 统一诊断协议

建议将 AED 定位为一套 audit protocol，而不是第二个重型算法贡献。

### 4.1 Score--stratum dependence

将 acquisition score 转换为分位区间 \(Q_L(u)\)，报告：

\[
D_{UG}=\frac{I(G;Q_L(u))}{H(G)}.
\]

正式实验使用 permutation correction：

\[
D_{UG}^{\mathrm{adj}}=
\frac{I_{\mathrm{obs}}-\mathbb E_\pi[I_{\mathrm{perm}}]}
{H(G)-\mathbb E_\pi[I_{\mathrm{perm}}]}.
\]

同时单独报告 permutation p-value。正文中应称其为 **adjusted uncertainty coefficient**，避免将自定义非对称系数写成标准 adjusted mutual information。

### 4.2 Acquisition shift curve

对预算 \(B\) 定义：

\[
D_{AS}(B)=\operatorname{TV}(p_B,p_{\mathrm{pool}}).
\]

同时报告每个子群的 signed shift、coverage ratio 和 worst-group coverage。Random 基线必须使用完整的嵌套 acquisition trajectories，而不是每个预算独立抽一次 Random。

当前代码已经实现：

- 每个预算的 Random 均值和 95% 包络；
- 跨预算 global max-statistic envelope；
- centered AAS 和 positive AAS；
- worst-group coverage；
- permutation-corrected dependence。

### 4.3 Behavior transmission

对训练后模型报告：

- overall accuracy、macro-F1 或 preference accuracy；
- 每个子群的召回率、错误率或 win-rate；
- 预测分布与训练监督分布的关联；
- capability preservation，尤其是 DPO 后的通用能力；
- 多随机种子均值和置信区间。

论文中的关键图不应只是“某方法比 Random 低几分”，而应展示：

\[
\Delta p_{\mathrm{supervision}}(g)
\quad\text{与}\quad
\Delta p_{\mathrm{prediction}}(g)
\]

之间的方向一致性。

## 5. 统一纠正方法：DGA

### 5.1 两层结构

DGA 将数据选择拆成两个明确层次：

1. **跨层预算分配**：限制不同任务子群或可观察 strata 的预算偏移；
2. **层内信息选择**：在预算约束内继续选择 uncertainty 高、梯度价值高或具有多样性的样本。

这一结构正是旧 PCSS 中最有推广价值的部分。旧方法的问题不是“分层预算 + 层内难度”思想错误，而是分层只使用二分类、层定义依赖 CRC defer/accept、预算强制等于点估计类别比例，并且 guide cost 没有被统一计算。

### 5.2 Random-equivalent guard

DGA 不强制

\[
p_S=p_{\mathrm{pool}}.
\]

它限制的是 uncertainty selection 相对于 Random 的**异常额外偏移**：

\[
D_{\mathrm{TV}}(p_S,p_{\mathrm{ref}})\leq \rho_B,
\]

其中 \(\rho_B\) 来自相同预算下 Random acquisition 的经验包络。这样可以回应“为什么原始类别比例一定最优”的质疑：DGA 不是追求严格比例复制，而是排除超出随机有限样本波动的 selector-induced distortion。

### 5.3 Oracle-DGA

在 benchmark 中利用隐藏真实标签构造机制上限：

\[
\max_{S:|S|=B}\sum_{i\in S}u_i
\]

满足每个 group 的数量区间：

\[
L_g(B)\le n_g(S)\le U_g(B).
\]

Oracle-DGA 回答：如果可以准确阻断 supervision shift，困难样本的信息收益能否恢复。它是机制验证，不是部署算法。

### 5.4 Practical-DGA

部署版本不观察真实 group，需要用选择前可观察信息形成 cells：

分类：

\[
H(x)=(\hat y(x),q_u(x)).
\]

DPO：

\[
H(z)=(\widehat{\mathrm{winner}},q_u(z),q_{\Delta\ell}(z)).
\]

从每个 cell 中无放回随机获取少量 pilot labels，估计

\[
q_{hg}=P(G=g\mid H=h).
\]

然后求解 cell budget \(b_h\)：

\[
\max_{b_h}\sum_h b_h\bar u_h
\]

满足总预算、cell 容量和 distribution guard。pilot 样本计入总预算并进入最终训练集。

### 5.5 WSR 的角色

Waudby-Smith and Ramdas 的 without-replacement confidence sequences 适合为有限 pool 中的 cell composition 提供 time-uniform 区间，并允许自适应停止。它只负责估计，不负责定义 acquisition utility，也不应被写成新的选择算法。

主文建议采用简单 Hoeffding/empirical-Bernstein without-replacement 区间或 WSR 的一个清晰实例。完整推导放附录，避免统计细节淹没核心现象。

### 5.6 层内选择

主版本建议：

- 在 observable cell 内均匀随机选择，确保 pilot composition 可以外推；或
- 先形成足够细的 predicted-class × uncertainty cells，再在 cell 内做简单多样性选择。

Oracle-DGA 可以在真实 group 内选 top uncertainty，用于测量 difficulty value。Practical-DGA 第一版不要在粗 cell 内再次 top-uncertainty，否则 cell 内 composition 可能继续漂移。

PCSS 应重新定义为：

> PCSS is the binary, two-stratum, point-estimate special case that motivated distribution-guarded acquisition.

## 6. 从二分类到多分类再到 DPO

### 6.1 二分类：现象发现与机制锚点

保留 IMDb、TwitterHate、FEVER 和 Codebase 证据，但重新解释旧选择方法：它们不是值得推广的算法，而是能够产生不同 selector bias 和 supervision shift 的实验 probes。

二分类需要形成以下闭环：

- 小模型在 label 0/1 上存在不同置信度结构；
- confidence-driven selection 改变 acquired label marginal；
- 训练后 prediction marginal 和 per-class recall 沿该方向变化；
- 固定 label counts 后，困难样本仍可能提供正收益；
- Oracle-DGA/PCSS 特例能够减少偏移并恢复或保留信息收益。

现有结果已经较强地支持前三项，但 PCSS 的性能证据不稳定，因此不能把旧 PCSS 结果当作最终方法成功证据。

### 6.2 多分类：排除 yes/no 特例解释

AG News 和 DBpedia-14 的作用不同：

- **AG News**：4 类、语义差异明显，适合展示 shift curve、class enrichment 和 2×2 分解；
- **DBpedia-14**：14 类、稀有 coverage 风险更明显，用于验证 worst-group coverage 和方法可扩展性。

多分类核心结果不是只证明 entropy selection 改变 class counts，而是证明：

1. adjusted uncertainty coefficient 显著；
2. 至少两个预算越过 global Random envelope；
3. 某些类别持续富集、另一些类别持续缺失；
4. 监督偏移与训练后 class-wise behavior shift 有方向对应；
5. DGA 在相近 utility 下改善 worst-class coverage 和 macro performance。

### 6.3 DPO：证明 task stratum 不等于 class label

HelpSteer2 的意义不是简单增加一个 LLM 实验，而是将 SISS 从类别分布扩展到 preference attributes。

建议依次诊断：

- response order disagreement；
- preferred/rejected length gap；
- longer-response-wins；
- helpfulness、correctness、coherence、complexity、verbosity gaps；
- prompt cluster；
- preference strength。

双顺序评分使用：

\[
p_{\mathrm{sym}}=\frac12\left[p(A\mid A\text{ first})+1-p(B\mid B\text{ first})\right].
\]

\(p_{\mathrm{sym}}\) 用于 uncertainty，order disagreement 只用于 scorer reliability。当前两条 smoke 样本的平均 disagreement 约为 0.856，说明现有 scorer 在全量实验前必须校准或替换。不能把位置偏置本身当作 acquisition value。

DPO 训练结果至少同时报告：

- preference accuracy 或 judge win-rate；
- general capability benchmarks；
- length和verbosity变化；
- 不同 prompt clusters 的表现；
- order sensitivity；
- Random-DPO 与 uncertainty-DPO 的方差。

## 7. 关键实验问题

### RQ1：SISS 是否跨任务存在？

在二分类、多分类和 DPO 属性上报告 score--stratum dependence、acquisition shift 和 Random-calibrated excess shift。

### RQ2：监督偏移是否传递到训练后行为？

比较 selected supervision composition、训练后 prediction composition 和 group-wise performance。

### RQ3：困难样本收益与分布偏移损失能否分离？

运行受控 2×2 factorial decomposition：

| | Reference counts | Shifted counts |
|---|---:|---:|
| Random within group | \(R_{ref}\) | \(R_{shift}\) |
| Top uncertainty within group | \(U_{ref}\) | \(U_{shift}\) |

将其称为 controlled factorial decomposition。只有在 trajectory 或样本分配经过随机化时，才使用 causal effect 语言。

### RQ4：SISS 是否由 selector state 驱动？

- 分类：向 logits 注入 class bias \(\beta_k\)；
- DPO：向 pair score 注入 length bias \(\gamma\)；
- 观察 intervention strength、dependence 和 acquisition shift 的剂量关系。

### RQ5：DGA 是否保留信息价值？

同时报告：

- distribution distortion；
- acquisition utility；
- downstream average performance；
- worst-group performance；
- pilot label cost；
- infeasibility slack。

## 8. Baselines

### 8.1 分类

最低配置：

- Random；
- Entropy；
- Margin；
- CoreSet；
- BADGE；
- GALAXY；
- Oracle-DGA；
- Practical-DGA。

CRC-error-mass、NS-error-mass 和 Defer-kcenter 放入“legacy heuristics”消融，不进入主 baseline 表。

### 8.2 DPO

最低配置：

- Random；
- uncertainty based on symmetric probability；
- ActiveDPO；
- implicit reward-gap difficulty selection；
- margin/noise-aware preference selection；
- Oracle-DGA；
- Practical-DGA。

如果无法完整复现所有近期方法，应明确区分 selection score reproduction 和完整 online acquisition reproduction。

## 9. 文献定位与差异

### 9.1 Active learning sampling bias

Farquhar, Gal, and Rainforth（arXiv:2101.11665）形式化了 active learning 使训练数据偏离 population distribution 的统计偏差，并讨论其有害或有益条件。Prabhu, Dognin, and Singh（arXiv:1909.09389）则给出深度文本分类中 entropy sampling 对 sampling bias 可能较鲁棒的反例。Krishnan et al.（arXiv:2109.06321）进一步从鲁棒 active learning 角度直接缓解 sampling bias。该文献脉络说明“active selection 会改变采样分布”本身不是本文的新颖性；SISS 必须研究更具体的条件：何时学生模型的 score 与任务 strata 纠缠、这种纠缠是否进入教师监督集、以及它是否进一步传递为蒸馏后的子群行为变化。

本文区别在于：研究对象不是无偏风险估计本身，而是学生模型参与选择后，监督分布如何改变并传递为蒸馏后的子群行为。

### 9.2 Label shift correction

MALLS（arXiv:2007.08479）处理 source/target label proportions 不同的外生 label shift，并在 class-balanced sampling 与 importance weighting variance 之间折中。SISS 是由 selector 在同一个 pool 内内生制造的 acquisition shift，并可作用于非标签属性和 DPO strata。

### 9.3 Uncertainty and diversity acquisition

BADGE（arXiv:1906.03671）在梯度嵌入中同时考虑不确定性和多样性；GALAXY 使用图结构改善极端 active learning 的类别覆盖。这些方法是必要 baseline，但它们优化 representation coverage 或 batch diversity，并不显式限制 selector-induced task-stratum shift。

### 9.4 Distillation data selection

现有 KD data selection 工作通常关注教师标签质量、噪声过滤、样本价值或动态课程。例如 Improve Knowledge Distillation via Label Revision and Data Selection（arXiv:2404.03693）联合处理标签修订和样本选择。本文的差异不是提出另一个 usefulness score，而是审计并约束由学生模型自身产生的监督分布失真。

### 9.5 Preference data selection

ActiveDPO（arXiv:2505.19241）明确让待对齐模型参数化 active selection，是最直接的对照。Less is More（arXiv:2502.14560）、Principled Data Selection for Alignment（arXiv:2502.09650）和 Difficulty-Based Preference Data Selection by DPO Implicit Reward Gap（arXiv:2508.04149）分别从噪声、模型能力匹配和 reward-gap difficulty 研究数据选择。

本文的补充问题是：这些 model-dependent selection scores 是否系统性改变 length、prompt、preference direction 或 quality-gap 分布，以及该变化是否牺牲某些能力。Random Is Hard to Beat（arXiv:2604.02766）报告 online DPO 中 uncertainty selection 难以稳定超过 Random，且 proxy win-rate 上升可能伴随通用能力下降，这为 capability preservation 指标提供直接动机。

### 9.6 Pairwise scorer bias

Wang et al. 的 *Large Language Models are not Fair Evaluators*（arXiv:2305.17926）和 Zheng et al. 的 *Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena*（arXiv:2306.05685）都报告了 pairwise evaluator 的位置偏置。Park et al. 的 *Disentangling Length from Quality in Direct Preference Optimization*（arXiv:2403.19159）进一步说明长度与 preference optimization 的纠缠。这些工作解释了 DPO 中 score--stratum dependence 的潜在来源，但没有替代本文对 acquisition distribution 和 downstream transmission 的完整测量。

### 9.7 Finite-population estimation

Confidence Sequences for Sampling Without Replacement（arXiv:2006.04347）为 finite pool pilot sampling 和自适应停止提供统计工具。它是 Practical-DGA 的 composition estimator，而不是论文的 acquisition novelty。

### 9.8 新颖性压力测试

最接近的已有工作与本文不能混淆：

| 已有研究线 | 已经解决的问题 | 本文仍需建立的缺口 | 必要证据 |
|---|---|---|---|
| Active-learning sampling bias | 主动采样会改变训练分布，并影响风险估计或鲁棒性 | 当前学生模型如何内生地产生 task-stratum supervision shift，并把偏移传递给蒸馏后的同一学生 | selector intervention、acquisition shift、downstream transmission |
| Label-shift correction / MALLS | 已知或可估计的 source--target label shift 下如何重采样和加权 | 不预设外部 target shift，约束同一 pool 中由 acquisition rule 新产生的异常偏移，并扩展到非标签 strata | Random-equivalent guard、multiview constraints |
| BADGE / GALAXY / diversity AL | 用不确定性、梯度或图结构提高 batch 信息量和覆盖 | 信息量优化是否会牺牲任务子群覆盖，以及如何显式控制这一副作用 | utility--distortion Pareto、worst-group coverage |
| KD data selection | 哪些样本更值得教师修订、标注或蒸馏 | 学生基于自身置信度选监督时，选择器偏置如何成为蒸馏数据偏差 | student-dependent selection、fixed-pool controls |
| Preference data selection / ActiveDPO | 哪些偏好对更难、更可靠或更匹配当前策略 | model-dependent pair selection 是否系统性改变长度、prompt、偏好方向和质量差分布 | dual-order scoring、attribute shift、capability preservation |

因此，论文不能把贡献写成“首次发现主动选择有 sampling bias”，也不能只展示 acquisition counts。AAAI 版本成立至少需要同时满足三点：跨二分类、多分类和 DPO 的统一测量；从监督构成到训练后行为的传递证据；DGA 相对 Random 和强 active-learning baselines 的 utility--distortion 优势。

## 10. 预期贡献列表

建议最终贡献控制在四条：

1. **Phenomenon**：定义 selector-induced supervision shift，并提出跨 score dependence、acquisition shift 和 behavior transmission 的统一测量链；
2. **Evidence**：在二分类、多分类和 preference optimization 中系统验证该现象，并通过 factorial controls 与 selector interventions 分离 difficulty value 和 distribution distortion；
3. **Method**：提出 DGA，将跨 stratum distribution guard 与层内 informativeness optimization 解耦；
4. **Practical audit**：提供 Random-calibrated diagnostics 和 pilot-based latent composition estimation，量化 guard 可行性、标注成本和剩余风险。

不要将 CRC certification 列为第五个核心贡献。它可以作为独立、正交的 deployment guarantee 放入附录。

## 11. AAAI 长文结构

### 1. Introduction

- 模型驱动数据选择越来越常见；
- 核心默认假设：高 uncertainty 只表示信息价值；
- 反例：uncertainty 同时编码模型已有的任务子群偏置；
- 提出 SISS、统一证据和 DGA；
- 四条贡献。

### 2. Related Work

按问题组织，而不是逐篇罗列：

- active learning sampling bias；
- data selection for distillation；
- label shift and distribution correction；
- preference data selection and evaluator bias。

### 3. Selector-Induced Supervision Shift

- 问题定义；
- 三段测量链；
- adjusted dependence；
- acquisition shift curve 和 Random envelope；
- behavior transmission。

### 4. Distribution-Guarded Acquisition

- 两层原则；
- Oracle-DGA；
- Practical-DGA；
- pilot estimation；
- optimization and infeasibility slack；
- PCSS special case。

### 5. Binary Discovery and Mechanism

压缩旧 EMNLP 的二分类实验，重点保留：

- 双向 shift；
- label composition 与 prediction behavior；
- 2×2 分解；
- selector intervention；
- Oracle-DGA。

### 6. Multiclass Generalization

AG News + DBpedia-14：

- class-wise dependence；
- shift curves；
- worst-class coverage；
- LoRA downstream behavior；
- DGA results。

### 7. Preference Alignment

HelpSteer2：

- dual-order reliability；
- attribute shifts；
- Random-DPO vs uncertainty-DPO vs DGA；
- general capability preservation。

### 8. Discussion and Limitations

- task strata 的定义依赖审计目标；
- reference distribution 不一定是最优训练 distribution；
- pilot cost；
- intersectional groups；
- scorer calibration；
- online adaptive feedback loops。

## 12. 主图和主表

### Figure 1：统一因果链

三列展示分类、多分类和 DPO：模型偏置如何进入 score、selected supervision 和训练后行为。

### Figure 2：Shift curves

AG News/DBpedia 不同 budget 下 Random envelope、Entropy、Margin、DGA 的 TV 曲线和 worst-group coverage。

### Figure 3：Controlled factorial decomposition

difficulty、distribution 和 interaction 在多个数据集上的效果。

### Figure 4：DPO attribute transmission

选择前后 length、verbosity、prompt cluster 与训练后输出属性变化。

### Table 1：跨任务现象汇总

每个任务报告 dependence、AAS、worst coverage、downstream gap。

### Table 2：方法比较

Random、uncertainty、diversity、DGA 的 average、worst-group、utility、pilot cost。

### Table 3：DPO

preference metric、general capability、length/verbosity、order disagreement。

## 13. 实验完成顺序

### Phase A：多分类现象

1. 完成 AG News 全量 scoring；
2. 运行 calibrated diagnostics；
3. gate 通过后训练 Random 与选定 uncertainty LoRA；
4. 运行 2×2 decomposition；
5. 实现 Oracle-DGA 并训练；
6. 在 DBpedia-14 复现。

停止条件：如果 AG News 和 DBpedia 都没有超过 Random envelope，则不能将多分类写成已建立现象，需要回到 score、模型或任务 strata 定义。

### Phase B：DPO 现象

1. 修复或替换当前位置偏置严重的 pairwise scorer；
2. 完成 HelpSteer2 dual-order scoring；
3. 诊断 length、attributes、preference direction 和 prompt shift；
4. 只有在 scorer reliable 且至少两个非 order domains 越过 Random envelope 时才训练 DPO；
5. 报告 preference 与 general capability；
6. 实现 Oracle-DGA。

### Phase C：Practical-DGA

1. 设计 observable cells；
2. 离线模拟 pilot reveal；
3. 比较固定 pilot、adaptive pilot 和 oracle composition；
4. 报告 utility--distortion--label cost 三维权衡；
5. 最后再决定 WSR 是否进入主文。

## 14. Go/No-Go 决策

### 可以继续 AAAI 主线的最低证据

- 二分类：已有；
- 多分类：至少两个数据集中的一个稳定越过 Random envelope，另一个提供支持性结果；
- downstream：shift direction 与 group behavior direction 对应；
- DGA：至少在多分类上改善 worst-group 且不明显损失 average performance；
- DPO：若现象不稳定，可以作为重要 negative result，但必须与 capability preservation 共同解释。

### 需要收缩论文的情况

- 多分类没有稳定 shift：论文收缩为 binary mechanism + diagnostic caution；
- DPO scorer 无法校准：DPO 只作为失败案例和限制，不训练新模型；
- DGA 只复制 Random：方法贡献改为 diagnostic framework，不宣称 acquisition improvement；
- pilot cost 大于训练预算：Practical-DGA 不进入主文。

## 15. 与旧 EMNLP 稿件的映射

| 旧内容 | AAAI 中的新角色 |
|---|---|
| CRC threshold | 二分类 probe 或附录 deployment certification |
| \(\varepsilon_{ent}\) | 二分类可视化，主指标改为 adjusted dependence |
| CRC-error-mass / NS-error-mass | 产生 selector shift 的 legacy heuristics |
| Defer-kcenter | 失败分析或消融 |
| PCSS | DGA 的 binary two-stratum special case |
| guide-set estimate | Practical-DGA pilot composition estimation |
| true label proportion equality | Random-equivalent distribution guard |
| label-ratio intervention | controlled factorial decomposition 的一部分 |

## 16. 摘要骨架

1. 模型驱动的数据获取常用小模型置信度决定教师查询；
2. 现有方法把 uncertainty 当成纯信息价值，忽略其与模型偏置和任务 strata 的纠缠；
3. 定义 SISS 并提出三阶段测量链；
4. 在 binary、multiclass、DPO 中验证；
5. 提出 DGA，两层解耦 budget guard 与 within-stratum utility；
6. 报告 average、worst-group、utility 和 cost；
7. 给出不夸大的结论：限制 selector bias 进入监督分布，而非消除模型偏见。

## 17. 参考文献起点

- Farquhar, Gal, and Rainforth. *On Statistical Bias In Active Learning: How and When To Fix It*. arXiv:2101.11665.
- Zhao et al. *Active Learning under Label Shift*. arXiv:2007.08479.
- Ash et al. *Deep Batch Active Learning by Diverse, Uncertain Gradient Lower Bounds*. arXiv:1906.03671.
- Zhang et al. *GALAXY: Graph-based Active Learning at the Extreme*. ICML 2022, PMLR 162.
- Waudby-Smith and Ramdas. *Confidence Sequences for Sampling Without Replacement*. arXiv:2006.04347.
- Lan et al. *Improve Knowledge Distillation via Label Revision and Data Selection*. arXiv:2404.03693.
- Lin et al. *ActiveDPO: Active Direct Preference Optimization for Sample-Efficient Alignment*. arXiv:2505.19241.
- Deng et al. *Less is More: Improving LLM Alignment via Preference Data Selection*. arXiv:2502.14560.
- Gao et al. *Principled Data Selection for Alignment: The Hidden Risks of Difficult Examples*. arXiv:2502.09650.
- Qi, Xu, and Jin. *Difficulty-Based Preference Data Selection by DPO Implicit Reward Gap*. arXiv:2508.04149.
- Oh et al. *Random Is Hard to Beat: Active Selection in online DPO with Modern LLMs*. arXiv:2604.02766.
- Wang et al. *Large Language Models are not Fair Evaluators*. arXiv:2305.17926.
- Zheng et al. *Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena*. arXiv:2306.05685.
- Park et al. *Disentangling Length from Quality in Direct Preference Optimization*. arXiv:2403.19159.

## 18. 当前最重要的写作纪律

- 在全量多分类和 DPO 结果完成前，摘要不得写“across classification and preference alignment we show”；应写“we investigate”或保留占位。
- 所有“significant”必须对应随机包络、重复实验或置信区间。
- 所有“causal”必须对应明确干预或随机化。
- 不把自定义指标包装成已有标准指标。
- 不把 negative result 隐藏；Random 很强本身就是论文关于 guard 必要性的组成部分。
- 主文只保留一个现象名和一个方法名，避免 CRC、AED、DGA、PCSS、WSR 同时争夺贡献位置。
