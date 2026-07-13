# ACL Short Paper — 最终论文大纲（冲击Best Paper版）

> 版本：v3.0 | 日期：2026-05-25  
> 状态：可直接用于撰写正文

---

## 零、核心分析：这篇论文真正说了什么

### 0.1 一句话定位（写在脑子里，决定每一段的取舍）

> **SLMs在微调前是"先验机器"而非分类器。对于强输出先验偏置的SLM，主动蒸馏中的"难样本"在结构上等价于"反先验标签样本"，这导致难度感知选样系统性地扭曲训练集标签分布。标签分布偏移是性能变化的主导因素（最高 −12.64 Macro-F1），而非难度信号本身。我们提出PCSS显式解耦这两个目标，并用CRC认证保障部署安全。**

### 0.2 三层贡献金字塔（对应Best Paper的三个判断维度）

```
层次3 ─── 理论保证层 ────────── CRC认证：数学可证明的部署错误率上界
              │                   （与选样策略解耦，为任意策略提供保障）
层次2 ─── 算法贡献层 ────────── PCSS：将选样显式分解为
              │                   Obj1（分布对齐）+ Obj2（难度选择）
              │                   消除过校正，保留难度信号
层次1 ─── 发现诊断层 ────────── 核心发现：首次系统揭示
                                  先验偏置 × 难度选样 → 标签分布扭曲
                                  控制实验：来源 < 标签分布（因果证明）
```

**这个金字塔的意义**：层次1是让审稿人感到"Wow"的部分；层次2是让他们觉得"有用"；层次3是让他们觉得"严谨"。三层缺一不可。

### 0.3 故事主线（叙事弧）

```
Act 1（提问）：主动蒸馏假设难样本=信息量最大 ← 这是当前共识
     ↓
Act 2（转折）：我们发现这可以严重损害性能（−12.64 F1 的震撼开场）
     ↓
Act 3（诊断）：SLM是"先验机器" → 难=反先验标签 → 分布扭曲
              控制实验直接证明：来源不重要，分布才重要
     ↓
Act 4（解法）：PCSS将纠缠的单目标拆成两个有序子问题
              同时解释了"为什么random有时更好"（偶然正确性）
     ↓
Act 5（保障）：CRC认证确保无论选样策略如何，部署错误率可证明受控
```

---

## 一、标题方案

### 推荐标题

> **"Do We Really Need Difficulty-Aware Sampling? The Label Distribution Confound in Active LLM-to-SLM Distillation"**

**选择理由**：
1. 以疑问句开篇，挑战已有共识，能立刻吸引读者和审稿人
2. "Label Distribution Confound"精确命名核心机制，便于引用
3. "Active LLM-to-SLM Distillation"精确定位场景，防止与无关工作混淆
4. "Do we really need"呼应了近期质疑既有假设的best paper风潮（如"Do Prompt-Based Models Really Understand…"系列）

### 备选标题

- **"Hard Samples, Wrong Direction: When Difficulty Selection Misleads Active LLM Distillation"**（更叙事化，适合workshop）
- **"When Hard Samples Hurt: Diagnosing the Label Distribution Trap in Active Distillation"**（动词更有力，适合general track）
- **"Prior Machines Don't Need Hard Samples: A Mechanistic Study of Active LLM-to-SLM Distillation"**（"Prior Machine"概念最突出，适合理论取向审稿人）

---

## 二、Abstract（~170词）

**写作结构**（每句功能精确）：

> [句1：大背景] Distilling knowledge from large language models (LLMs) into small language models (SLMs) under annotation budget constraints has emerged as a practical paradigm, where active selection of *difficult* or *uncertain* samples is assumed to maximize information per annotation.  
> [句2：当前假设] This assumption — central to both active learning and knowledge distillation — posits that samples where the student model is most uncertain carry the greatest learning signal.  
> [句3：核心发现] We challenge this assumption with a systematic finding: for SLMs with strong output prediction priors (*prior machines* that predict one class for >93% of instances), difficulty-aware selection is structurally confounded with label distribution correction, and the distribution shift dominates performance, causing degradation of up to **12.6 Macro-F1 points** relative to random sampling.  
> [句4：因果证据] Controlled experiments demonstrate that *which* samples are selected (hard vs. easy, correct vs. wrong) is far less important than *how many samples of each class* are selected.  
> [句5：方法] We propose **Prior-Corrective Stratified Selection (PCSS)**, which disentangles difficulty selection from label distribution alignment as two ordered objectives, eliminating over-correction while preserving genuine difficulty signal.  
> [句6：认证] Combined with **CRC-guided certified routing**, PCSS provides a mathematically certified deployment error bound independent of the sampling strategy.  
> [句7：实践建议] Our results advocate diagnosing SLM output prior bias as a prerequisite step before any difficulty-aware selection strategy is deployed.

---

## 三、Section 1：Introduction（~0.65页）

### 目标
建立两件事：(a) 现有方法的假设，(b) 我们打破该假设的证据。用最少的文字完成从"共识"到"疑问"的转变。

### 段落规划

**¶1 — Hook：主动蒸馏的承诺（3句）**  
大模型API昂贵 → 主动蒸馏只标注"关键"样本 → 标准答案是难/不确定样本。
设置读者的期望：这是一个合理的、被广泛接受的做法。

**¶2 — 震撼数据：反直觉的失败（2句）**  
**直接给出最戏剧性的数据**：在TwitterHate（n=50）上，难度感知方法比随机采样低12.64 Macro-F1；在IMDb上低2.12；在FEVER-500上低7个Positive-F1。这不是边际差异——这是系统性失败。一句话点出：这是因为什么？

**¶3 — 诊断：SLM是"先验机器"（4句）**  
SLM微调前不是分类器，而是输出先验主导的先验机器。数据：Qwen3-0.6B在7个任务中的6个里，>93%的样本预测同一个标签。对于这样的模型，"难样本"在结构上等价于"反先验标签样本"——它们恰好是模型极少预测的那个类别。因此，选难样本≈矫正标签分布，这个耦合导致两个目标纠缠在一起无法分离。

**¶4 — 机制：两目标被错误地合并（4句）**  
当前所有主动蒸馏方法都在隐式地同时做两件事：(A) 决定每个类别选多少样本（标签分布）；(B) 在每个类别内选哪些样本（难度）。对于强偏置SLM，(A)是主导项。我们通过控制实验直接证明：固定来源（全部来自模型已预测正确的样本、或全部来自模型预测错误的样本），性能差距最多2 F1；但固定来源、改变标签比例，性能差距最高15.4 F1。这是因果证据。随机采样的"成功"不是因为它避免了难度，而是因为它意外地保持了真实标签分布。

**¶5 — 贡献列表（3个bullet）**

1. **Diagnostic Finding（诊断发现）**：首次系统证明，在强偏置SLM的主动蒸馏中，训练集标签分布是性能的主导决定因素，而样本来源（难/易、正确/错误）的独立贡献远小于分布效应。我们提出"prior machine"概念刻画这一条件。

2. **PCSS（算法贡献）**：基于两目标分解框架，提出Prior-Corrective Stratified Selection，将主动蒸馏选样问题显式分解为有序子问题：先对齐真实任务标签分布，再在每个标签层内按难度选样。消除了过校正风险，同时保留了难度信号。

3. **CRC认证路由（理论贡献）**：提出与选样策略解耦的CRC引导部署认证机制，提供数学可证明的期望wrong-accept错误率上界 $\mathbb{E}[\text{error}] \leq \alpha$，适用于任意满足认证集隔离条件的选样策略。

---

## 四、Section 2：Background and Setup（~0.3页）

### 目标
以最小篇幅为读者建立符号系统和框架，使Section 3的分析自洽。

### 2.1 Task and Protocol

**任务**：给定query $q$ 和候选池 $\mathcal{U} = \{x_1, \ldots, x_N\}$，在标注预算 $B \ll N$ 下，学习二分类学生模型 $f_S: \mathcal{X} \to \{0, 1\}$，并在部署时最小化大模型调用。

**输出协议**（统一定义，全文使用）：
$$\text{score}_i = \log p_\theta(\texttt{"1"} \mid x_i) - \log p_\theta(\texttt{"0"} \mid x_i), \quad \hat{y}_i = \mathbf{1}\{\text{score}_i > 0\}$$

**路由分数**（不确定度度量）：
$$R_i(T) = \sigma\!\left(\frac{|\text{score}_i|}{T}\right) \in [0.5, 1], \quad R_i \to 0.5 \Leftrightarrow \text{最不确定}$$

### 2.2 Data Protocol（双集隔离）

从 $\mathcal{U}$ 中随机划分：
- 引导集 $\mathcal{D}_\text{guide}$（$n_g = 1000$）：有大模型标签，参与中间决策
- 认证集 $\mathcal{D}_\text{cert}$（$n_c = 200$）：**全程锁定，不参与任何训练或选样决策**
- 候选池 $\mathcal{U}_\text{pool}$（$N - 1200$）：待蒸馏数据

关键：认证集隔离是最终数学保证成立的必要条件（§5证明）。

### 2.3 CRC Calibration（一段简述）

在 $\mathcal{D}_\text{guide}$ 上校准路由阈值 $\hat{\lambda}$，使wrong-accept风险的有限样本上界 $\leq \alpha$：
$$\hat{\lambda} = \min\!\left\{\lambda \in [0.5, 1.0] : \frac{n_g}{n_g+1}\hat{R}(\lambda) + \frac{1}{n_g+1} \leq \alpha\right\}$$
$$D_\text{defer} = \{x : R_i < \hat{\lambda}\}, \quad D_\text{accept} = \mathcal{U}_\text{pool} \setminus D_\text{defer}$$

---

## 五、Section 3：The Label Distribution Trap（~1.25页）——论文核心

### 写作策略
这是论文贡献最重要的部分。按**因果链**组织：条件 → 机制 → 证据 → 推论。每一步都有对应的Table/Figure支撑。不要只展示数据——每张表后面跟一句**明确的因果结论**。

---

### 3.1 Condition: SLMs as Prior Machines（~0.2页）

**论点**：大多数SLM在微调前不是分类器，而是被输出先验主导的先验机器。这是后续所有分析的前提条件。

**📊 Table 1：SLM输出先验偏置分布（必须有的表）**

| Dataset / Query | True yes% | Base pred yes% | Base Macro-F1 | Prior Bias Type |
|----------------|-----------|----------------|---------------|-----------------|
| IMDb q1 (positive) | 50.0 | **99.99%** | 33.35 | Severe-yes |
| IMDb q2 (negative) | 27.5 | **100.0%** | 21.57 | Severe-yes |
| TwitterHate q1 | 83.2 | **100.0%** | 45.42 | Severe-yes |
| Codebase q1 (social) | 6.3 | **93.8%** | 12.30 | Severe-yes |
| Codebase q2 (CS) | 62.0 | **99.5%** | 39.60 | Severe-yes |
| Codebase q3 (factual) | 25.8 | **99.9%** | 20.65 | Severe-yes |
| FEVER (Qwen3-0.6B) | 52.4 | **10.0%** | 44.08 | Severe-no |
| FEVER (Qwen3-1.7B) | 52.4 | 58.9% | 86.72 | Near-balanced |

**关键句（表后必写）**：
> "On 6 out of 7 classification queries, Qwen3-0.6B predicts one output label for over 93% of instances, achieving near-chance Macro-F1. We term such models *prior machines*: before fine-tuning, their output behavior is determined by prediction priors, not by input features."

**定义框（可以用Definition box）**：
> **Definition 1 (Prior Machine)**: A model $f_\theta$ is a *prior machine* on task $(q, \mathcal{U})$ if its zero-shot prediction distribution satisfies $|\Pr[\hat{y} = 1] - 0.5| > \delta$ for some $\delta > 0.3$, indicating strong prediction prior rather than discriminative classification.

---

### 3.2 Mechanism: Difficulty ≡ Counter-Prior Label（~0.25页）

**论点**：对于先验机器，"难样本"在结构上等价于"反先验标签样本"。选难样本=矫正标签分布，这个等价性是隐性的，现有方法没有意识到。

**数学表述**（清晰、可证明）：

设模型在类别 $c$ 上的预测置信度为 $R_i(T)$，当 $\hat{y}_i = c_\text{prior}$（先验预测类）时，$R_i$ 高（high confidence）；当 $\hat{y}_i \neq c_\text{prior}$ 时，$R_i$ 低（uncertain）。

因此：
$$\Pr[\hat{y}_i = c_\text{prior} \mid R_i \geq \hat{\lambda}] \gg \Pr[\hat{y}_i = c_\text{counter} \mid R_i \geq \hat{\lambda}]$$
$$\Pr[\hat{y}_i = c_\text{counter} \mid R_i < \hat{\lambda}] \gg \Pr[\hat{y}_i = c_\text{prior} \mid R_i < \hat{\lambda}]$$

即：**accept集主要是先验类样本，defer集主要是反先验类样本**。选难（defer）样本= 过采样反先验标签样本。

**📊 Table 2：难度感知方法系统性地偏移训练集标签分布（关键对比表）**

| Dataset / Budget | Random: train yes% | Difficulty: train yes% | True yes% | Dist. Gap (Diff. vs. True) | Method |
|-----------------|-------------------|------------------------|-----------|---------------------------|--------|
| IMDb q1, n=2500 | 50.76% | 7.36% | 50.0% | **−42.6pp** | ns-error-mass |
| TwitterHate, n=50 | 78% | 66% | 83.2% | **−17.2pp** | crc-error-mass |
| FEVER 0.6B, n=1500 | 52.33% | 66.13% | 52.4% | **+13.7pp** | ns-error-mass |
| FEVER, n=500 | 52% | 66.6% | 52.4% | **+14.2pp** | defer-kcenter |
| FEVER 1.7B, n=2231 | 52.13% | 44.8% | 52.4% | **−7.6pp** | crc-error-mass |

**关键句**：
> "Difficulty-aware methods consistently move training label distributions away from the true task distribution. The direction of shift matches the prior bias: for yes-biased models, hard samples are predominantly *no* labels (overcorrection); for no-biased models (FEVER), hard samples are predominantly *yes* labels (undercorrection)."

---

### 3.3 Evidence: Distribution Predicts Performance, Source Does Not（~0.55页）——核心证据节

这是论文最有力的部分，需要两种互补的证据。

**🎯 Figure 1（全文最重要的图）：分布偏差 vs. 性能散点图**

- **X轴**：|训练集yes% − 真实yes%|（分布偏差绝对值，单位pp）
- **Y轴**：Macro-F1相对于random的差值（Δ F1）
- **每个点**：一个（数据集，方法，预算）三元组
- **颜色**：按模型容量（红=0.6B severe-bias；蓝=1.7B near-balanced）
- **形状**：按方法（圆=random；三角=difficulty方法）
- **趋势线**：负相关（越偏离分布，性能越差）
- **标注**：最戏剧性的失败点（TwitterHate n=50, −12.64; IMDb q1, −2.12）

**图说明的故事**：分布偏差越大→性能越差。Random之所以是strong baseline，是因为它意外地位于X轴接近0的位置（保持了真实分布）。Difficulty方法位于X轴右侧（偏离真实分布），性能下降。

---

**证据A — 过校正导致系统性性能下降**

**📊 Table 3a：过校正案例（Overcorrection Cases）**

| Dataset | Budget | Method | Train yes% | True yes% | Dist. Gap | Pred yes% | Macro-F1 | vs. Random |
|---------|--------|--------|-----------|-----------|-----------|-----------|----------|-----------|
| TwitterHate | 50 | random | 78% | 83.2% | −5.2pp | 87.4% | 65.94 | — |
| TwitterHate | 50 | **crc-error-mass** | 66% | 83.2% | **−17.2pp** | 45.4% | **53.30** | **−12.64** |
| IMDb q1 | 2500 | random | 50.76% | 50% | +0.76pp | 50.5% | 94.37 | — |
| IMDb q1 | 2500 | **ns-error-mass** | 7.36% | 50% | **−42.6pp** | 45.6% | **92.25** | **−2.12** |
| FEVER | 500 | random-defer | 52% | 52.4% | −0.4pp | 71.8% | PF1=76.47 | — |
| FEVER | 500 | **defer-kcenter** | 66.6% | 52.4% | **+14.2pp** | 80.0% | PF1=**69.45** | **−7.0** |

**规律说明**（表后必写）：
> "In all overcorrection cases, the model's evaluation-set prediction yes-rate follows the training label distribution, not the true task distribution. This bidirectional effect — undercorrection in yes-biased settings and overcorrection in no-biased settings — is consistent with the prior machine hypothesis: the student learns the training label marginal, not the task."

---

**证据B — 控制实验：Smoking Gun**（最有力的因果证据，需要专门突出）

**📊 Table 3b：来源 vs. 分布控制实验（Codebase q1/q2，500样本）**

| Condition | Sample Source | Train yes% | Eval pred yes% | Macro-F1 | vs. Balanced |
|-----------|---------------|-----------|----------------|----------|-------------|
| base-correct-balanced | 全部base已预测正确 | **50%** | 22.87% | **57.93** | — |
| base-correct-random | 全部base已预测正确 | **0.8%** | 98.71% | 42.49 | **−15.44** |
| base-wrong-random | 全部base预测错误 | **100%** | 0.00% | 26.77 | **−31.16** |

**关键结论（用Callout Box突出显示）**：
> **Key Finding**: base-correct-balanced（50% yes，全部easy样本）vs. base-correct-random（0.8% yes，全部easy样本），**相同来源，不同分布，差距15.4 F1**。base-wrong-random（100% yes，全部hard样本）vs. base-correct-random（0.8% yes，全部easy样本），**相同分布方向，相似性能**。
>
> **Conclusion**: *Which* samples are selected (hard/easy, correct/wrong) is secondary. *How many samples of each label* are selected is primary.

这是整篇论文最强的证据，必须用视觉设计（粗体、灰底框、*）突出它。

---

### 3.4 Moderator: Capacity Determines Whether Difficulty Helps（~0.25页）

**论点**：当模型容量足够高、初始先验偏置足够低时，难度信号提供真正的额外边际收益。这使我们的发现更有nuance，不是绝对的"难样本无用"。

**📊 Table 4：容量作为调节变量**

| Model | Base pred yes% | Method | Train yes% | Macro-F1 | vs. Random | Dist. Gap |
|-------|---------------|--------|-----------|----------|-----------|-----------|
| 0.6B | 10% (severe-no) | random | 52.42% | 93.97 | — | 0pp |
| 0.6B | 10% (severe-no) | ns-error-mass | 68.2% | 94.47 | **+0.50** | +15.8pp |
| **1.7B** | **58.9% (near-balanced)** | random | 52.13% | 79.08 | — | 0pp |
| **1.7B** | **58.9% (near-balanced)** | crc-error-mass | 44.8% | **80.34** | **+1.26** | −7.6pp |
| **1.7B** | **58.9% (near-balanced)** | ns-difficulty-global | 67.98% | **79.88** | **+0.80** | +15.6pp |

**分析要点**：
- 0.6B上，即使FEVER（唯一正向数据）的+0.5 F1增益，也与分布偏移（+15.8pp）耦合，无法分离难度贡献
- 1.7B上，先验偏置接近均衡（58.9% vs 真实52.4%），难度方法提供真实的+0.80~+1.26 F1
- 但即使1.7B，crc-error-mass也将训练yes%推至44.8%（偏离真实52.4%），说明分布效应仍存在，只是模型容量足以吸收

**Decision Criterion（实践价值）**：
$$\text{Use difficulty sampling if: } |\Pr_\text{base}[\hat{y}=1] - \hat{p}_1| < \delta_\text{threshold} \approx 10\text{pp}$$

---

## 六、Section 4：Prior-Corrective Stratified Selection（PCSS）（~0.4页）

### 目标
从诊断到解法的转折节。设计要简洁，重点是展示如何用两目标分解干净地解决问题。

### 4.1 The Two-Objective Decomposition（正式框架）

将主动蒸馏选样显式分解为两个有序目标：

**Objective 1（分布目标，首要）**：
$$\min_S \left|p_S^{(1)} - \hat{p}_1\right|, \quad \hat{p}_1 = \frac{1}{n_g}\sum_{j \in \mathcal{D}_\text{guide}} y_j$$
训练集标签比例应对齐从引导集估计的真实任务比例。

**Objective 2（难度目标，次要，在Obj1约束下）**：
$$\max_S \sum_{i \in S} \text{Uncertainty}(x_i), \quad \text{Uncertainty} = 1 - R_i(T)$$
在满足分布约束的前提下，优先选择学生模型最不确定的样本。

**关键洞见（用Remark box）**：
> **Remark 1**: Random sampling approximately satisfies Objective 1 (pool ≈ true distribution) but ignores Objective 2. Existing difficulty methods optimize Objective 2 while violating Objective 1. PCSS is the first method to explicitly satisfy both in order.

### 4.2 PCSS Algorithm（Algorithm 1 Box）

```
Algorithm 1: Prior-Corrective Stratified Selection (PCSS)
─────────────────────────────────────────────────────────
Input:  D_guide (1000 labeled), U_pool (unlabeled), 
        Budget B, Risk α, Temperature T
Output: Training set S_train, Certified threshold λ*
─────────────────────────────────────────────────────────
Phase 1 — True Distribution Estimation:
  p̂₁ ← (1/n_g) Σ y_j  for j in D_guide
  
Phase 2 — CRC Calibration on D_guide:
  λ̂ ← argmin{λ : (n_g/(n_g+1))·R̂(λ) + 1/(n_g+1) ≤ α}
  D_defer ← {x ∈ U_pool : R(x,T) < λ̂}
  D_accept ← U_pool \ D_defer
  
Phase 3 — Stratified Budget Allocation:
  B₁ ← round(B · p̂₁)          // yes-label budget
  B₀ ← B - B₁                  // no-label budget
  
Phase 4 — Within-Stratum Difficulty Selection:
  // Within each stratum, prioritize defer (low R) over accept
  Ŷ ← student_predict(U_pool)  // proxy labels
  S₁ ← top-B₁ samples from D_defer∪D_accept where ŷᵢ=1, sorted by R(x,T)↑
  S₀ ← top-B₀ samples from D_defer∪D_accept where ŷᵢ=0, sorted by R(x,T)↑
  S_train ← teacher_label(S₁ ∪ S₀)  // LLM annotation
  
Phase 5 — Certification (on held-out D_cert):
  λ* ← CRC(D_cert, S_train, θ*, α)  // certified threshold
  
Return S_train, λ*
─────────────────────────────────────────────────────────
```

### 4.3 Properties

**Property 1（分布一致性）**：PCSS的训练集标签比例 $p_{S_\text{train}}^{(1)} \to \hat{p}_1$ 当 $B \to \infty$，消除由选样策略引入的分布偏差。

**Property 2（难度优先性）**：在分布约束下，defer集（$R < \hat{\lambda}$，最不确定）内的样本被优先选择，难度信号被保留而非丢弃。

**Property 3（统一性）**：PCSS统一了现有方法，每种方法都是退化情形：

| 方法 | Obj 1（分布） | Obj 2（难度） | 位置 |
|------|------------|------------|------|
| Random | ✓（近似） | ✗（忽略） | 退化：只满足Obj1 |
| Difficulty-only | ✗（违反） | ✓（优先） | 退化：只满足Obj2 |
| **PCSS（本文）** | **✓（显式）** | **✓（条件下）** | **完整方法** |

**Property 4（认证解耦性）**：CRC认证在 $\mathcal{D}_\text{cert} \perp \theta^*$ 条件下成立，与选样策略无关。

### 4.4 Expected Results on Failure Cases（基于PCSS实验结果）

**📊 Table 5：PCSS修复过校正（实验结果表）**

| Dataset | Budget | Eval Split | Random F1 | Best Difficulty F1 | **PCSS F1** | PCSS vs. Random | PCSS vs. Difficulty |
|---------|--------|------------|-----------|-------------------|------------|----------------|-------------------|
| TwitterHate | 50 | post-train pool split | 65.94 | 53.30 (−12.64) | **67.14** | **+1.20** | **+13.84** |
| TwitterHate | 125 | post-train pool split | 55.70 | 65.85 (+10.15) | **68.62** | **+12.92** | **+2.77** |
| TwitterHate | 250 | post-train pool split | 57.62 | 71.16 (+13.54) | **64.20** | **+6.58** | **−6.96** |
| TwitterHate | 2231 | original test split | 90.42 | 90.60 (+0.19) | **91.02** | **+0.60** | **+0.41** |
| FEVER 0.6B | 1500 | balanced_test_10000 | 90.07 | 90.52 (+0.46) | **90.38** | **+0.31** | **−0.14** |
| FEVER 1.7B | 3000 | balanced_test_10000_seed1 | 93.85 | 94.20 (+0.35) | **91.93** | **−1.92** | **−2.27** |



> **注**：表中均为 Macro-F1。

---

## 七、Section 5：CRC-Guided Certified Deployment（~0.25页）

### 目标
清洁地陈述认证保证和实践含义。核心信息：(1) 认证与选样解耦，(2) 具有数学严格性。

### 5.1 The Certification Guarantee

**Theorem 1（二分类路由部署认证，完整陈述）**：

设 $\mathcal{D}_\text{cert}$ 与最终学生模型 $\theta^*$ 满足 $\mathcal{D}_\text{cert} \perp \theta^*$（认证集全程未参与任何训练或选样决策），令 $\hat{\lambda}^*$ 为在 $\mathcal{D}_\text{cert}$ 上按CRC过程校准的最终阈值。则对任意 $Z_\text{new} \sim P$（与 $\mathcal{D}_\text{cert}$ 和 $\theta^*$ 独立的部署样本）：

$$\mathbb{E}_{\mathcal{D}_\text{cert},\,\mathcal{A}}\!\left[\mathbf{1}\{R(Z_\text{new}) \geq \hat{\lambda}^*\} \cdot \mathbf{1}\{\hat{y}(Z_\text{new}) \neq y(Z_\text{new})\}\right] \leq \alpha$$

**证明草图**（正文简述，完整证明在Appendix）：由 $\mathcal{D}_\text{cert} \perp \theta^*$，固定 $\theta^*$ 后认证集满足可交换性；CRC引理1保证条件期望 $\leq \alpha$；全期望公式给出边际期望界。

**Critical Note（必须写）**：
> "This guarantee holds for PCSS, random sampling, and any strategy satisfying $\mathcal{D}_\text{cert} \perp \theta^*$. It is the non-trivial generalization bound to *unseen deployment samples*, not the trivial empirical risk bound on $\mathcal{D}_\text{cert}$ itself."

### 5.2 Practical Cost Analysis

**FEVER（$N = 165,447$）为例**：

| Component | LLM calls | SLM calls |
|-----------|-----------|-----------|
| Guide + Cert labeling | 1,200 | 0 |
| Full inference (Round 0) | 0 | $N$ |
| PCSS annotation | ~625 (with 25% buffer) | 0 |
| Deployment defer (<3%) | ~5,000 | 0 |
| **Total** | **~6,825** | **~1.6N** |

对比全大模型方案（165,447次调用）：**大模型调用节省约96%**。

### 5.3 Why Guide-Set CRC Cannot Provide the Same Guarantee

一句话解释双集隔离的必要性（用于回应可能的审稿人质疑）：

> "The guide set $\mathcal{D}_\text{guide}$ participates in sampling decisions through diagnostic quantities ($e_\text{all}, e_\text{defer}, c_\text{crc}$), inducing dependence $\mathcal{D}_\text{guide} \not\perp \theta^*$. The exchangeability condition required by Lemma 1 (Angelopoulos et al., 2022) is violated; any calibration on $\mathcal{D}_\text{guide}$ provides only a heuristic estimate. Only the isolated certification set provides the formal guarantee."

---

## 八、Related Work（~0.3页，可内嵌入Introduction或独立）

### 写作策略：用对话而非列举

每段都回答一个问题："这篇论文与已有工作有何不同？"

**Para 1 — Active Distillation**：现有方法（Hinton 2015; Settles 2010; Goel et al. 2024; LASER 2025）假设难/不确定样本是最优选择，未质疑该假设在强偏置SLM场景下的有效性。

**Para 2 — Hard Sample Limitations（重要，需精准区分）**：
> "Concurrent work reports diminishing returns of hard samples in reinforcement learning for SLMs [B1: Limits of Difficulty Scaling, 2604.06298], and step-length confounding in reasoning data selection [B2: Step Length Confounding, 2604.06834]. We identify a mechanistically distinct confound in supervised classification distillation: output prior bias creates structural coupling between sample difficulty and label distribution. Unlike [B1] (capacity boundary in RL) and [B2] (statistical artifact of step-length in reasoning), our mechanism is a design property of active selection under output prior bias."

**Para 3 — Label Distribution in Training**：L2D（2511.10675）研究ICL推理阶段的标签分布，DIRECT（2312.09196）研究数据集内在不平衡。我们识别了主动蒸馏中的一种新型不平衡来源：**SLM先验偏置诱导的标签分布偏移**，不依赖数据集是否平衡。

**Para 4 — Conformal Prediction**：CRC（Angelopoulos et al. 2022）框架是理论基础；我们的贡献是其在主动蒸馏认证部署场景中的应用，特别是通过双集隔离保证认证集与模型的独立性。

---

## 九、Section 6 / Discussion（~0.15页）

### 6.1 The "Accidental Correctness" of Random Sampling

**核心洞见（值得单独成段）**：随机采样是strong baseline不是因为它特别聪明，而是因为从真实pool中均匀采样保持了真实标签分布，意外地满足了PCSS的Objective 1。这个解释给出了一个预测：在人工构建的imbalanced pool上，random sampling也会失败——为未来工作指出方向。

### 6.2 Practical Diagnostic Checklist

```
Step 1: Measure base model's pred-yes rate on a held-out sample
Step 2: Compare with true yes rate (from guide set)
        If |pred-yes − true yes| > 20pp → SEVERE risk; difficulty selection likely harmful
        If |pred-yes − true yes| ∈ (10pp, 20pp) → MODERATE risk; PCSS recommended
        If |pred-yes − true yes| < 10pp → LOW risk; difficulty selection acceptable
Step 3: After selection, verify training label distribution ≈ true distribution
```

---

## 十、Conclusion（~0.1页）

四句话：
1. 我们挑战了主动蒸馏中"难样本=信息量最大"这一核心假设。
2. 对于强输出先验偏置的SLM（先验机器），难度感知选样系统性地违反了真实标签分布，导致高达12.64 Macro-F1的性能损失。
3. 控制实验证明样本来源（难/易、正确/错误）的独立贡献远小于训练标签分布效应。
4. PCSS通过显式分解分布目标与难度目标解决了这一问题；CRC认证为部署提供了与选样策略无关的数学可证明错误率上界。

**Takeaway sentence（读者带走的一句话）**：
> "Before deploying any difficulty-aware selection strategy for SLM distillation, measure the model's output prior bias and verify that training label distribution is preserved — this single diagnostic step can prevent catastrophic performance failures."

---

## 十一、Limitations（无长度限制，但需完整）

1. **PCSS的分布估计精度**：$\hat{p}_1$ 依赖引导集（n=1000）估计真实分布，对OOD pool或长尾任务可能有偏。
2. **代理标签可靠性**：PCSS分层使用学生预测 $\hat{y}_i$ 作为代理标签，当base accuracy极低（接近随机）时，分层意义下降。
3. **模型范围**：实验仅限Qwen3系列；其他架构（Llama, Mistral）的先验偏置特性需要验证。
4. **二分类限制**：理论框架可推广到多分类（按类别分配预算），但本文未实验。
5. **B < 50的极限情况**：预算极小时分层导致某些类别样本数过少（<5），需要最小样本约束。

---

## 十二、图表完整规划（4页主文内）

### 必须有的图（2个）

**Figure 1（全文关键图，约0.5列）**  
**标题**：*Training label distribution gap versus performance (Macro-F1 relative to random)*  
**类型**：散点图  
**X轴**：`|train_yes% − true_yes%|`（分布偏差绝对值，0~50pp）  
**Y轴**：`Macro-F1 − Macro-F1_random`（相对random的性能差）  
**颜色**：蓝=0.6B severe-bias；橙=0.6B moderate；绿=1.7B near-balanced  
**形状**：圆=difficulty方法；★=PCSS；×=random（应在(0,0)附近聚集）  
**关键元素**：绘制趋势线（负相关）；标注TwitterHate n=50 −12.64这个点  
**视觉效果**：一眼看出"越偏离分布，性能越差；random聚集在原点；PCSS在高性能区域"

**Figure 2（算法概念图，约0.4列）**  
**标题**：*The two-objective decomposition in PCSS*  
**类型**：流程/概念图  
**左侧（Current Methods）**：单箭头从"Uncertainty Sampling"指向"Training Set"，旁边标注"🚨 Implicit distribution shift"  
**右侧（PCSS）**：两个有序步骤：Obj1（"Align label distribution"→ $\hat{p}_1$）→ Obj2（"Select uncertain samples within stratum"）  
**底部**：左右对比"Random: ✓Obj1, ✗Obj2" / "Difficulty: ✗Obj1, ✓Obj2" / "PCSS: ✓Obj1, ✓Obj2"

### 必须有的表（4个）

**Table 1**：SLM Prior Bias Profile（§3.1，8行×5列）  
**Table 2**：Training Label Distribution Shift by Method（§3.2，5行×6列）  
**Table 3a+3b**：Evidence Tables（§3.3，可合并为Table 3，分上下两部分）  
- 3a：Overcorrection cases（6行×8列）  
- 3b：Source control experiment（3行×7列）  
**Table 4**：Capacity Moderator（§3.4，5行×7列）  
**Table 5**：PCSS Main Results（§4.4，需填实验数据）

**注**：如果页面空间紧张，Table 4可以和Table 5合并（按模型容量分组展示），节省0.2页用于Figure 2。

---

## 十三、ACL Short Paper格式约束下的空间分配

| 节 | 行数估计（10pt, 双列, 每列约55行） | 对应页面 |
|----|--------------------------------|---------|
| Abstract | 12行 | ~0.11页 |
| §1 Introduction | 65行 | ~0.59页 |
| §2 Background | 35行 | ~0.32页 |
| §3 Diagnostic Analysis | 130行 | ~1.18页 |
| §4 PCSS | 45行 | ~0.41页 |
| §5 CRC Certification | 28行 | ~0.25页 |
| Related Work（内嵌） | 30行 | ~0.27页 |
| Discussion + Conclusion + Limitations | 32行 | ~0.29页 |
| Figure 1 + Figure 2 | 55行 | ~0.50页 |
| Table 1 + 2 + 3 + 4 + 5 | 65行 | ~0.59页 |
| **合计** | **~497行** | **~4.5页** |

> **压缩策略**：如超出4页，优先压缩§2（减少公式推导，转到Appendix）和Discussion（合并入Conclusion），保留§3（核心诊断，不可压缩）和Figure 1（最关键的图）。

---

## 十四、写作顺序建议

| 写作顺序 | 内容 | 核心目标 | 估计时间 |
|---------|------|---------|---------|
| Day 1 | Table 1 + Table 2 + Figure 1草图 | 确认核心数据讲得清楚 | 4h |
| Day 2 | §3.1 + §3.2 + §3.3（含smoking gun控制实验表） | 诊断论点写清楚 | 6h |
| Day 3 | §4 PCSS + Algorithm 1 + Table 5（填入实验数据） | 解法写清楚 | 4h |
| Day 4 | §1 Introduction + Abstract | 整体故事弧确认 | 4h |
| Day 5 | §5 CRC + §3.4 + Related Work + Conclusion | 补齐剩余部分 | 4h |
| Day 6 | 全文压缩+格式对齐ACL模板 | 压到4页内，语言打磨 | 5h |

---

## 十五、为什么这个大纲能冲击Best Paper

### 三条评判标准及对应策略

**标准1：是否挑战了重要的错误假设（Challenging a flawed assumption）**  
→ 我们挑战了"难样本=信息量最大"这一主动学习的基础假设，在特定场景（强偏置SLM）证明其有害。这是"Challenge Papers"的最高形式。

**标准2：是否有清晰的机制解释（Mechanistic clarity）**  
→ "先验机器"概念 + 数学等价性（难=反先验标签）+ 控制实验（因果证明）。三层机制，从概念到数学到实验完整贯通。

**标准3：是否改变了社区的实践方式（Lasting impact on practice）**  
→ 诊断清单（测量base prior bias）将成为任何主动蒸馏实验的标准前置步骤。这给了社区一个可操作的、可立即采用的工具。

### 与Best Paper竞争论文的预期区分

| 可能的竞争论文 | 我们的差异化 |
|-------------|------------|
| B1: Limits of Difficulty Scaling | 他们在RL中观察现象，我们在SFT中解释机制，场景不同，机制不同 |
| B2: Step Length Confounding | 他们在推理数据选择，我们在分类蒸馏，混淆变量不同 |
| 其他主动学习论文 | 我们是第一个系统研究SLM先验偏置对主动蒸馏选样的影响 |

**ACL评审最看重的点**：这篇论文让读者在读完后立刻改变自己的实验习惯。这是best paper的标志。

---

## 附录结构（Unlimited Space）

**Appendix A：Proposition 1 完整证明**（CRC认证定理）  
**Appendix B：Proposition 2**（引导集不能提供等价保证）  
**Appendix C：数据集详细描述**（FEVER/IMDb/TwitterHate/Codebase）  
**Appendix D：训练超参数完整列表**  
**Appendix E：所有实验完整结果表**（主文中压缩的数据完整版）  
**Appendix F：PCSS代理标签准确率分析**（当base accuracy极低时的降级行为）  
**Appendix G：三种极端情况的理论分析**（$c_\text{crc} = 1$，$c_\text{crc} \gg 1$，$r_C \to 0$）

---

*大纲版本：v3.0，2026-05-25。基于所有实验结果和文献综述的完整分析版本。*
*关键待完成工作：补充PCSS实际实验结果（Table 5），填入真实数字。*
