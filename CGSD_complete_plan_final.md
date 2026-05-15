# CGSD: CRC-Guided Selective Distillation

## 面向 Semantic Filter 的统一训练-部署精度保障框架

## 完整技术方案

---

# 第一部分：问题与动机

## 1. 要解决的问题

**Semantic filter** 是非结构化数据分析中最基础的算子：给定自然语言谓词 $q$（如"该文档是否支持该声明？"）和 $N$ 篇文档 $\{d_1, \dots, d_N\}$，对每篇文档输出布尔判断 $y_i \in \{0,1\}$。

基线方案对每篇文档调用大模型（如 GPT-5），成本为 $N \times c_T$，$N$ 大时不可接受。

## 2. 现有方案的缺陷

**方案 A：Zero-shot Cascade**（LOTUS, BARGAIN, 我们的前序 Batch Cascade 工作）

用固定能力的小模型做第一层预测，不确信的样本 defer 到大模型。问题：小模型能力不变，defer 比例由模型的先天不足决定，无法改善。

**方案 B：通用蒸馏**（Distilling Step-by-Step, TensorZero 等）

用大模型标注数据训练小模型。问题：(1) 如何选择标注数据是启发式的；(2) 没有部署时的精度保证——用户无法预先知道"训练后的模型准确率能到多少"。

**方案 C：Active Learning**（CoPAL, CRC-AL 等）

用不确定性信号选择标注数据。问题：active learning 的标签复杂度定理依赖不可验证的分布假设（Tsybakov 参数等），在具体数据集上无法兑现精度承诺。

## 3. 本方案的核心洞察

**CRC（Conformal Risk Control）可以同时解决训练和部署两个问题：**

- **指导训练**：CRC 的 defer 集指出模型的弱点 → 蒸馏目标
- **量化训练效果**：每轮训练后 CRC 重校准的 defer 率下降量 → 训练进度
- **保障部署精度**：CRC 保证 $\mathbb{E}[\text{error}] \leq \alpha$ → 数学承诺

关键理论性质：**精度保证与模型训练质量完全解耦**。精度 $\alpha$ 由 CRC 校准保证，不论模型怎么训练。训练质量只影响 defer 率（即成本），不影响精度。

这构成一个闭环：CRC 识别弱点 → 蒸馏修补弱点 → CRC 重校准 → defer 减少 → 重复，全程精度保证不变。Active learning 只覆盖"选什么数据"，知识蒸馏只覆盖"怎么训练"，conformal prediction 只覆盖"怎么保障"。CGSD 是三者的统一。

---

# 第二部分：理论基础

## 4. CRC 的精度-成本解耦定理

### 4.1 CRC 校准回顾

设校准单元损失 $L_1(\lambda), \dots, L_n(\lambda) \in [0,1]$，关于 $\lambda$ 单调不增。CRC 选择：

$$\hat{\lambda} = \min\left\{\lambda \in \Lambda : \frac{n}{n+1}\cdot\frac{1}{n}\sum_{j=1}^n L_j(\lambda) + \frac{1}{n+1} \leq \alpha\right\}$$

**定理（Angelopoulos et al., ICLR 2024）**：设校准样本与测试样本可交换，则 $\mathbb{E}[L_{\text{test}}(\hat{\lambda})] \leq \alpha$。

### 4.2 CRC 保证对任意固定模型成立

**定理 1（跨训练策略的 CRC 保证）**

设 $\mathcal{S}_\theta$ 是任意方法训练得到的固定模型。设 $\mathcal{D}_{\text{cal}}$ 是独立于训练数据的校准集，其标签由 teacher 提供。定义 accept 损失：

$$L_j(\theta, \lambda) = \mathbf{1}[R_j(\theta) \geq \lambda] \cdot \mathbf{1}[\hat{y}_j^\theta \neq y_j]$$

若以下条件成立：
1. 训练数据 $S_{\text{train}} \cap \mathcal{D}_{\text{cal}} = \emptyset$
2. $\mathcal{D}_{\text{cal}}$ 中的样本与测试样本可交换
3. $L_j(\theta, \lambda)$ 关于 $\lambda$ 单调不增（$\lambda$ 增大 → accept 条件更严 → $L_j$ 不增）

则 $\hat{\lambda}$ 由 CRC 在 $\mathcal{D}_{\text{cal}}$ 上选出后：

$$\mathbb{E}[L_{\text{test}}(\theta, \hat{\lambda})] \leq \alpha$$

**证明**：模型 $\theta$ 在校准前已固定，不依赖 $\mathcal{D}_{\text{cal}}$。因此 $\theta$ 对 $\mathcal{D}_{\text{cal}} \cup \{d_{\text{test}}\}$ 中的每个样本施加同一确定性映射 $d \mapsto (\hat{y}^\theta(d), R(d))$。该映射下损失 $L_j(\theta, \lambda)$ 是 $d_j$ 的确定性函数。由条件 2 的可交换性，$\{L_j\}_{j \in \mathcal{D}_{\text{cal}}} \cup \{L_{\text{test}}\}$ 可交换。由条件 3 的单调性和 $L \in [0,1]$，CRC 定理的所有条件满足，结论成立。$\square$

**意义**：不论 $\theta$ 是 zero-shot、random SFT、CGSD 蒸馏还是任何方法训练——只要校准时 $\theta$ 已固定且校准集独立，精度保证成立。精度保证不依赖于训练的具体细节。

### 4.3 三角关系

$$\boxed{m \;\xrightarrow{\text{蒸馏}}\; \text{acc}(\theta_m) \;\xrightarrow{\text{CRC 校准}}\; \rho(m, \alpha) \;\xrightarrow{\text{成本}}\; C(m, \alpha)}$$

- $m$：标注预算（投入）
- $\text{acc}(\theta_m)$：蒸馏后模型的裸准确率（中间变量）
- $\rho(m, \alpha)$：CRC 校准后 defer 率（可直接测量的训练效果）
- $C(m, \alpha) = (m + n_{\text{cal}} + \rho \cdot N) \cdot c_T + N \cdot c_S$：总成本

**在这条链的每个点上，系统精度都满足 $\mathbb{E}[\text{error}] \leq \alpha + \beta_3$**（$\beta_3$ 为 teacher 错误率）。

### 4.4 Defer 率是训练质量的单调度量

**命题**：设模型 $\theta_A$ 在校准集上的 accept accuracy 高于 $\theta_B$。则在相同风险预算 $\alpha$ 下，$\hat{\lambda}(\theta_A) \leq \hat{\lambda}(\theta_B)$，因而 $\rho(\theta_A, \alpha) \leq \rho(\theta_B, \alpha)$。

**证明**：模型更准 → 同一 $\lambda$ 下 accept 错误更少 → $\hat{R}_n^+(\lambda) \leq \alpha$ 在更小 $\lambda$ 处成立 → $\hat{\lambda}$ 更小 → 更多样本被 accept → defer 率更低。$\square$

**实践意义**：每轮训练后 $\Delta\rho = \rho^{(t)} - \rho^{(t+1)}$ 直接度量该轮蒸馏的 CRC 认证的改善量。

## 5. 数据选择的理论依据

### 5.1 为什么 Defer 集是好的蒸馏目标

**来自 active learning 的论据**：对线性分类器，margin-based selection（选 $|\ell_i|$ 最小的样本）在 Tsybakov 噪声条件下达到标签复杂度 $O(d \cdot \text{polylog}(1/\varepsilon))$，相比被动学习 $O(d/\varepsilon)$ 有指数级改进（Balcan et al., 2006; Yan & Zhang, AAAI 2016）。

**与我们场景的对应**：CRC defer 集 = $\{i : R_i < \hat{\lambda}\}$ = logit margin 低于阈值的样本 ≈ margin-based active learning 的选择准则。

**LoRA rank-1 的近似线性性**：在 NTK 线性化 regime 中，LoRA SFT 等价于在冻结特征 $\phi(x) = \nabla_{\Delta\theta} f_{\theta_0}(x)$ 上拟合线性模型。因此 active learning 的标签复杂度改进近似适用。

**限定**：(1) NTK 线性化是近似，需实验验证；(2) 上述定理假设目标函数在假设类中（realizability），对 defer 集中超出 student 能力的样本不成立。

### 5.2 不是所有 Defer 样本都有价值

**来自 ICLR 2025 的警示**（*Medium-Difficulty Samples Constitute Smoothed Decision Boundary for Knowledge Distillation on Pruned Datasets*）：

在知识蒸馏中，由于 student-teacher 的 capacity gap，student 无法完美学习 teacher 在最困难样本上的决策。训练在最困难样本上反而导致 decision boundary drift，损害泛化。

**Defer 集的异质性分解**：

$$\mathcal{D}_{\text{def}} = \underbrace{\mathcal{D}_{\text{learnable}}}_{\text{student 缺知识，可修复}} \cup \underbrace{\mathcal{D}_{\text{unlearnable}}}_{\text{超出 student 能力}}$$

- $\mathcal{D}_{\text{learnable}}$：student 不确信因缺乏训练信号。teacher 高置信。训练后可学会。
- $\mathcal{D}_{\text{unlearnable}}$：需要 student 不具备的能力（长链推理等）。即使训练也无法正确分类。训练反而有害。

**可学习性的 proxy 信号**：teacher 置信度 $c_i^\mathcal{T}$。若 teacher 高置信而 student 不确信 → 知识缺口，可学习。若 teacher 也不确信 → 本质模糊或超出能力。

---

# 第三部分：完整方法

## 6. 记号汇总

| 符号 | 含义 | 典型值 |
|------|------|--------|
| $q$ | 自然语言谓词（query） | — |
| $\mathcal{U} = \{d_1,\dots,d_N\}$ | 文档池 | $N = 165{,}447$（FEVER） |
| $y_i \in \{0,1\}$ | Oracle 标签 $y_i = \mathcal{T}(q, d_i)$ | — |
| $\mathcal{T}$ | Teacher（GPT-5），单次成本 $c_T$ | input \$1.25/M tokens |
| $\mathcal{S}_\theta$ | Student（Qwen3-0.6B + LoRA） | input ~\$0.03/M tokens |
| $\theta_0$ | Student 基座参数（frozen） | 0.6B |
| $\ell_i(\theta)$ | Logit margin: $\log p(\text{yes}) - \log p(\text{no})$ | — |
| $R_i(\theta) \in [0,1]$ | 路由分数: $\sigma(\|\ell_i\|/T)$ | — |
| $\hat{y}_i(\theta)$ | Student 预测: $\mathbf{1}[\ell_i > 0]$ | — |
| $\hat{\lambda}$ | CRC 阈值 | 在 $\Lambda = \{0.00, 0.01, \dots, 1.00\}$ 上搜索 |
| $\alpha$ | 用户风险预算 | 0.05–0.12 |
| $m_{\text{total}}$ | 总蒸馏标注预算 | 500 |
| $n_{\text{cal}}$ | 校准集大小 | 200 |

**Qwen3-0.6B 架构参数**：28 层，hidden dim 1024，GQA（16 Q heads / 8 KV heads），FFN 3072，vocab 151936，max context 32768。

## 7. 数据划分

全量数据 $\mathcal{U}$ 在算法启动前做一次性随机划分：

$$\mathcal{U} = \mathcal{D}_{\text{cal}} \;\sqcup\; \mathcal{U}_{\text{pool}}$$

- $\mathcal{D}_{\text{cal}}$（$n_{\text{cal}} = 200$ 个）：CRC 校准集，调用 teacher 标注（一次性成本）
- $\mathcal{U}_{\text{pool}}$（$N - 200$ 个）：候选池，从中选择蒸馏样本和最终部署推理

**校准集大小选取依据**：CRC 有限样本修正项 $1/(n+1)$。$n_{\text{cal}} = 200$ 时修正 $\approx 0.5\%$。推荐 $n_{\text{cal}} \geq \max(200, 5/\alpha)$。

**核心不变式**（全程强制）：
1. $S_{\text{train}} \cap \mathcal{D}_{\text{cal}} = \emptyset$（训练集与校准集不重叠）
2. 划分在模型训练前固定

## 8. 完整算法

### 8.1 算法总览

```
Phase 0: 数据准备
  - 划分 D_cal 和 U_pool
  - Teacher 标注 D_cal（一次性）
  - 计算所有文档的 pair embedding（一次性）

Phase 1: 初始评估（Round 0）
  - Zero-shot student 对全量推理
  - CRC 校准 → 初始 λ̂, 初始 defer 集

Phase 2: 迭代蒸馏（Round t = 0, 1, ..., T_max-1）
  - 从 defer 集选择 m_t 个样本（DBDS）
  - Teacher 标注选中样本
  - LoRA SFT（从基座重新训练）
  - Student 重推理全量数据
  - CRC 重校准 → 新 λ̂, 新 defer 集
  - 检查停止条件

Phase 3: 最终部署
  - 固定模型 θ*, CRC 阈值 λ̂*
  - R_i ≥ λ̂* → accept（输出 student 预测）
  - R_i < λ̂* → defer（调用 teacher）
```

### 8.2 Phase 0：数据准备

**8.2.1 校准集标注**

对 $\mathcal{D}_{\text{cal}}$ 中的 200 个文档调用 teacher 获取标签。Prompt：

```
System: You are a precise fact-checking assistant.
        Answer with exactly "yes" or "no".
User:   Query: {q}
        Document: {d_j}
        Does the document satisfy the query?
```

**8.2.2 Pair Embedding 计算**

对所有文档计算 query-aware embedding，使用 Qwen3-Embedding-0.6B（专用 embedding 模型）：

$$z_i = \text{Qwen3-Emb-0.6B}\!\left(\texttt{"Instruct: Classify.\textbackslash nQuery: } q \texttt{\textbackslash nDocument: } d_i\texttt{"}\right) \in \mathbb{R}^{1024}$$

**为什么用专用 embedding 模型而非 student 的 hidden states**：(1) 专用模型的 embedding 经过对比学习，语义相似度更准确；(2) embedding 在迭代中保持固定（不随 SFT 变化），确保跨轮次 k-Center 选择在同一空间进行。

### 8.3 Phase 1：初始评估

**8.3.1 Zero-shot Student 推理**

对每个文档构造输入（使用 Qwen3 chat template，non-thinking mode）：

```
<|im_start|>system
You are a precise classifier. Answer only "yes" or "no"./no_think<|im_end|>
<|im_start|>user
Query: {q}
Document: {d_i}
Does the document satisfy the query?<|im_end|>
<|im_start|>assistant
```

在第一个 output token 位置读取 logits：

$$\ell_i = \log p_{\theta_0}(\text{tok}_{\text{yes}} | \text{context}_i) - \log p_{\theta_0}(\text{tok}_{\text{no}} | \text{context}_i)$$

**预测标签**：$\hat{y}_i = \mathbf{1}[\ell_i > 0]$

**路由分数**：$R_i = \sigma(|\ell_i| / T)$，其中 $\sigma(x) = 1/(1+e^{-x})$，$|\ell_i|$ 取绝对值表示不论预测方向的确信程度。

**温度 $T$ 的选取**：严格实验固定 $T = 15$。不要用同一个 $\mathcal{D}_{\text{cal}}$ 的标签扫描温度后再做 CRC 声明，否则 routing score 的定义会依赖校准标签，引入 adaptivity。若确实需要调参，应另设 tuning split 选择 $T$，再用独立 calibration split 做 CRC。

**8.3.2 初始 CRC 校准**

对 $\mathcal{D}_{\text{cal}}$ 中的样本，计算 accept 损失：

$$L_j(\lambda) = \mathbf{1}[R_j \geq \lambda] \cdot \mathbf{1}[\hat{y}_j \neq y_j]$$

在网格 $\Lambda = \{0.00, 0.01, \dots, 1.00\}$ 上扫描：

$$\hat{\lambda}^{(0)} = \min\left\{\lambda \in \Lambda : \frac{n_{\text{cal}}}{n_{\text{cal}}+1} \cdot \frac{1}{n_{\text{cal}}}\sum_{j} L_j(\lambda) + \frac{1}{n_{\text{cal}}+1} \leq \alpha\right\}$$

若无可行 $\lambda$，则 $\hat{\lambda}^{(0)} = 1.01$（全部 defer）。

**8.3.3 识别 Defer 集**

$$\mathcal{D}_{\text{def}}^{(0)} = \{i \in \mathcal{U}_{\text{pool}} : R_i < \hat{\lambda}^{(0)}\}$$

记录初始 defer 率 $\rho^{(0)} = |\mathcal{D}_{\text{def}}^{(0)}| / |\mathcal{U}_{\text{pool}}|$。

### 8.4 Phase 2：迭代蒸馏

每轮 $t = 0, 1, \dots, T_{\max}-1$ 执行以下步骤。

**8.4.1 DBDS 数据选择（Defer-Boundary Diversified Selection）**

目标：从当前 defer 集 $\mathcal{D}_{\text{def}}^{(t)}$（排除已选样本）中选 $m_t$ 个最有价值的样本。

**Step A：分 Band**

按 student 路由分数将 defer 集分为三个 band：

- **Band B（边界带）**：$R_i \in [\hat{\lambda}^{(t)} - \delta,\; \hat{\lambda}^{(t)})$，默认 $\delta = 0.1$
  - 翻转潜力最高——差一点就能 accept
- **Band M（中间带）**：$R_i \in [\hat{\lambda}^{(t)} - 2\delta,\; \hat{\lambda}^{(t)} - \delta)$
  - 中等不确信
- **Band F（深度带）**：$R_i < \hat{\lambda}^{(t)} - 2\delta$
  - 模型完全无信心——包含 $\mathcal{D}_{\text{unlearnable}}$ 的概率最高

**Step B：分配名额**

$$m_t^B = \lfloor 0.6 \cdot m_t \rfloor, \quad m_t^M = \lfloor 0.3 \cdot m_t \rfloor, \quad m_t^F = m_t - m_t^B - m_t^M$$

比例 $(0.6, 0.3, 0.1)$ 的依据：Band B 获得最多名额（边界样本翻转对 defer 率的即时影响最大）。Band F 比例最小（最可能不可学习，训练价值低甚至有害——依据 ICLR 2025 的 decision boundary drift 发现）。若某 band 样本不足，剩余名额分给其他 band。

**Step C：Band 内 k-Center Greedy**

在每个 band 内部，使用 k-Center Greedy 在 pair embedding 空间 $\{z_i\}$ 中选择多样性最大化的子集：

```
初始化 S' = {距中心最远的点}
重复 m'-1 次:
    选择距 S' 中最近点最远的未选样本加入 S'
返回 S'
```

该算法保证 2-近似最优覆盖半径（Sener & Savarese, ICLR 2018）。

**Step D：Easy Anchor（可选）**

在训练集中额外加入 $\lfloor 0.1 \cdot m_t \rfloor$ 个 student 高置信且预测正确的样本。这些样本训练信号极弱（loss 接近 0），但起正则化作用，防止 LoRA 参数过度偏向 defer 区域。easy anchor 不能使用 student pseudo-label，也必须使用 teacher 或离线 ground truth 替代标签。

**8.4.2 Teacher 标注**

对选中的 $m_t$ 个 defer 样本和 $a_t$ 个 easy anchor 调用 teacher，获取标签 $y_j$。若使用离线实验，可用 ground truth 替代真实 API，但调用量统计要区分真实 API 与 ground truth substitute。

**同时提取 teacher 的 logit 信息**（用于可学习性加权）：

$$c_j^\mathcal{T} = \sigma(|\ell_j^\mathcal{T}| / T_\mathcal{T})$$

若使用 GPT-5 等 API，大部分支持返回 log-probability，可直接获取。若不支持，可用 3 次调用的一致性估计。

**8.4.3 LoRA SFT**

**训练数据**：累积训练集 $S_{\text{train}} = \bigcup_{t'=0}^t S_{t'}$。

**每轮从基座 $\theta_0$ 重新训练**（非 continual training），原因：
- $\theta^{(t+1)} = \text{SFT}(\theta_0, S_{\text{train}})$ 是 $S_{\text{train}}$ 的纯函数，理论分析简洁
- LoRA rank-1 SFT 仅需 1-3 分钟，重训成本可忽略
- 避免 continual training 的 catastrophic forgetting

**LoRA 配置**：

| 参数 | 值 | 说明 |
|------|-----|------|
| Rank | 1 | 二分类无 rank threshold（arXiv 2605.03724） |
| Alpha | 16 | 缩放系数 |
| Target modules | `q_proj`, `v_proj` | 仅 attention |
| Dropout | 0.05 | 轻量正则化 |

**可训练参数**：每层 `q_proj`（$1024 \to 1024$）= 2048 params，`v_proj`（$1024 \to 512$，因 GQA 8 KV heads）= 1536 params。28 层总计 **100,352 参数**（总参数的 0.017%）。

**训练输入格式**（与推理相同）：

```
<|im_start|>system
You are a precise classifier. Answer only "yes" or "no"./no_think<|im_end|>
<|im_start|>user
Query: {q}
Document: {d_j}
Does the document satisfy the query?<|im_end|>
<|im_start|>assistant
{yes 或 no}<|im_end|>
```

**损失**：标准 cross-entropy，仅对 assistant 回复部分的 token（"yes"/"no" + `<|im_end|>`）计算 loss。

**Teacher 置信度加权**（可选增强）：

$$\mathcal{L} = -\frac{1}{|S|}\sum_{j \in S} (c_j^\mathcal{T})^\beta \left[y_j \log \hat{p}_j + (1-y_j)\log(1-\hat{p}_j)\right], \quad \beta = 1$$

teacher 高置信样本获得更高权重。这使得可学习样本主导梯度，不可学习样本（teacher 也不确信）的影响被降低。

**训练超参数**：

| 参数 | 值 |
|------|-----|
| Optimizer | AdamW ($\beta_1=0.9, \beta_2=0.999$, weight\_decay=0.01) |
| Learning rate | $2 \times 10^{-4}$, cosine decay, 10% warmup |
| Epochs | 3 |
| Batch size | 16 |
| Max seq len | 512 |
| Precision | bf16 |

**训练时间**：500 样本 × 3 epochs，batch=16 → ~94 steps/epoch → 约 **1-3 分钟**（单 A100）。

**8.4.4 重推理与重校准**

SFT 完成后，用新模型 $\mathcal{S}_{\theta^{(t+1)}}$ 对 $\mathcal{U}_{\text{pool}}$ 和 $\mathcal{D}_{\text{cal}}$ 重新推理，提取新的 $(\hat{y}_i^{(t+1)}, R_i^{(t+1)})$。

在 $\mathcal{D}_{\text{cal}}$ 上重新 CRC 校准 → $\hat{\lambda}^{(t+1)}$。

识别新 defer 集 $\mathcal{D}_{\text{def}}^{(t+1)}$，计算新 defer 率 $\rho^{(t+1)}$。

> **关于校准集复用的严格处理**：中间轮的 $\hat{\lambda}^{(t)}$ 仅用于 DBDS 数据选择（启发式目的），不构成精度声明。若同一个 $\mathcal{D}_{\text{cal}}$ 也影响了训练样本选择、温度选择、停止或模型选择，则最终模型不再独立于该校准集；此时最终 CRC 只能作为经验验证。要声明严格最终保证，需要额外保留没有参与任何选择决策的独立最终校准集。

**8.4.5 停止准则**

满足以下任一条件时停止：

1. 预算耗尽：$\sum m_t \geq m_{\text{total}}$
2. 达到最大轮次：$t \geq T_{\max}$
3. 边际收益消失：$\Delta\rho = \rho^{(t)} - \rho^{(t+1)} < 0.005$
4. 经济学停止：$\Delta\rho \cdot N \cdot c_T < m_t \cdot c_T$（节省的 defer 成本不抵标注成本）

### 8.5 Phase 3：最终部署

模型 $\theta^*$ 固定后，在 $\mathcal{D}_{\text{cal}}$ 上做最终 CRC 校准 → $\hat{\lambda}^*$。

对每个文档 $d_i$：
- $i \in S_{\text{train}}$：直接输出已有 teacher 标签
- $R_i(\theta^*) \geq \hat{\lambda}^*$：**accept**，输出 student 预测 $\hat{y}_i$
- $R_i(\theta^*) < \hat{\lambda}^*$：**defer**，调用 teacher

**精度保证**：CRC 直接控制的是 $\mathbb{E}[\mathbf{1}\{\text{accept 且 wrong}\}] \leq \alpha$，不是 accept 子集条件错误率 $\Pr(\text{wrong}\mid\text{accept})$。若 teacher 错误率为 $\beta_3$，则在独立最终校准条件成立时，系统整体错误可按 teacher defer 部分叠加 $\beta_3$ 分析。

### 8.6 默认超参汇总

| 超参 | 默认值 | 说明 |
|------|--------|------|
| $n_{\text{cal}}$ | 200 | 校准集大小 |
| $m_{\text{total}}$ | 500 | 总蒸馏预算 |
| $T_{\max}$ | 3 | 最大迭代轮次 |
| 预算分配 | 递减: 250, 150, 100 | 第一轮最多 |
| $\delta$ | 0.1 | DBDS band 宽度 |
| Band 比例 | $(B, M, F) = (0.6, 0.3, 0.1)$ | 边界优先 |
| $T$ | 15 | 固定温度缩放；调参需独立 tuning split |
| $\alpha$ | 0.07 | 风险预算 |
| LoRA rank | 1 | — |
| LoRA alpha | 16 | — |
| Target modules | q\_proj, v\_proj | — |
| LR | $2 \times 10^{-4}$ | — |
| Epochs | 3 | — |
| $\beta$ (teacher 加权) | 1 | 0 = 不加权 |

### 8.7 与前序 Batch Cascade 的整合

CGSD 训练的 0.6B 模型可直接替换前序 Batch Cascade 中的 4B zero-shot 模型：

| 层 | 前序工作 | 替换后 |
|----|---------|--------|
| Layer 1 | Qwen3-4B zero-shot batch | Qwen3-0.6B CGSD-SFT batch |
| Layer 2 | Qwen3-4B 逐样本复核 | Qwen3-0.6B CGSD-SFT 逐样本 |
| Layer 3 | GPT-5 兜底 | GPT-5 兜底 |

优势：模型更小（0.6B vs 4B），推理成本降 ~6x；基础 accuracy 更高（SFT vs zero-shot），defer 更少。两阶段加性风险上界 $\mathbb{E}[L^{\text{total}}] \leq \alpha_1 + \alpha_2 + \beta_3$ 仍然成立。

---

# 第四部分：理论分析

## 9. 已严格证明的结论

| 编号 | 结论 | 性质 |
|------|------|------|
| 定理 1 | CRC 保证对任意固定模型成立（含 SFT 后） | **严格**（CRC 定理直接推论） |
| 命题（§4.4） | Defer 率关于模型 accept accuracy 单调 | **严格**（CRC 单调性） |
| k-Center 覆盖 | DBDS 中 k-Center 保证 2-近似最优覆盖 | **严格**（经典结果） |
| 成本公式 | $N_\mathcal{T} = m + n_{\text{cal}} + \rho \cdot N$ | **精确**（会计恒等式） |

## 10. 有理论支撑但需实验验证的论证

| 编号 | 论证 | 理论依据 | 需验证 |
|------|------|---------|--------|
| Defer 集选择 > 随机选择 | Margin-based AL 标签复杂度定理 | NTK 线性化是否成立 |
| Band F 价值 < Band B | Capacity gap + boundary drift (ICLR 2025) | 在 per-query 二分类中是否严重 |
| Teacher 加权改善训练 | 噪声学习理论 | 加权 vs 不加权的实际差异 |
| LoRA rank-1 足够 | PŁ 条件 (arXiv 2605.03724) | 在 0.6B + 二分类上的实际表现 |

## 11. 与现有工作的对比

| 工作 | 数据选择 | 模型适配 | 精度保证 | 训练效果度量 |
|------|---------|---------|---------|------------|
| LOTUS (VLDB'25) | ✗ | ✗ | SUPG 采样 | ✗ |
| BARGAIN (2025) | ✗ | ✗ | 统计估计 | ✗ |
| ScaleDoc (2025) | ✗ | MLP proxy | 精度目标 | ✗ |
| AdaConG (2025) | ✗ 全量训练 | ✓ 重加权 | CP 引导 | ✗ |
| CoPAL (2024) | CP 集大小 | ✗ | ✗ | ✗ |
| Active Learning | 不确定性 | ✓ | ✗ 不可验证 | ✗ |
| **CGSD（本文）** | **CRC defer** | **LoRA SFT** | **CRC 保证** | **$\Delta\rho$ 可量化** |

---

# 第五部分：实验计划

## 12. 实验列表

### 实验 1：CGSD 核心有效性

在 FEVER（165K 样本）上验证完整 CGSD pipeline。

报告：每轮 defer 率 $\rho^{(t)}$、裸 accuracy、CRC 校准后 final accuracy、总 teacher 调用量。

### 实验 2：数据选择策略对比

固定 $m = 500$，比较以下策略训练后模型的 defer 率和 accuracy：

| 策略 | 描述 |
|------|------|
| Random | 均匀随机选 500 个 |
| Uncertainty-only | 选 $R_i$ 最低的 500 个 |
| k-Center-only | embedding 空间 k-Center |
| CGSD-Single ($T_{\max}=1$) | 一轮 DBDS |
| CGSD-Iter ($T_{\max}=3$) | 三轮迭代 DBDS |

### 实验 3：标注预算曲线

$m \in \{50, 100, 200, 300, 500, 700, 1000\}$，绘制 $m$ vs $\rho(m, \alpha)$ 曲线。确定 $\rho$ 饱和的临界 $m$。

### 实验 4：消融实验

| 变量 | 取值 | 其他参数固定 |
|------|------|------------|
| Band 比例 | $(1,0,0)$, $(0.6,0.3,0.1)$, $(0.33,0.33,0.34)$, $(0,0,1)$ | $m=500$ |
| $\delta$（band 宽度） | $\{0.05, 0.1, 0.15, 0.2\}$ | $m=500$ |
| 迭代轮次 $T_{\max}$ | $\{1, 2, 3, 5\}$ | $m=500$ |
| LoRA rank | $\{1, 2, 4, 8\}$ | $m=500$ |
| Teacher 加权 $\beta$ | $\{0, 0.5, 1, 2\}$ | $m=500$ |
| Easy anchor 比例 | $\{0, 0.05, 0.1, 0.2\}$ | $m=500$ |

### 实验 5：端到端系统对比

| 方法 | 小模型 | 精度保证 |
|------|--------|---------|
| Full Teacher (GPT-5) | — | Oracle |
| Qwen3-4B ZS Cascade（前序工作） | 4B zero-shot | CRC |
| Qwen3-4B 二次分流（前序工作最佳） | 4B zero-shot | CRC |
| Random SFT + CRC Cascade | 0.6B random SFT | CRC |
| **CGSD** | **0.6B CGSD SFT** | **CRC** |

关键对比：CGSD（0.6B SFT + 500 标注）是否匹配或超过前序工作（4B zero-shot，无标注）？

### 实验 6：CRC 保证验证

在每轮 $t$，记录 CRC 名义风险 $\alpha$、wrong-accept 风险和诊断用 accept 子集条件错误率。20 次随机划分取平均，验证 wrong-accept 风险的均值是否 $\leq \alpha$。若同一个校准集参与过选择或停止判断，该实验只能作为经验验证；严格保证需要独立最终校准集。

### 实验 7：三角关系验证

绘制 $m \to \rho(m, \alpha) \to C(m, \alpha)$ 的完整曲线，验证理论预测：
- $\rho(m, \alpha)$ 单调不增
- 每个点上 $\mathbb{E}[\text{error}] \leq \alpha + \beta_3$
- 存在最优 $m^*$ 最小化总成本

---

# 第六部分：成本估算

以 FEVER（$N = 165{,}447$），$m = 500$，$n_{\text{cal}} = 200$ 为例：

| 操作 | 次数/规模 | 成本 |
|------|---------|------|
| Embedding（一次性） | $N$ 次 | ~\$1.31 |
| 校准集标注（一次性） | 200 次 teacher | ~\$0.10 |
| 蒸馏标注 | 500 次 defer 样本 + easy anchors | 以 usage 账本为准 |
| SFT 训练（3 轮） | 3 × 3min GPU | ~\$0.10 |
| Student 推理（4 轮全量） | $4N$ 次 0.6B | ~\$7.84 |
| **离线总计** | | **~\$9.60** |
| 部署 defer（若 $\rho = 2\%$） | 3309 次 teacher | ~\$2.58 |
| **系统总计** | | **~\$12.18** |

对比：全 GPT-5 ~\$81.63；前序 4B Cascade ~\$13.78。

当前实现默认使用 `model/qwen3-0.6b`；Student 推理成本和参数量应按该实际 checkpoint 记录和复核。

**优化空间**：第 2-4 轮推理可仅对 defer 集做（accept 集不需要重推理），将推理成本降至 ~\$3-4。

---

# 第七部分：论文叙事

## 论文标题

**CRC-Guided Selective Distillation: Unifying Training Data Selection and Deployment Calibration for LLM Cascades**

## 核心贡献

1. **框架贡献**：提出 CRC 作为统一训练与部署的精度证书框架。精度保证与模型训练质量完全解耦（定理 1）：训练质量仅影响成本（defer 率），不影响精度。这是 CRC 理论在 training-time optimization 中的首次系统性应用。

2. **算法贡献**：基于上述理论设计 CGSD 算法。CRC 的 defer 集驱动蒸馏数据选择（从 defer 边界 + 多样性选择），CRC 的 defer 率度量训练进度（经济学最优停止），CRC 的阈值保障部署精度。

3. **实证贡献**：在 FEVER 上，0.6B 模型 + 500 样本标注实现与 4B zero-shot 可比的 accuracy，同时保持 CRC 精度保证。

## 审稿人预期问题与回应

**Q：CRC model-agnostic 是已知的，何来贡献？**
A：Angelopoulos 证了单模型的 CRC 保证。我们的贡献是展示如何利用此性质指导模型的训练过程——defer 集 → 训练目标，defer 率 → 训练度量，精度保证 → 全程不变。

**Q：DBDS 的 band 比例 (0.6, 0.3, 0.1) 有理论依据吗？**
A：边界优先有 margin-based AL 的理论支持；deep defer 降权有 ICLR 2025 capacity gap 的实证支持。具体比例是默认值，消融实验 A4 覆盖 5 种配置。

**Q：为什么不用 influence function 选数据？**
A：Influence function 计算成本为 $O(N \cdot p)$，对 165K 样本和 100K 参数约需几小时。DBDS 仅需 embedding + k-Center，几秒完成。消融实验中可对比。

**Q：每轮重推理全量数据是否太贵？**
A：默认是全量推理（\$7.84），但后续轮可仅推理 defer 集（accept 集结果不变）。实际中 defer 集逐轮缩小，后续轮成本递减。
