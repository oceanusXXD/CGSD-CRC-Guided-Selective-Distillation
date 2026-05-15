# CGSD 实验方案详述

---

## 一、整体方案检查：已修正的问题

在整合过程中发现并修正了以下细节：

**修正 1：温度 $T$ 的选择**。原方案说"在 $\mathcal{D}_{\text{cal}}$ 上扫描 $T$"。但 $T$ 影响 $R_i$ 的计算，而 CRC 要求 $R_i$ 的计算方式在校准前固定。如果 $T$ 依赖 $\mathcal{D}_{\text{cal}}$ 的标签，会引入 adaptivity。严格处理：**固定 $T = 15$**（与前序工作一致）。如需调参，从 $\mathcal{D}_{\text{cal}}$ 中拆出 20% 做 $T$ 选择，剩余 80% 做 CRC 校准。

**修正 2：Easy anchor 的标签来源**。原方案说用 student 自身的高置信预测做 pseudo-label。但 student 可能高置信错误。修正：对 easy anchor 样本也调用 teacher 获取标签，额外成本约 50 次 teacher 调用（可忽略）。

**修正 3：重推理范围**。原方案说每轮对全量 $\mathcal{U}_{\text{pool}}$ 重推理。SFT 可能改变 accept 集样本的预测（damage rate），因此严格来说需要全量重推理。但如果实验验证 damage rate $\approx 0$，后续可仅重推理 defer 集作为工程优化。第一版实验中保守地做全量重推理。

---

## 二、数据集与评价基准

### 2.1 FEVER

- **来源**：FEVER (Fact Extraction and VERification) 数据集
- **规模**：165,447 条 claim-document 对
- **任务**：给定一个 claim（query）和一篇 Wikipedia 文档，判断文档是否支持该 claim
- **标签**：二分类（supported / refuted）
- **Oracle 定义**：GPT-5 (或 ground-truth label，视实验而定) 的判断作为 $y_i$
- **选择理由**：规模大（足够展示 defer 率的变化），有公开 ground truth（可交叉验证 teacher 的准确性），前序工作已在此数据集上有结果（可直接对比）

### 2.2 LROBench

- **来源**：LROBench 的 Select / Row-wise Filter 任务
- **规模**：30 queries, 1,162 条 row-level 样本
- **任务**：判断每一行是否满足给定的筛选条件
- **选择理由**：多 query 场景（30 个不同 query），规模小（测试小标注预算场景），前序工作有对比数据

---

## 三、实验 1：CGSD 核心有效性

### 3.1 目的

验证 CGSD 的完整 pipeline 是否有效：迭代蒸馏是否逐轮降低 defer 率。

### 3.2 设置

- 数据集：FEVER
- 配置：默认超参（$m_{\text{total}}=500$, $T_{\max}=3$, 分配 250/150/100, $\alpha=0.07$, $n_{\text{cal}}=200$）
- 随机种子：5 个不同的数据划分种子，报告均值 ± 标准差

### 3.3 每轮记录的指标

| 指标 | 含义 | 计算方式 |
|------|------|---------|
| $\text{acc}_{\text{raw}}^{(t)}$ | SFT 后 student 的裸准确率 | $\frac{1}{|\mathcal{U}_{\text{pool}}|}\sum_i \mathbf{1}[\hat{y}_i^{(t)} = y_i]$ |
| $\hat{\lambda}^{(t)}$ | CRC 校准阈值 | CRC 算法输出 |
| $\rho^{(t)}$ | Defer 率 | $|\mathcal{D}_{\text{def}}^{(t)}| / |\mathcal{U}_{\text{pool}}|$ |
| $\text{acc}_{\text{final}}^{(t)}$ | CRC cascade 后的最终准确率 | accept 样本用 student 预测，defer 样本用 oracle 标签 |
| $\text{risk}_{\text{accept-wrong}}^{(t)}$ | CRC 直接控制的 wrong-accept 风险 | $\frac{1}{|\mathcal{U}_{\text{pool}}|}\sum_i \mathbf{1}[R_i \geq \hat{\lambda}] \cdot \mathbf{1}[\hat{y}_i \neq y_i]$ |
| $\text{err}_{\text{accept}}^{(t)}$ | Accept 子集条件错误率（诊断指标） | $\frac{\sum_i \mathbf{1}[R_i \geq \hat{\lambda}] \cdot \mathbf{1}[\hat{y}_i \neq y_i]}{\sum_i \mathbf{1}[R_i \geq \hat{\lambda}]}$ |
| $N_\mathcal{T}^{(t)}$ | 累积 teacher 调用量 | $n_{\text{cal}} + \sum_{t'=0}^t (m_{t'} + a_{t'}) + n_{\text{def}}^{(t)}$ |
| $\text{FR}^{(t)}$ | 翻转率 | 见前文 §4.2 |
| $\text{DR}^{(t)}$ | 损害率 | 见前文 §4.2 |

### 3.4 预期结果格式

| Round | $\text{acc}_{\text{raw}}$ | $\hat{\lambda}$ | $\rho$ | $\text{acc}_{\text{final}}$ | Teacher calls | FR | DR |
|-------|--------------------------|-----------------|--------|----------------------------|---------------|-----|-----|
| 0 (ZS) | ~85% | ~0.65 | ~20% | ~93% | 200+0+33K | — | — |
| 1 | ~89% | ~0.55 | ~12% | ~95% | +250 + anchors | ? | ? |
| 2 | ~91% | ~0.50 | ~8% | ~96% | +150 + anchors | ? | ? |
| 3 | ~92% | ~0.47 | ~6% | ~96.5% | +100 + anchors | ? | ? |

（数值为预估，需实验验证）

### 3.5 关键观察项

1. $\rho^{(t)}$ 是否逐轮下降？若某轮 $\Delta\rho < 0$（defer 反而增加），需分析 $\text{DR}^{(t)}$ 是否显著
2. $\text{acc}_{\text{raw}}$ 与 $\rho$ 的对应关系是否与理论预测一致（acc 高 → $\hat{\lambda}$ 低 → $\rho$ 低）
3. $\text{risk}_{\text{accept-wrong}}$ 在最终独立校准口径下是否 $\leq \alpha$；$\text{err}_{\text{accept}}$ 只作为诊断，不直接受 CRC 的 $\alpha$ 约束

---

## 四、实验 2：数据选择策略对比

### 4.1 目的

验证 CGSD 的数据选择（从 defer 集 + DBDS）是否优于其他策略。

### 4.2 设置

固定 $m = 500$，$T_{\max} = 1$（单轮，消除迭代的影响，纯比较数据选择效果）。

| 策略 | 描述 | 实现细节 |
|------|------|---------|
| **Random** | 从 $\mathcal{U}_{\text{pool}}$ 均匀随机选 500 个 | `np.random.choice(U_pool, 500, replace=False)` |
| **Uncertainty** | 选 $R_i$ 最低的 500 个 | 按 $R_i$ 升序取前 500 |
| **k-Center** | 在 embedding 空间做 k-Center | 不考虑 $R_i$，纯多样性 |
| **Defer-Random** | 从 defer 集中随机选 500 个 | 先识别 defer 集，在其中随机选 |
| **DBDS** | 本文方法 | 从 defer 集分 band + k-Center |

### 4.3 评价指标

对每种策略，用相同的 LoRA SFT 配置训练后，在完整 $\mathcal{U}_{\text{pool}}$ 上评价：

1. **裸准确率** $\text{acc}_{\text{raw}}$：SFT 后 student 的准确率
2. **CRC defer 率** $\rho(\alpha)$：在 $\alpha = 0.07$ 下的 defer 率
3. **总 teacher 调用量** $N_\mathcal{T}$：$n_{\text{cal}} + 500 + a + \rho \cdot N$，其中 $a$ 是 easy anchor 数；若关闭 anchor 则 $a=0$

### 4.4 统计显著性

每种策略重复 5 次（不同随机种子影响数据划分和 k-Center 初始化）。使用 paired t-test 比较 DBDS 与各 baseline 的 $\rho$ 差异，报告 p-value。

---

## 五、实验 3：标注预算曲线

### 5.1 目的

确定 $m$ 与 $\rho$ 的关系曲线，找到边际收益递减的拐点。

### 5.2 设置

使用 DBDS 策略，$T_{\max}=1$（单轮），扫描 $m \in \{0, 50, 100, 200, 300, 500, 700, 1000, 2000\}$。

$m = 0$ 对应 zero-shot（无蒸馏），作为基线。

### 5.3 输出

绘制两条曲线：

**曲线 A**：$m$ vs $\rho(m, \alpha=0.07)$ —— 标注预算与 defer 率

**曲线 B**：$m$ vs 总成本 $C(m) = (m + n_{\text{cal}} + \rho \cdot N) \cdot c_T + N \cdot c_S$ —— 标注预算与总货币成本

曲线 B 有一个 U 形（$m$ 小时 defer 成本高，$m$ 大时标注成本高），其最低点即为经济学最优 $m^*$。

### 5.4 数据选择策略的对比

在同一张图上叠加 Random 和 Uncertainty 策略的曲线，展示 DBDS 在所有 $m$ 下的优势。

---

## 六、实验 4：消融实验

### 6.1 Band 比例消融

固定 $m=500$, $T_{\max}=1$。改变 $(B, M, F)$ 的比例：

| 编号 | Band B | Band M | Band F | 含义 |
|------|--------|--------|--------|------|
| A1 | 1.0 | 0.0 | 0.0 | 全边界 |
| A2 | 0.6 | 0.3 | 0.1 | 默认 |
| A3 | 0.33 | 0.34 | 0.33 | 均匀 |
| A4 | 0.0 | 0.0 | 1.0 | 全深度 defer |
| A5 | 0.0 | 1.0 | 0.0 | 全中间 |

预期：A1 或 A2 最优（边界样本翻转价值最高），A4 最差（deep defer 样本最可能不可学习）。若 A4 确实最差，则验证了 ICLR 2025 的 decision boundary drift 在我们场景中存在。

### 6.2 迭代轮次消融

固定 $m_{\text{total}}=500$，改变 $T_{\max} \in \{1, 2, 3, 5\}$，预算均匀分配。

| $T_{\max}$ | 每轮 $m_t$ | 重训次数 | 重推理次数 |
|-----------|-----------|---------|----------|
| 1 | 500 | 1 | 1 |
| 2 | 250 | 2 | 2 |
| 3 | 167 | 3 | 3 |
| 5 | 100 | 5 | 5 |

预期：2-3 轮后边际收益递减。$T_{\max}=1$ 是最简单的 baseline（单轮 CGSD = "从 zero-shot defer 集选数据"）。

### 6.3 LoRA Rank 消融

固定 $m=500$, $T_{\max}=1$。Rank $\in \{1, 2, 4, 8\}$。

预期：rank=1 已足够（理论预测）。若 rank=1 显著差于 rank=4，说明 per-query 二分类的假设类需要更高秩，需调整理论论述。

### 6.4 Teacher 加权消融

固定 $m=500$, $T_{\max}=1$。$\beta \in \{0, 0.5, 1, 2\}$。

$\beta=0$ 即不加权（均匀 loss），$\beta=1$ 为默认。

预期：$\beta > 0$ 优于 $\beta = 0$，尤其在 $m$ 较大（训练集中含更多 deep defer 样本）时。

---

## 七、实验 5：端到端系统对比

### 7.1 对比方法

| 方法 | 小模型 | 训练？ | 精度保证 | 来源 |
|------|--------|--------|---------|------|
| Full GPT-5 | — | — | Oracle | 上界 |
| Qwen3-0.6B ZS | 0.6B zero-shot | ✗ | — | 下界 |
| Qwen3-4B ZS Cascade | 4B zero-shot | ✗ | CRC | 前序工作 |
| Qwen3-4B 二次分流 | 4B zero-shot | ✗ | CRC | 前序最佳 |
| Random SFT + CRC | 0.6B, random 500 | ✓ | CRC | 简单 baseline |
| **CGSD** | **0.6B, DBDS 500** | **✓** | **CRC** | **本文** |

### 7.2 公平对比原则

1. **相同 $\alpha$**：所有 CRC 方法使用相同的 $\alpha = 0.07$
2. **相同 oracle**：所有方法的 "ground truth" 均为 GPT-5 或 FEVER 标注
3. **总成本核算**：包含所有 teacher 调用（标注 + 校准 + defer），所有 student 推理，SFT 训练成本

### 7.3 汇报格式

| 方法 | $\text{acc}_{\text{raw}}$ | $\text{acc}_{\text{final}}$ | Defer % | Teacher calls | Total cost |
|------|--------------------------|----------------------------|---------|---------------|------------|
| Full GPT-5 | — | 100% | — | 165,447 | \$81.63 |
| 0.6B ZS | 85% | — | — | — | — |
| 4B ZS Cascade | 87% | 90.3% | 8.25% | 13,654 | ~\$14 |
| 4B 二次分流 | 87% | 92.8% | 1.50% | 2,485 | ~\$14 |
| Random SFT+CRC | ~88% | ? | ? | 200+500+a+? | ? |
| **CGSD** | **~91%** | **?** | **?** | **200+500+a+?** | **?** |

### 7.4 关键论证目标

**论证 1**：CGSD（0.6B SFT + 500 标注）的 $\text{acc}_{\text{final}}$ ≥ 4B ZS Cascade 的 $\text{acc}_{\text{final}}$

若成立，证明 "少量定向标注 + 超小模型 SFT > 大模型 zero-shot"。

**论证 2**：CGSD 的总成本 < 4B ZS Cascade 的总成本

成立条件：CGSD 的 defer 率足够低，使得额外标注成本（700 次 teacher）被更低的推理成本（0.6B vs 4B）和更低的 defer 成本抵消。

**论证 3**：CGSD 的 defer 率 < Random SFT + CRC 的 defer 率（相同 $m$）

直接验证 DBDS 数据选择比随机选择更高效。

---

## 八、实验 6：CRC 保证验证

### 8.1 目的

验证定理 1 在实践中是否成立：$\mathbb{E}[\mathbf{1}\{\text{accept 且 wrong}\}] \leq \alpha$。注意这不是 accept 子集条件错误率。

### 8.2 方法

1. 在完整 CGSD pipeline 的最终轮，记录 wrong-accept 风险 $\text{risk}_{\text{accept-wrong}}$，同时记录诊断用的 $\text{err}_{\text{accept}}$
2. 重复 20 次，每次使用不同的 $\mathcal{D}_{\text{cal}} / \mathcal{U}_{\text{pool}}$ 随机划分
3. 计算 20 次中 $\text{risk}_{\text{accept-wrong}} > \alpha$ 的次数（经验违反率）

若训练样本选择、温度选择或停止判断复用了同一个 $\mathcal{D}_{\text{cal}}$，这里验证的是当前 pipeline 的经验表现；严格 theorem-level guarantee 需要额外独立最终校准集。

### 8.3 预期

CRC 提供的是**期望**保证（$\mathbb{E} \leq \alpha$），不是确定性保证。因此单次实验可能出现 $\text{risk}_{\text{accept-wrong}} > \alpha$。但 20 次的均值应 $\leq \alpha$，且违反率不应显著高于名义水平。

### 8.4 多 $\alpha$ 扫描

对 $\alpha \in \{0.03, 0.05, 0.07, 0.09, 0.12\}$ 各跑 20 次，绘制：
- 名义 $\alpha$ vs 实际平均 wrong-accept 风险
- 名义 $\alpha$ vs 平均 defer 率

理想结果：实际平均 wrong-accept 风险在每个 $\alpha$ 下都 $\leq \alpha$（保证成立），且 defer 率随 $\alpha$ 单调递减（更宽松 → 更少 defer）。

---

## 九、实验 7：三角关系验证

### 9.1 目的

验证 $m \to \rho \to C$ 的完整三角关系。

### 9.2 方法

结合实验 3（标注预算曲线）的数据，在同一图中展示：

**图 A**：$m$ vs $\rho$（标注预算 → defer 率），叠加 CRC 保证线 $\text{err}_{\text{accept}} \leq \alpha$

**图 B**：$m$ vs $C$（标注预算 → 总成本），标注最优 $m^*$

**图 C**：$m$ vs $\text{acc}_{\text{final}}$（标注预算 → 最终精度），用水平线标注 $1 - \alpha - \beta_3$（精度保证下界）

### 9.3 关键验证点

1. **$\rho(m)$ 单调不增**：更多训练 → 更少 defer
2. **每个 $m$ 下 $\text{acc}_{\text{final}} \geq 1 - \alpha - \beta_3$**：精度保证在每个标注预算水平都成立
3. **存在 $m^*$**：$C(m)$ 呈 U 形，有明确的最优标注预算

---

## 十、LROBench 实验

### 10.1 与 FEVER 的区别

LROBench 有 30 个不同 query，每个 query 的文档数约 39 条。这意味着：
- Per-query 训练集极小（每 query 可能只有 10-20 个训练样本）
- 校准集也极小（CRC 有限样本修正显著）
- 更适合测试 cross-query transfer 和极小数据场景

### 10.2 设置

两种模式：

**模式 A（per-query 独立训练）**：对每个 query 独立运行 CGSD，$m$ 按 query 文档数的比例分配。

**模式 B（cross-query transfer）**：前 15 个 query 独立训练 LoRA，后 15 个 query 从前者的平均 LoRA 参数初始化。

---

## 十一、实现检查清单

按以下顺序实现和验证：

**阶段 1：基础设施**
- [ ] Qwen3-0.6B 推理 pipeline：输入 prompt → 提取 "yes"/"no" logits → 计算 $\ell_i, R_i, \hat{y}_i$
- [ ] Qwen3-Embedding-0.6B embedding pipeline
- [ ] GPT-5 teacher 标注 pipeline（含 logprob 提取）
- [ ] CRC 校准函数实现
- [ ] k-Center Greedy 实现
- [ ] DBDS 完整实现
- [ ] LoRA SFT pipeline（基于 PEFT / Hugging Face）

**阶段 2：单组件验证**
- [ ] 验证 zero-shot 0.6B 在 FEVER 上的 $\text{acc}_{\text{raw}}$（预期 ~85%）
- [ ] 验证 CRC 校准输出的 $\hat{\lambda}$ 是否合理
- [ ] 验证 LoRA SFT 收敛（训练 loss 下降曲线）
- [ ] 验证 SFT 后 $\text{acc}_{\text{raw}}$ 提升（用随机 500 样本）

**阶段 3：核心实验**
- [ ] 实验 1（CGSD 有效性）→ 验证迭代循环工作正常
- [ ] 实验 2（数据选择对比）→ DBDS vs baselines
- [ ] 实验 3（预算曲线）→ $m$ vs $\rho$
- [ ] 实验 4（消融）→ band 比例、轮次、rank、加权

**阶段 4：完整对比与理论验证**
- [ ] 实验 5（端到端对比）→ 与前序工作对比
- [ ] 实验 6（CRC 验证）→ 20 次重复
- [ ] 实验 7（三角关系）→ 综合图表

---

## 十二、结果呈现模板

### 12.1 论文主表

论文核心表格应包含以下列：

| Method | Student model | Training data | $\text{acc}_{\text{raw}}$ | $\text{acc}_{\text{final}}$ | Defer % | Total $N_\mathcal{T}$ | Cost | Guarantee |
|--------|---------------|---------------|-----|------|---------|------|------|-----------|

### 12.2 论文核心图

**Figure 1**：架构图（即上方已生成的 CGSD 流程图）

**Figure 2**：每轮迭代的 $\rho^{(t)}$ 下降曲线（实验 1 结果），叠加 $\text{acc}_{\text{raw}}^{(t)}$ 的上升

**Figure 3**：$m$ vs $\rho$ 的 Pareto 曲线（实验 3 结果），对比不同选择策略

**Figure 4**：CRC 保证验证（实验 6 结果），名义 $\alpha$ vs 实际 wrong-accept 风险的 20 次散点，另附 accept 子集条件错误率作诊断

**Figure 5**：Band 比例消融的柱状图（实验 4a 结果），展示 "全边界 > 均匀 > 全深度"

### 12.3 附录表

- 各消融实验的完整数值结果
- LROBench 的详细 per-query 结果
- 成本拆分明细
