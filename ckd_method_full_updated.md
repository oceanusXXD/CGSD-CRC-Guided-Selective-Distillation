# CKD 方法完整流程：逐步文字说明

---

## 总体思路（一段话读完）

我们有一个大模型（teacher，如 GPT-5）和一个小模型（student，如 Qwen3-0.6B）。目标是让小模型尽量接近大模型的判断能力，同时尽可能少地调用大模型。方法分五步：（1）先拿出两小组数据让大模型标注，一组叫"引导集"用来在训练过程中发现小模型哪里不行，另一组叫"认证集"留到最后验证精度；（2）让小模型对所有数据做一遍预测，利用引导集上的 CRC 校准找出小模型"不靠谱"的那些样本（defer 集），并根据 CRC defer rate 与错误浓缩度自适应计算本轮 accept/defer 采样比例；（3）按这个比例从 accept 集和 defer 集中挑选训练样本，送给大模型标注；（4）用大模型标注的数据训练小模型，然后回到第 2 步重复；（5）训练结束后，在认证集上做最终校准，得到一个数学保证：部署后系统的错误率不超过用户设定的上限 α。

---

## Phase 0：准备阶段

### 第一步：把数据分成三份

我们手上有 N 个未标注的文档（比如 FEVER 的 16.5 万条）。在做任何事情之前，先随机地从中抽出两小批文档：

- **引导集**（$\mathcal{D}_{\text{guide}}$）：1000 个文档。这一批数据的作用是在训练过程中帮助我们判断"小模型在哪些文档上不靠谱"。我们把这 1000 个文档发给大模型，获取它们的正确标签（yes/no），并永久保存。

- **认证集**（$\mathcal{D}_{\text{cert}}$）：200 个文档。这一批数据的作用是在最后提供精度保证。同样发给大模型获取正确标签。**但关键区别是：认证集在整个训练过程中完全不参与任何决策——它被锁起来，直到最后一步才打开。** 这个隔离是精度保证成立的关键条件。

- **候选池**（$\mathcal{U}_{\text{pool}}$）：剩下的 N - 1200 个文档。这是我们后续要从中选择训练数据的池子，也是最终部署时小模型需要处理的数据。

### 第二步：计算所有文档的 embedding

对全部 N 个文档，用一个固定的 embedding 模型（如 Qwen3-Embedding-0.6B）计算每个文档与 query 拼接后的向量表示。这个 embedding 在后续所有轮次中保持不变，用于衡量文档之间的相似度，指导我们选择多样化的训练样本。这是一次性的计算开销。

### 第三步：选择温度参数 T

温度 T 控制路由分数的"灵敏度"。对每个样本，小模型先产生 logit margin：

$$
score_i=\log p_i(1)-\log p_i(0)
$$

预测为：

$$
\hat{y}_i=\mathbf{1}\{score_i>0\}
$$

路由分数定义为：

$$
R_i(T)=\sigma\left(\frac{|score_i|}{T}\right)
$$

其中 T 越大，$R_i(T)$ 越接近 0.5，模型置信度被压平；T 越小，$R_i(T)$ 越接近 1，模型置信度被放大。

需要注意的是，T 不只影响 CRC 阈值 $\hat{\lambda}$，也会影响候选池中被划为 defer 的比例。因为：

$$
R_i(T)\ge \hat{\lambda}
\Longleftrightarrow
|score_i|\ge T\cdot \operatorname{logit}(\hat{\lambda})
$$

所以真实起作用的是原始 score 空间里的 margin cutoff：

$$
\tau_{\text{crc}}=T\cdot \operatorname{logit}(\hat{\lambda})
$$

选定 T 后，后续所有轮次中固定使用同一个 T。每次实验需要记录：

$$
T,\quad \hat{\lambda},\quad \tau_{\text{crc}}
$$

这样可以区分：是 CRC 阈值变化导致 defer 集变化，还是 T 改变了 score 到 routing score 的映射。

---

## Phase 1-3 迭代（重复 2-3 轮）

每一轮做以下六件事：

### 第一件事：小模型对所有文档做预测

用当前的小模型对候选池和引导集中的每个文档做推理。具体地，对每个文档 $d_i$，给小模型一个提示："Query: ... Document: ... 这个文档满足 query 吗？回答 yes 或 no。"

小模型在生成输出时，我们不只看它最终说了"yes"还是"no"，还要看它内部有多确信。具体做法是：在模型输出第一个 token 的位置，读取"yes"和"no"两个词的 log-probability，计算它们的差值。这个差值叫做 **logit margin**（记为 $score_i$ 或 $\ell_i$）：

- $\ell_i > 0$：模型倾向于说"yes"，$\ell_i$ 越大越确信
- $\ell_i < 0$：模型倾向于说"no"，$|\ell_i|$ 越大越确信
- $\ell_i \approx 0$：模型完全不确定

然后把 $|\ell_i|$ 通过带温度的 sigmoid 函数映射到 0.5 到 1 之间，得到 **路由分数** $R_i(T)$。$R_i(T)$ 接近 1 表示小模型很确信（不论预测 yes 还是 no），$R_i(T)$ 接近 0.5 表示小模型完全不确定。

**第一轮（$t=0$）** 的特殊性：此时小模型还没经过任何训练（zero-shot），因此需要对全部候选池做推理。**后续轮次的优化**：由于 LoRA 训练的参数变化极小，远离训练区域的文档的预测几乎不变。因此后续轮次只需要对上一轮 defer 集中的文档重新推理，accept 集的路由分数可以复用。这大幅减少了推理成本。

### 第二件事：用 CRC 在引导集上画出"靠谱线"

现在我们有了引导集中每个文档的路由分数 $R_j(T)$、小模型预测 $\hat{y}_j$，以及大模型给出的标签 $y_j$。我们要用 CRC 找一个阈值 $\hat{\lambda}_{\text{guide}}$，使得被 accept 的错误样本在整个引导集上的错误质量不超过用户设定的风险预算 $\alpha$。

这里需要注意：当前 CRC 口径控制的是 **wrong-accept / 全引导集大小**，而不是 accept 子集内部错误率。也就是说，它控制的是：

$$
\frac{1}{n_g}\sum_{j=1}^{n_g}
\mathbf{1}\{R_j(T)\ge \lambda\}
\mathbf{1}\{\hat{y}_j\ne y_j\}
$$

具体做法如下：

1. 准备一个候选阈值列表，从 0.50 到 1.00，每隔 0.01 一个，共 51 个候选值。

2. 对每个候选阈值 $\lambda$，在引导集上计算 CRC loss：

$$
L_j(\lambda)
=
\mathbf{1}\{R_j(T)\ge \lambda\}
\mathbf{1}\{\hat{y}_j\ne y_j\}
$$

3. 计算 empirical risk：

$$
\widehat{risk}(\lambda)
=
\frac{1}{n_g}
\sum_{j=1}^{n_g}L_j(\lambda)
$$

4. 做有限样本修正：

$$
risk\_bound(\lambda)
=
\frac{n_g}{n_g+1}\widehat{risk}(\lambda)
+
\frac{1}{n_g+1}
$$

5. 从小到大扫描 $\lambda$，找到第一个使下式成立的阈值：

$$
risk\_bound(\lambda)\le \alpha
$$

这就是：

$$
\hat{\lambda}_{\text{guide}}
$$

$\hat{\lambda}_{\text{guide}}$ 的直觉含义是：路由分数在这条线以上的文档被小模型直接 accept 时，wrong-accept 风险受到 CRC 控制；路由分数低于这条线的文档进入 defer 集，被视为小模型当前不可靠的区域。

### 第三件事：识别 defer 集，并计算自适应 accept/defer 采样比例

有了 $\hat{\lambda}_{\text{guide}}$ 后，我们把候选池中所有路由分数低于该阈值的文档归入 **defer 集**：

$$
D_{\text{defer}}
=
\{x_i\in \mathcal{U}_{\text{pool}}: R_i(T)<\hat{\lambda}_{\text{guide}}\}
$$

其余样本归入 **accept 集**：

$$
D_{\text{accept}}
=
\{x_i\in \mathcal{U}_{\text{pool}}: R_i(T)\ge\hat{\lambda}_{\text{guide}}\}
$$

然后计算候选池上的 CRC defer rate：

$$
r_U=
\frac{|D_{\text{defer}}|}{|\mathcal{U}_{\text{pool}}|}
=
\frac{1}{|\mathcal{U}_{\text{pool}}|}
\sum_{x_i\in \mathcal{U}_{\text{pool}}}
\mathbf{1}\{R_i(T)<\hat{\lambda}_{\text{guide}}\}
$$

只看 $r_U$ 还不够。因为 $r_U$ 会受到温度 T 和候选池 score 分布的影响。如果 T 让 defer 集变得很小，简单使用 $r_U+(1-r_U)^2$ 可能会把训练集推成极端 defer-heavy，导致 accept anchor 太少，训练不稳定。

因此我们额外在引导集上估计 defer 集是否真的浓缩错误。

引导集整体错误率：

$$
e_{\text{all}}
=
\frac{1}{n_g}
\sum_{j=1}^{n_g}
\mathbf{1}\{\hat{y}_j\ne y_j\}
$$

引导集 defer rate：

$$
r_C=
\frac{1}{n_g}
\sum_{j=1}^{n_g}
\mathbf{1}\{R_j(T)<\hat{\lambda}_{\text{guide}}\}
$$

引导集 defer 内部错误率：

$$
e_{\text{defer}}
=
\frac{
\sum_{j=1}^{n_g}
\mathbf{1}\{R_j(T)<\hat{\lambda}_{\text{guide}}\}
\mathbf{1}\{\hat{y}_j\ne y_j\}
}{
\sum_{j=1}^{n_g}
\mathbf{1}\{R_j(T)<\hat{\lambda}_{\text{guide}}\}
}
$$

错误浓缩比：

$$
c_{\text{crc}}
=
\frac{e_{\text{defer}}}{e_{\text{all}}}
$$

其中：

- $c_{\text{crc}}\approx1$：defer 集没有明显比整体更容易错，不应该强烈过采样；
- $c_{\text{crc}}>1$：defer 集确实浓缩了错误，可以提高 defer 采样比例；
- $c_{\text{crc}}$ 越大，说明 defer 集越有训练价值。

如果引导集上没有 defer 样本，或整体错误率为 0，则令 $c_{\text{crc}}=1$、$\eta_{\text{crc}}=0$，表示不额外 boost defer。

然后把错误浓缩比转换成 defer boost 系数：

$$
\eta_{\text{crc}}
=
\operatorname{clip}
\left(
\frac{\log c_{\text{crc}}}{\log(1/r_C)},
0,
1
\right)
$$

其中 $\operatorname{clip}(x,0,1)$ 表示把数值限制在 $[0,1]$ 区间内。

最终训练集采样比例定义为：

$$
s_{\text{defer}}
=
r_U+\eta_{\text{crc}}(1-r_U)^2
$$

$$
s_{\text{accept}}
=
1-s_{\text{defer}}
$$

给定本轮标注预算 $B_t$，本轮从 defer 和 accept 中分别采样：

$$
B_{\text{defer}}=\operatorname{round}(B_t\cdot s_{\text{defer}})
$$

$$
B_{\text{accept}}=B_t-B_{\text{defer}}
$$

直觉上，$r_U$ 表示当前 pool 中小模型不可靠区域有多大；$c_{\text{crc}}$ 表示这个不可靠区域是否真的浓缩错误；$\eta_{\text{crc}}$ 决定 defer boost 的强弱。这样可以避免只由 $r_U$ 决定采样比例，从而降低温度 T 造成的比例不稳定。

**为什么 CRC defer 集比简单的"选 margin 最低的 20%"更好？** 因为 CRC 阈值是在真实标签上校准的，它控制的是 wrong-accept 风险，而不是 margin 的百分位数。如果小模型的校准很差（比如对某些类型的文档过度自信但经常预测错），简单的 margin 百分位数会遗漏那些"高 margin 但预测错"的文档。CRC 阈值更贴近真实错误风险。

### 第四件事：按自适应比例从 accept/defer 中选择要标注的样本

我们的标注预算有限（比如总共 500 个样本，分三轮，第一轮 250 个，第二轮 150 个，第三轮 100 个）。每一轮先根据上一节的公式计算：

$$
B_{\text{accept}},\quad B_{\text{defer}}
$$

然后分别从 accept 集和 defer 集中选择样本。

**defer 部分**：defer 集是小模型当前不可靠的区域，优先提供 hard / informative samples。可以使用两种策略：

1. **defer random**：从 defer 集中随机采样 $B_{\text{defer}}$ 个样本。这个版本最稳定，适合作为主实验。
2. **defer k-Center**：在 defer 集的 embedding 空间中做 k-Center Greedy，选择尽可能分散的 $B_{\text{defer}}$ 个样本。这个版本用于测试 diversity 是否带来额外收益。

k-Center Greedy 的过程是：

1. 首先选离所有 defer 样本的质心最远的一个样本作为起点。
2. 然后逐个加入：每次选择离已选集合中最近点距离最大的样本。
3. 重复直到选够 $B_{\text{defer}}$ 个样本。

**accept 部分**：accept 集提供 easy / anchor samples，防止训练集过度偏向 hard cases。accept 样本可以从 accept 集中随机选择 $B_{\text{accept}}$ 个，也可以优先选择路由分数较高的 high-confidence accept 样本。

#### 消融策略：NS-difficulty global 选样

上面的主流程先用 CRC 把候选池分成 accept/defer，再按
$B_{\text{accept}}$ 和 $B_{\text{defer}}$ 分别选样。为了验证
neighbor support 本身是否能独立提供有效的难度信号，可以加入一个
**NS-difficulty global** 消融：不先划分 accept/defer，而是在整个
$\mathcal{U}_{\text{pool}}$ 上按局部难度分数直接选择 $B_t$ 个样本。

这个消融的关键点是：难度分数的采样比例不能手写成固定的
"easy 20%、middle 60%、hard 20%"。分数分布会随模型、数据集和温度 T
变化，固定分位数会引入新的不可解释超参数。因此 global 难度选样应
使用前面 CRC 已经估计出的错误浓缩量来自动决定"应该偏向难样本到什么
程度"。

首先，对任意候选样本 $x_i$，用全部引导集作为支持库。设 $z_i$ 是样本
embedding，$z_j$ 是引导集样本 embedding，相似度权重为：

$$
w_{ij}
=
\max\{\cos(z_i,z_j),0\}
$$

为了让局部准确率对应到当前 student 的预测方向，只使用与 $x_i$ 预测
相同的 guide 邻居：

$$
\mathcal{N}_i
=
\{j\in \mathcal{D}_{\text{guide}}:\hat{y}_j=\hat{y}_i\}
$$

于是 $x_i$ 的 guide 局部准确率定义为：

$$
a_i
=
\frac{
\sum_{j\in\mathcal{N}_i}w_{ij}\mathbf{1}\{\hat{y}_j=y_j\}
}{
\sum_{j\in\mathcal{N}_i}w_{ij}
}
$$

若分母为 0，则不强行相信这个样本很难或很简单，而是直接令它的局部难度
回退到整体错误率 $e_{\text{all}}$；等价地，局部准确率回退到
$1-e_{\text{all}}$。实际实现中可以记录为低支持样本，并在采样权重中只
给它平均难度。局部难度为：

$$
d_i=1-a_i
$$

但 $d_i$ 只是一个相似邻域错误率的原始估计。为了避免 embedding 相似度
未校准导致过度相信某些极端分数，需要在 guide 上做一次 leave-one-out
校准。对每个 guide 样本 $g_j$，用其他 guide 样本计算 $d_j^{(-j)}$，
并记真实错误标签：

$$
u_j=\mathbf{1}\{\hat{y}_j\ne y_j\}
$$

然后用单调校准把 $d_j^{(-j)}$ 映射成局部错误概率：

$$
\hat{p}_{\text{err}}(d)
\approx
\Pr(\hat{y}\ne y\mid d)
$$

这里推荐使用 isotonic / PAV 单调回归，因为它不需要预设 bucket 个数，
只利用"难度越高，错误概率不应越低"这个顺序假设。这样每个 pool 样本
都得到一个校准后的难度概率：

$$
p_i=\hat{p}_{\text{err}}(d_i)
$$

接下来用 CRC 诊断量自动确定本轮训练集的目标错误密度。先由
$e_{\text{all}}$、$r_C$ 和 $e_{\text{defer}}$ 推出 guide accept 区域的
错误率：

$$
e_{\text{accept}}
=
\operatorname{clip}
\left(
\frac{
e_{\text{all}}-r_Ce_{\text{defer}}
}{
1-r_C
},
0,
1
\right)
$$

若 $r_C=1$，则退化为 $e_{\text{accept}}=e_{\text{all}}$。前面已经得到
CRC error-mass 的采样比例 $s_{\text{accept}}$、$s_{\text{defer}}$，
因此 global 难度选样的目标错误密度定义为：

$$
e_{\text{target}}
=
s_{\text{accept}}e_{\text{accept}}
+
s_{\text{defer}}e_{\text{defer}}
$$

这个式子的含义是：即使 global 消融不显式划分 accept/defer，它仍然
继承 CRC 对"本轮应该多关注多少错误区域"的估计。若 defer 并没有错误
浓缩（$c_{\text{crc}}\approx1$），则 $s_{\text{defer}}$ 不会被明显放大，
$e_{\text{target}}$ 会接近整体错误率，global 选样自然退化到接近 random。
若 defer 明显浓缩错误，则 $e_{\text{target}}$ 会高于 $e_{\text{all}}$，
但通常低于 $e_{\text{defer}}$，避免全取最难样本。

最后，用 $e_{\text{target}}$ 反推出采样权重，而不是手写分位比例。令

$$
\epsilon=\frac{1}{n_g+1}
$$

这个 $\epsilon$ 沿用 CRC 有限样本修正中的尺度。对每个 pool 样本定义：

$$
\tilde{w}_i(\beta)
=
\left(
\frac{p_i+\epsilon}{e_{\text{all}}+\epsilon}
\right)^\beta
$$

其中 $\beta$ 控制相对 random 的难度偏置，$\beta=0$ 时所有样本等权，
即退化为 random。$\beta$ 不是手动超参数，而是通过一维搜索自动确定，
使加权候选池的平均难度接近 $e_{\text{target}}$：

$$
\frac{\sum_i \tilde{w}_i(\beta)p_i}{\sum_i \tilde{w}_i(\beta)}
\approx
e_{\text{target}}
$$

为了防止全取最难样本或被噪声 hard cases 主导，使用 CRC 错误浓缩比
$c_{\text{crc}}$ 作为权重上限和下限：

$$
w_i(\beta)
=
\operatorname{clip}
\left(
\tilde{w}_i(\beta),
\frac{1}{c_{\text{crc}}},
c_{\text{crc}}
\right)
$$

如果 $c_{\text{crc}}\le1$，则所有权重直接设为 1，表示当前 guide 没有
证据支持难例过采样。给定最终权重 $w_i$ 后，从整个
$\mathcal{U}_{\text{pool}}$ 中按无放回 PPS（probability proportional to
size）采样 $B_t$ 个样本。输出时需要记录实际选中集合的平均
$p_i$，用于检查是否接近 $e_{\text{target}}$。

这一路径形成独立消融：

$$
S_t^{\text{NS-global}}
=
\operatorname{PPSSample}
\left(
\mathcal{U}_{\text{pool}},
B_t,
w_i(\beta)
\right)
$$

它回答的问题是：**只使用 guide-NS 局部难度，不显式使用 accept/defer
分区，是否已经能比 random 更有效？** 如果这个消融有效，再进入下一步
组合消融：保留 CRC 的 $B_{\text{accept}}/B_{\text{defer}}$ 预算，在
accept 和 defer 两个集合内部各自按 NS error mass 做 PPS 选样。

对主流程而言，本轮训练候选集仍然由 accept 部分和 defer 部分组成：

$$
S_t
=
S_{\text{accept},t}
\cup
S_{\text{defer},t}
$$

其中：

$$
|S_{\text{accept},t}|=B_{\text{accept}},
\quad
|S_{\text{defer},t}|=B_{\text{defer}}
$$

NS error mass 版本不再沿用 global 消融里的
$e_{\text{target}}/\beta$ 目标均值控制。原因是 accept 或 defer 子集的
平均 $p_i$ 可能已经高于目标均值，此时目标均值逻辑会把
$\beta$ 设为 0，导致该子集退化成均匀随机。组合策略里 CRC 已经负责了
accept/defer 的外层预算，NS 只需要决定子集内部样本的相对概率：

$$
\Pr(i\mid h)
=
\frac{p_i}{\sum_{j\in\mathcal{U}_h}p_j},
\quad
h\in\{\text{accept},\text{defer}\}
$$

其中 $\mathcal{U}_h$ 是对应 CRC 子集，$p_i$ 是 guide-NS 单调校准后的
错误概率。这样任意难度区域 $A\subseteq\mathcal{U}_h$ 的期望样本数为：

$$
\mathbb{E}[|S_h\cap A|]
=
B_h
\frac{\sum_{i\in A}p_i}{\sum_{j\in\mathcal{U}_h}p_j}
$$

因此不同难度的样本都会以其 estimated error mass 自然进入训练集，不需要
手写 bucket 数、分位比例或 top-k 阈值。

**为什么不只选 defer？** 纯 defer 采样会让训练集过度 hard，尤其当 $r_U$ 很小但公式把 defer 大幅放大时，accept anchor 太少会导致模型输出分布变保守或训练不稳定。保留一部分 accept 样本，可以稳定原始分布，防止 recall 下降。

**为什么不直接选最不确信的？** 如果只选路由分数最低的样本，它们可能集中在 embedding 空间的某个小区域。k-Center 可以保证 defer 样本覆盖 defer 集的不同区域。但 k-Center 也可能选到低密度或非典型样本，因此它应作为 ablation，而不是默认替代 random defer。

**为什么要多选一些（buffer）？** 如果下一步要做 teacher confidence 过滤，可以对 $S_t$ 多选 25% 的 buffer 样本。例如本轮需要 250 个样本，可以先选到 312 个，再经过 teacher confidence 过滤保留最可靠的 250 个。

### 第五件事：大模型标注 + confidence 过滤

把上一步选出的候选样本发给大模型标注。大模型对每个文档给出标签（yes/no），同时我们获取大模型的 logprob——也就是大模型有多确信它的判断。

**Teacher confidence 过滤的逻辑**：

- 大模型确信（logprob margin 大）的样本：标签可靠，而且很可能是小模型"能学会"的（大模型觉得不难，小模型训练后也应该能掌握）。
- 大模型也不确信（logprob margin 小）的样本：标签可能不准确，而且可能是本质上模糊的——即使训练小模型也学不好。

我们按大模型的确信度从高到低排序，保留最确信的 $m_t$ 个（本轮标注量）。被过滤掉的 buffer 样本就丢弃了。

**理论依据**：定理 1 的式 (13) 表明，如果标签错误（大模型标错了），训练在该样本上会**损害**小模型——效果等同于引入异类噪声。过滤掉大模型不确信的样本，就是在减少标签错误的风险。

**如果大模型 API 不提供 logprob**：跳过过滤步骤，直接使用所有候选样本。此时不需要 buffer。

### 第六件事：训练小模型

把本轮和之前所有轮次累积的标注数据合在一起，从小模型的原始基座参数重新做 SFT 训练。训练数据由两部分组成：一部分是 accept anchor 样本，另一部分是 defer hard 样本。两者比例由 CRC defer rate 和 CRC 错误浓缩度共同决定，而不是固定写死为某个 accept/defer 比例。

如果本轮选出的训练集正负例比例明显偏移，可以在 LoRA 训练中加入 class weight 作为 ablation；但选样阶段不应使用真实标签强行控制正负例平衡，因为真实部署场景中没有 gold label。

**为什么从头训而不是接着上一轮训（continual training）？** 从头训保证了模型是训练数据的"纯函数"——不存在上一轮的残留影响。这使得理论分析更简洁。而且 LoRA rank-1 训练非常快（500 样本约 1-2 分钟），从头训的成本几乎可以忽略。

训练完成后，我们得到一个更强的小模型。然后**回到第一件事**，用这个更强的模型重新推理、重新识别 defer 集、重新选样本——如此循环。

**每轮循环的效果**：小模型在 defer 集上的准确率提升 → defer 集缩小 → 下一轮选择的样本更加精准（因为剩下的 defer 样本是真正的难点）。这就是闭环的核心逻辑。

---

## Phase 4：最终认证

达到预设训练轮数，或 defer 率不再显著下降后，小模型参数固定为 $\theta^*$。

现在打开之前锁起来的**认证集**。用最终的小模型 $\theta^*$ 对认证集中的 200 个文档做推理，然后执行 CRC 校准——过程与引导集上的完全相同（扫描阈值、计算修正后的错误率、选最小可行阈值），但这次得到的阈值 $\hat{\lambda}^*$ 具有**数学保证**：

$$
\mathbb{E}[\text{accept 错误率}] \leq \alpha
$$

这个保证成立的原因：认证集从未参与过任何中间决策——它没有影响引导集的 CRC 阈值、没有影响 defer 集的定义、没有影响训练数据的选择、没有影响模型的训练。因此认证集与最终模型是完全独立的，CRC 定理的前提条件严格满足。

**引导集为什么不能提供同样的保证？** 因为引导集在中间轮的 CRC 校准中被反复使用，间接影响了训练数据的选择和模型的训练。模型"见过"引导集的信息（虽然是间接的），因此引导集不再是独立的。

---

## Phase 5：部署

对候选池中剩余的每个文档 $d_i$：

1. 用最终小模型推理，得到预测标签 $\hat{y}_i$ 和路由分数 $R_i$。
2. 如果 $R_i \geq \hat{\lambda}^*$：**accept**——直接输出小模型的预测 $\hat{y}_i$。
3. 如果 $R_i < \hat{\lambda}^*$：**defer**——调用大模型获取标签并输出。

对于已经在训练过程中被大模型标注过的样本（$S_{\text{train}}$ 中的）：直接输出已有的大模型标签，不需要再次调用。

---

## 各阶段的成本汇总

以 FEVER（N = 165,447）为例：

| 阶段               | 操作                                    | 大模型调用次数               | 小模型调用次数         |
| ------------------ | --------------------------------------- | ---------------------------- | ---------------------- |
| Phase 0            | 引导集+认证集标注                       | 1200                         | 0                      |
| Phase 0            | Embedding 计算                          | 0                            | N（用 embedding 模型） |
| Phase 1 轮 0       | 小模型全量推理                          | 0                            | N                      |
| Phase 1 轮 1-2     | 小模型 defer 集推理                     | 0                            | ~0.2N × 2              |
| Phase 2 每轮       | 引导集推理                              | 0                            | 1000 × 3               |
| Phase 4            | 认证集推理                              | 0                            | 200                    |
| Phase 5 蒸馏标注   | 自适应 accept/defer 选样 + teacher 标注 | ~625（含 buffer）            | 0                      |
| Phase 5 部署 defer | 残余 defer                              | $n_{\text{def}}$（目标 <2%） | 0                      |
| **合计**           |                                         | **~925 + $n_{\text{def}}$**  | **~1.6N**              |

对比全大模型方案：大模型调用 165,447 次。CKD 的大模型调用仅约 1,825 + 3,300 ≈ 5,125 次（假设最终 defer 率 2%），节省 96.9%。

---

## 为什么每一步是这样设计的？简明理由表

| 设计决策                                                  | 为什么                                                                                                              |
| --------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------- |
| 拆引导集和认证集                                          | 引导集参与中间决策，不能提供最终保证；认证集隔离保证保证的严格性                                                    |
| 用 CRC 阈值（而非 margin 百分位）定义 defer 集            | CRC 控制 wrong-accept 风险；阈值基于真实标签校准，不是任意 margin 百分位                                            |
| 用 $r_U$ 和 $c_{\text{crc}}$ 自适应决定 accept/defer 比例 | $r_U$ 表示当前 pool 中 defer 区域大小；$c_{\text{crc}}$ 表示 defer 是否真的浓缩错误；二者共同决定是否应过采样 defer |
| 保留 accept anchor                                        | 避免训练集过度 hard，降低 recall 掉、输出分布变保守的风险                                                           |
| defer random / defer k-Center                             | random defer 作为稳定主线；k-Center 用于测试 diversity 是否带来额外收益                                             |
| NS-difficulty global 消融                                 | 不手写 hard/easy 分位比例；用 guide 上的局部错误率和 CRC 推出的 $e_{\text{target}}$ 自动决定难度偏置                 |
| Teacher confidence 过滤                                   | 过滤标签不可靠的样本，减少异类噪声对训练的损害（定理 1 式 13）                                                      |
| 从基座重训（而非 continual training）                     | 保证模型是训练数据的纯函数，理论分析简洁，成本可忽略                                                                |
| 每轮只对 defer 集重推理                                   | Accept 集的预测几乎不受 LoRA 微调影响（kernel 局部性），节省推理成本                                                |
| 认证集在最后才使用                                        | 模型不依赖认证集 → CRC 的可交换性条件成立 → 精度保证有效                                                            |

---

## 建议新增记录字段

每一轮实验建议记录：

$$
T,\quad \alpha,\quad \hat{\lambda},\quad \tau_{\text{crc}},\quad r_U,\quad r_C,\quad e_{\text{all}},\quad e_{\text{defer}},\quad c_{\text{crc}},\quad \eta_{\text{crc}},\quad s_{\text{accept}},\quad s_{\text{defer}}
$$

这里记录的是 CRC 划分和错误浓缩度诊断量，不记录本轮选样预算。实际选样数量
$B_{\text{accept}}$ 和 $B_{\text{defer}}$ 只在选样阶段由目标训练样本数临时计算，
不参与 CRC 阈值校准，也不写入校准摘要。

对应表格字段：

| 字段         | 含义                                         |
| ------------ | -------------------------------------------- |
| `T`          | routing score 温度                           |
| `alpha`      | CRC 风险预算                                 |
| `lambda_hat` | CRC 选出的路由阈值                           |
| `tau_crc`    | 原始 score 空间阈值，$T\cdot logit(\lambda)$ |
| `r_U`        | pool 上 defer rate                           |
| `r_C`        | guide/calibration set 上 defer rate          |
| `e_all`      | guide/calibration set 整体错误率             |
| `e_defer`    | guide/calibration set defer 内部错误率       |
| `c_crc`      | defer 错误浓缩比                             |
| `eta_crc`    | defer boost 系数                             |
| `s_accept`   | 最终 accept 采样比例                         |
| `s_defer`    | 最终 defer 采样比例                          |

如果启用 NS-difficulty global 选样，还需要额外记录：

$$
e_{\text{accept}},\quad e_{\text{target}},\quad
\epsilon,\quad \beta,\quad c_{\text{crc}},\quad
\overline{p}_{\text{pool}},\quad
\overline{p}_{\text{selected}},\quad
w_{\min},\quad w_{\max}
$$

对应表格字段：

| 字段                    | 含义                                                                 |
| ----------------------- | -------------------------------------------------------------------- |
| `ns_score_source`        | 难度来源，例如 `guide_leave_one_out_isotonic`                        |
| `ns_e_accept`            | 由 $e_{\text{all}}$、$r_C$、$e_{\text{defer}}$ 推出的 accept 错误率   |
| `ns_e_target`            | CRC 推出的目标训练集错误密度                                         |
| `ns_epsilon`             | 有限样本平滑项，默认 $1/(n_g+1)$                                     |
| `ns_beta`                | 一维搜索得到的难度偏置指数，不手动指定                               |
| `ns_weight_floor`        | 采样权重下限，默认 $1/c_{\text{crc}}$                                |
| `ns_weight_cap`          | 采样权重上限，默认 $c_{\text{crc}}$                                  |
| `ns_pool_mean_p_error`   | 全 pool 的平均校准局部错误概率                                       |
| `ns_selected_mean_p_error` | 选中训练集的平均校准局部错误概率，应接近 `ns_e_target`              |
| `ns_selected_easy_count` | 仅用于诊断：选中样本中低于 pool 中位 $p_i$ 的数量                    |
| `ns_selected_hard_count` | 仅用于诊断：选中样本中高于 pool 中位 $p_i$ 的数量                    |
| `ns_weighting`           | `target_mean` 表示 global 的目标均值权重；`ns-error-mass` 表示 split 内 $P(i)\propto p_i$ |

---
## 训练配置字段
lora_r: 8
lora_alpha: 16
lora_dropout: 0.05
target_modules:
  - q_proj
  - k_proj
  - v_proj
  - o_proj

learning_rate_0_6b: 1.0e-4

num_train_epochs: 4
per_device_train_batch_size: 4
gradient_accumulation_steps: 4
effective_batch_size: 16

weight_decay: 0.01
warmup_ratio: 0.03
lr_scheduler_type: cosine
max_seq_length: 4096
bf16: true
gradient_checkpointing: true
