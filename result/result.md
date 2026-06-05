# 结果

在当前实验里，difficulty-aware sampling 到底学到了任务难度，还是主要改变了 0.6B student 的输出先验。整体结论是：对 Qwen3-0.6B non-thinking student 来说，普通二分类任务上的训练后行为主要由训练集中反向于原始预测偏置的样本数量决定；accept/defer、base-correct/base-error、hard/easy 来源本身不是稳定主因。FEVER 和 1.7B 结果显示，当 student 容量更高、原始偏置更低时，难样本选择才出现小幅正效应。

## 1. 原始预测偏置

表 1 显示，0.6B 在 IMDb、TwitterHate 和 Codebase 上几乎总是预测 yes；FEVER 上则相反，0.6B 明显偏 no。这里的反偏置标签指的是训练中能够抵消 base 输出偏置的标签方向：base 偏 yes 时是 no，base 偏 no 时是 yes。

**表 1. 原始 0.6B 预测偏置。**

| 数据/问题 | 模型 | 样本数 | 真实 yes % | base 预测 yes % | base Macro-F1 | base 偏置 | 反偏置标签 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| IMDb q1 positive | Qwen3-0.6B | 49990 | 49.99 | 99.99 | 33.35 | yes | no |
| IMDb q2 negative | Qwen3-0.6B | 49990 | 27.49 | 100 | 21.57 | yes | no |
| TwitterHate | Qwen3-0.6B | 17348 | 83.2 | 100 | 45.42 | yes | no |
| Codebase q1 social link | Qwen3-0.6B | 9298 | 6.25 | 93.82 | 12.3 | yes | no |
| Codebase q2 CS interest | Qwen3-0.6B | 9298 | 61.95 | 99.45 | 39.6 | yes | no |
| Codebase q3 factual ID | Qwen3-0.6B | 9298 | 25.75 | 99.88 | 20.65 | yes | no |
| FEVER support | Qwen3-0.6B | 165447 | 52.4 | 10.01 | 44.08 | no | yes |
| FEVER support | Qwen3-1.7B reference | 165447 | 52.4 | 58.94 | 86.72 | 接近平衡 |  |

这一步说明，0.6B 在多数普通二分类 query 上不是可靠分类器，而是强输出先验模型。后续训练结果必须和这个原始偏置一起解释，否则很容易把标签先验修正误读成 difficulty-aware learning。

## 2. 训练集构成

第二步看每个训练集实际选到了什么。表 2 同时列出 label yes/no、accept/defer、base-error 和反偏置数量。这个表是机制分析的核心：如果两个方法最终表现不同，首先要看它们是否选到了不同数量的反偏置样本。

**表 2. 训练集 yes/no、accept/defer 与反偏置样本构成。**

| 实验组 | 数据/设置 | 方法 | 训练数 | base 偏置 | 反偏置标签 | 反偏置数 | 反偏置 % | 训练 yes % | 训练 no % | accept | defer | base-error % |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| IMDb full-test | IMDb q1 | pool-random | 2500 | yes | no | 1231 | 49.24 | 50.76 | 49.24 | 1499 | 1001 | 49.2 |
| IMDb full-test | IMDb q1 | ns-error-mass | 2500 | yes | no | 2316 | 92.64 | 7.36 | 92.64 | 819 | 1681 | 92.6 |
| IMDb full-test | IMDb q2 | pool-random | 2500 | yes | no | 1805 | 72.2 | 27.8 | 72.2 | 697 | 1803 | 72.2 |
| IMDb full-test | IMDb q2 | ns-error-mass | 2500 | yes | no | 1975 | 79 | 21 | 79 | 577 | 1923 | 79 |
| Base-correct/wrong control | Codebase q1 | base-correct-balanced | 500 | yes | no | 250 | 50 | 50 | 50 |  |  | 0 |
| Base-correct/wrong control | Codebase q2 | base-correct-random | 500 | yes | no | 4 | 0.8 | 99.2 | 0.8 |  |  | 0 |
| Base-correct/wrong control | Codebase q2 | base-wrong-random | 500 | yes | no | 500 | 100 | 0 | 100 |  |  | 100 |
| FEVER 0.6B budget | FEVER n=1500 | pool-random | 1500 | no | yes | 785 | 52.33 | 52.33 | 47.67 | 434 | 1066 | 46.33 |
| FEVER 0.6B budget | FEVER n=1500 | ns-error-mass | 1500 | no | yes | 992 | 66.13 | 66.13 | 33.87 | 380 | 1120 | 65.6 |
| FEVER 0.6B budget | FEVER n=3000 | pool-random | 3000 | no | yes | 1568 | 52.27 | 52.27 | 47.73 | 848 | 2152 | 47.57 |
| FEVER 0.6B budget | FEVER n=3000 | ns-error-mass | 3000 | no | yes | 2041 | 68.03 | 68.03 | 31.97 | 760 | 2240 | 67.37 |
| FEVER 0.6B budget | FEVER n=4500 | pool-random | 4500 | no | yes | 2366 | 52.58 | 52.58 | 47.42 | 1262 | 3238 | 47.44 |
| FEVER 0.6B budget | FEVER n=4500 | ns-error-mass | 4500 | no | yes | 3073 | 68.29 | 68.29 | 31.71 | 1140 | 3360 | 67.6 |
| FEVER 0.6B budget | FEVER n=6000 | pool-random | 6000 | no | yes | 3145 | 52.42 | 52.42 | 47.58 | 1670 | 4330 | 47.55 |
| FEVER 0.6B budget | FEVER n=6000 | ns-error-mass | 6000 | no | yes | 4092 | 68.2 | 68.2 | 31.8 | 1520 | 4480 | 67.48 |
| Low-resource 0.6B | codebase q2 n=125 | random | 125 | yes | no | 39 | 31.2 | 68.8 | 31.2 | 48 | 77 | 32 |
| Low-resource 0.6B | codebase q2 n=125 | crc-error-mass | 125 | yes | no | 41 | 32.8 | 67.2 | 32.8 | 41 | 84 | 32.8 |
| Low-resource 0.6B | codebase q3 n=250 | random | 250 | yes | no | 188 | 75.2 | 24.8 | 75.2 | 39 | 211 | 74.8 |
| Low-resource 0.6B | codebase q3 n=250 | crc-error-mass | 250 | yes | no | 192 | 76.8 | 23.2 | 76.8 | 40 | 210 | 76 |
| Low-resource 0.6B | twitter_hate q1 n=50 | random | 50 | yes | no | 11 | 22 | 78 | 22 | 41 | 9 | 22 |
| Low-resource 0.6B | twitter_hate q1 n=50 | crc-error-mass | 50 | yes | no | 17 | 34 | 66 | 34 | 27 | 23 | 34 |
| Low-resource 0.6B | twitter_hate q1 n=125 | random | 125 | yes | no | 27 | 21.6 | 78.4 | 21.6 | 102 | 23 | 21.6 |
| Low-resource 0.6B | twitter_hate q1 n=125 | crc-error-mass | 125 | yes | no | 35 | 28 | 72 | 28 | 67 | 58 | 28 |
| Low-resource 0.6B | twitter_hate q1 n=250 | random | 250 | yes | no | 47 | 18.8 | 81.2 | 18.8 | 203 | 47 | 18.8 |
| Low-resource 0.6B | twitter_hate q1 n=250 | crc-error-mass | 250 | yes | no | 69 | 27.6 | 72.4 | 27.6 | 134 | 116 | 27.6 |
| FEVER 0.6B formula500 | FEVER n=500 | defer_kcenter | 500 | no | yes | 333 | 66.6 | 66.6 | 33.4 | 65 | 435 |  |
| FEVER 0.6B formula500 | FEVER n=500 | random_defer | 500 | no | yes | 260 | 52 | 52 | 48 | 65 | 435 |  |

表 2 的模式很清楚：IMDb q1 中 ns-error-mass 选到 92.64% no，而 random 只有 49.24% no；FEVER 0.6B 中 ns-error-mass 在所有 budget 都选到更多 yes；base-wrong-random 则是 100% no。也就是说，不同方法名背后首先改变的是训练标签比例和反偏置样本数量。

## 3. 训练后结果

第三步看训练后输出。表 3 把训练构成和最终预测比例、性能放在一起。0.6B 的主要变化是评测集 pred yes 比例被训练集标签比例拉动：反偏置样本太少时仍保留原始偏置，反偏置样本太多时会向相反方向过校正。

**表 3. 训练后预测比例和结果。**

| 实验组 | 数据/设置 | 方法 | base 偏置 | 反偏置数 | 反偏置 % | 训练 yes % | accept/defer | 评测 pred yes % | 指标 | 分数 | 相对基线 | 解释 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| IMDb full-test | IMDb q1 | pool-random | yes | 1231 | 49.24 | 50.76 | 1499/1001 | 50.49 | macro-F1 | 94.37 | 0 | 随机基线 |
| IMDb full-test | IMDb q1 | ns-error-mass | yes | 2316 | 92.64 | 7.36 | 819/1681 | 45.59 | macro-F1 | 92.25 | -2.12 | 反偏置标签为 no；没有超过随机基线 |
| IMDb full-test | IMDb q2 | pool-random | yes | 1805 | 72.2 | 27.8 | 697/1803 | 26.87 | macro-F1 | 85.72 | 0 | 随机基线 |
| IMDb full-test | IMDb q2 | ns-error-mass | yes | 1975 | 79 | 21 | 577/1923 | 25.75 | macro-F1 | 85.54 | -0.18 | 反偏置标签为 no；没有超过随机基线 |
| Base-correct/wrong control | Codebase q1 | base-correct-balanced | yes | 250 | 50 | 50 |  | 22.87 | macro-F1 | 57.93 |  | 来源变化不如标签比例关键 |
| Base-correct/wrong control | Codebase q2 | base-correct-random | yes | 4 | 0.8 | 99.2 |  | 98.71 | macro-F1 | 42.49 |  | 来源变化不如标签比例关键 |
| Base-correct/wrong control | Codebase q2 | base-wrong-random | yes | 500 | 100 | 0 |  | 0 | macro-F1 | 26.77 |  | 来源变化不如标签比例关键 |
| FEVER 0.6B budget | FEVER n=1500 | pool-random | no | 785 | 52.33 | 52.33 | 434/1066 | 50.22 | macro-F1 | 92.24 | 0 | 随机基线 |
| FEVER 0.6B budget | FEVER n=1500 | ns-error-mass | no | 992 | 66.13 | 66.13 | 380/1120 | 50.61 | macro-F1 | 93.11 | 0.87 | 更多反偏置 yes；小幅提升但与标签比例耦合 |
| FEVER 0.6B budget | FEVER n=3000 | pool-random | no | 1568 | 52.27 | 52.27 | 848/2152 | 48.82 | macro-F1 | 92.5 | 0 | 随机基线 |
| FEVER 0.6B budget | FEVER n=3000 | ns-error-mass | no | 2041 | 68.03 | 68.03 | 760/2240 | 50.27 | macro-F1 | 93.51 | 1.01 | 更多反偏置 yes；小幅提升但与标签比例耦合 |
| FEVER 0.6B budget | FEVER n=4500 | pool-random | no | 2366 | 52.58 | 52.58 | 1262/3238 | 49.79 | macro-F1 | 93.03 | 0 | 随机基线 |
| FEVER 0.6B budget | FEVER n=4500 | ns-error-mass | no | 3073 | 68.29 | 68.29 | 1140/3360 | 50.91 | macro-F1 | 94.03 | 1 | 更多反偏置 yes；小幅提升但与标签比例耦合 |
| FEVER 0.6B budget | FEVER n=6000 | pool-random | no | 3145 | 52.42 | 52.42 | 1670/4330 | 50.21 | macro-F1 | 93.97 | 0 | 随机基线 |
| FEVER 0.6B budget | FEVER n=6000 | ns-error-mass | no | 4092 | 68.2 | 68.2 | 1520/4480 | 50.21 | macro-F1 | 94.47 | 0.5 | 更多反偏置 yes；小幅提升但与标签比例耦合 |
| Low-resource 0.6B | codebase q2 n=125 | random | yes | 39 | 31.2 | 68.8 | 48/77 | 87.08 | macro-F1 | 61.52 | 0 | 成对对比：反偏置数量改变输出先验 |
| Low-resource 0.6B | codebase q2 n=125 | crc-error-mass | yes | 41 | 32.8 | 67.2 | 41/84 | 72.31 | macro-F1 | 71.73 | 10.21 | 成对对比：反偏置数量改变输出先验 |
| Low-resource 0.6B | codebase q3 n=250 | random | yes | 188 | 75.2 | 24.8 | 39/211 | 5.81 | macro-F1 | 56.06 | 0 | 成对对比：反偏置数量改变输出先验 |
| Low-resource 0.6B | codebase q3 n=250 | crc-error-mass | yes | 192 | 76.8 | 23.2 | 40/210 | 2.58 | macro-F1 | 50.65 | -5.41 | 成对对比：反偏置数量改变输出先验 |
| Low-resource 0.6B | twitter_hate q1 n=50 | random | yes | 11 | 22 | 78 | 41/9 | 87.38 | macro-F1 | 65.94 | 0 | 成对对比：反偏置数量改变输出先验 |
| Low-resource 0.6B | twitter_hate q1 n=50 | crc-error-mass | yes | 17 | 34 | 66 | 27/23 | 45.37 | macro-F1 | 53.3 | -12.64 | 成对对比：反偏置数量改变输出先验 |
| Low-resource 0.6B | twitter_hate q1 n=125 | random | yes | 27 | 21.6 | 78.4 | 102/23 | 97.63 | macro-F1 | 55.7 | 0 | 成对对比：反偏置数量改变输出先验 |
| Low-resource 0.6B | twitter_hate q1 n=125 | crc-error-mass | yes | 35 | 28 | 72 | 67/58 | 93.33 | macro-F1 | 65.85 | 10.15 | 成对对比：反偏置数量改变输出先验 |
| Low-resource 0.6B | twitter_hate q1 n=250 | random | yes | 47 | 18.8 | 81.2 | 203/47 | 90.26 | macro-F1 | 71.59 | 0 | 成对对比：反偏置数量改变输出先验 |
| Low-resource 0.6B | twitter_hate q1 n=250 | crc-error-mass | yes | 69 | 27.6 | 72.4 | 134/116 | 87.47 | macro-F1 | 73.58 | 1.99 | 成对对比：反偏置数量改变输出先验 |
| FEVER 0.6B formula500 | FEVER n=500 | defer_kcenter | no | 333 | 66.6 | 66.6 | 65/435 | 79.97 | positive-F1 | 69.45 |  | 更多反偏置 yes，输出先验过校正 |
| FEVER 0.6B formula500 | FEVER n=500 | random_defer | no | 260 | 52 | 52 | 65/435 | 71.83 | positive-F1 | 76.47 |  | 标签比例更接近评测集，结果更好 |

IMDb 是一个直接例子：ns-error-mass 确实选到更多 base-error/counter-bias 样本，但 q1 的 Macro-F1 从 random 的 94.37 降到 92.25，q2 也从 85.72 降到 85.54。TwitterHate train=50 也是过校正例子：CRC 比 random 选到更多 no，eval pred yes 从 87.38% 降到 45.37%，Macro-F1 下降 12.64 点。FEVER 1500-6000 则是有限正例：更多 yes 反偏置样本把 pred yes 拉回 50% 左右，并带来 0.50 到 1.01 的小幅 Macro-F1 增益。

## 4. 容量更高时的 FEVER 结果

第四步单独看 1.7B。这里 base pred yes 为 58.94%，已经接近 FEVER 真实 yes 比例 52.40%，不再是 0.6B 那种严重偏置。此时 targeted selection 有小幅收益，但幅度仍然不大。

**表 4. FEVER 1.7B 容量检查。**

| 方法 | seed 数 | base pred yes % | 训练反偏置 no % | 训练 yes % | accept % | defer % | Acc | Macro-F1 | 相对 random | 解释 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| random | 2 | 58.94 | 47.87 | 52.13 | 94.77 | 5.23 | 79.58 | 79.08 | 0 | 随机基线 |
| crc-error-mass | 2 | 58.94 | 55.2 | 44.8 | 59.37 | 40.63 | 80.72 | 80.34 | 1.26 | 容量较足时有小幅收益 |
| ns-difficulty-global | 2 | 58.94 | 32.02 | 67.98 | 95.1 | 4.9 | 80.34 | 79.88 | 0.8 | 容量较足时有小幅收益 |
| ns-difficulty-crc-split | 2 | 58.94 | 55.15 | 44.85 | 60.7 | 39.3 | 80.03 | 79.59 | 0.51 | 容量较足时有小幅收益 |

1.7B 的结果支持容量依赖结论：当原始预测偏置较低、模型本身已有基本判别能力时，crc-error-mass、ns-difficulty-global 和 ns-difficulty-crc-split 分别比 random 高 1.26、0.80 和 0.51 Macro-F1。这个结果是正向的，但仍然只是小幅提升。

## 5. 补充小表

下面保留原先的小表，方便完整检查每组结果。四个大表给出主链条，小表用于展开每个诊断。

### 5.1 base-correct / base-wrong 控制

这个控制实验显示，样本来自 base-correct 还是 base-wrong 并不是主要因素；训练 label ratio 更关键。Codebase q2 的 base-correct-random 几乎全是 yes，训练后也几乎全预测 yes；base-wrong-random 全是 no，训练后也全预测 no。

| 问题 | 方法 | 训练 yes % | 测试 yes % | 评测 pred yes % | Acc | Macro-F1 | 说明 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| q1 | base |  | 3.76 | 96.31 | 7.32 | 7.32 | q1 held-out test 上的 base |
| q1 | base-correct-balanced | 50.00 | 3.76 | 22.87 | 80.58 | 57.93 | 250 个 base-correct label0 + 250 个 base-correct label1 |
| q2 | base |  | 63.44 | 99.43 | 63.86 | 40.24 | q2 held-out test 上的 base |
| q2 | base-correct-random | 99.20 | 63.44 | 98.71 | 64.70 | 42.49 | 只用 base 已经预测正确的样本训练 |
| q2 | base-wrong-random | 0.00 | 63.44 | 0.00 | 36.56 | 26.77 | 只用 base 预测错误的样本训练 |

### 5.2 FEVER 0.6B 1500-6000 详细表

这组结果展示有限正例：ns-error-mass 在每个 budget 都选到更多反向标签 yes，并带来小幅提升；同时，base false-negative 数量也同步上升。因此这里不能把提升完全归因于难样本来源，标签比例和 base-error 来源是耦合的。

| 训练数 | 方法 | 反向标签 yes 数 | 反向标签 yes % | base false-negative 数 | 训练 yes % | 评测 pred yes % | Macro-F1 | 相对 random |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1500 | pool-random | 785 | 52.33 | 654 | 52.33 | 50.22 | 92.24 | 0.00 |
| 1500 | ns-error-mass | 992 | 66.13 | 976 | 66.13 | 50.61 | 93.11 | +0.87 |
| 3000 | pool-random | 1568 | 52.27 | 1347 | 52.27 | 48.82 | 92.50 | 0.00 |
| 3000 | ns-error-mass | 2041 | 68.03 | 2005 | 68.03 | 50.27 | 93.51 | +1.01 |
| 4500 | pool-random | 2366 | 52.58 | 2018 | 52.58 | 49.79 | 93.03 | 0.00 |
| 4500 | ns-error-mass | 3073 | 68.29 | 3019 | 68.29 | 50.91 | 94.03 | +1.00 |
| 6000 | pool-random | 3145 | 52.42 | 2694 | 52.42 | 50.21 | 93.97 | 0.00 |
| 6000 | ns-error-mass | 4092 | 68.20 | 4019 | 68.20 | 50.21 | 94.47 | +0.50 |

### 5.3 低资源 50-250 pairwise 反例

低资源结果说明反偏置样本不是越多越好。12 个 paired comparison 中，胜者有更多反偏置样本的只有 8 个；其余情况显示过校正或几乎无效。

| 数据 | 问题 | 训练数 | 更好方法 | 更好方法反偏置数 | 另一方法反偏置数 | 胜者反偏置更多 | 胜者 pred yes % | 另一方法 pred yes % | 胜者 Macro-F1 | 另一方法 Macro-F1 | 差值 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| codebase | q1 | 50 | random | 21 | 19 | 是 | 16.91 | 15.81 | 69.08 | 69.07 | +0.00 |
| codebase | q1 | 125 | random | 96 | 94 | 是 | 11.99 | 3.92 | 75.81 | 72.11 | +3.70 |
| codebase | q1 | 250 | random | 221 | 219 | 是 | 9.31 | 4.58 | 80.83 | 77.33 | +3.50 |
| codebase | q2 | 50 | crc-error-mass | 13 | 14 | 否 | 68.70 | 75.91 | 68.91 | 66.85 | +2.06 |
| codebase | q2 | 125 | crc-error-mass | 41 | 39 | 是 | 72.31 | 87.08 | 71.73 | 61.52 | +10.21 |
| codebase | q2 | 250 | crc-error-mass | 91 | 87 | 是 | 58.69 | 69.80 | 78.04 | 76.42 | +1.62 |
| codebase | q3 | 50 | random | 42 | 38 | 是 | 0.01 | 0.01 | 42.58 | 42.58 | 0.00 |
| codebase | q3 | 125 | random | 94 | 98 | 否 | 0.21 | 0.05 | 42.69 | 42.56 | +0.12 |
| codebase | q3 | 250 | random | 188 | 192 | 否 | 5.81 | 2.58 | 56.06 | 50.65 | +5.41 |
| twitter_hate | q1 | 50 | random | 11 | 17 | 否 | 87.38 | 45.37 | 65.94 | 53.30 | +12.64 |
| twitter_hate | q1 | 125 | crc-error-mass | 35 | 27 | 是 | 93.33 | 97.63 | 65.85 | 55.70 | +10.15 |
| twitter_hate | q1 | 250 | crc-error-mass | 69 | 47 | 是 | 87.47 | 90.26 | 73.58 | 71.59 | +1.99 |

### 5.4 FEVER train=500 过校正

FEVER train=500 的公式比例实验是最清楚的过校正例子。defer-kcenter 选到更多 label=1 反偏置样本，但 eval pred yes 被推到 79.97%，最终比 random-defer 更差。

| 方法 | 反偏置数 | 反偏置 % | 训练 yes % | 评测 pred yes % | Acc | Positive-F1 |
| --- | --- | --- | --- | --- | --- | --- |
| defer_kcenter | 333 | 66.60 | 66.60 | 79.97 | 60.29 | 69.45 |
| random_defer | 260 | 52.00 | 52.00 | 71.83 | 71.33 | 76.47 |

### 5.5 balanced-guide 500/1000

balanced-guide 设置中，NS 与 CRC 的差异仍然混合。Codebase q1 正向，Codebase q2/q3 和 TwitterHate 为负或接近 0。

| 数据 | 问题 | 训练数 | 方法 | Macro-F1 | accept | defer | defer % |
| --- | --- | --- | --- | --- | --- | --- | --- |
| codebase | 1 | 500 | crc-error-mass | 80.12 | 7744 | 380 | 4.68 |
| codebase | 1 | 500 | ns-error-mass | 83.91 | 7948 | 176 | 2.17 |
| codebase | 2 | 500 | crc-error-mass | 86.18 | 6984 | 1141 | 14.04 |
| codebase | 2 | 500 | ns-error-mass | 83.71 | 6858 | 1267 | 15.59 |
| codebase | 3 | 500 | crc-error-mass | 76.94 | 6367 | 1766 | 21.71 |
| codebase | 3 | 500 | ns-error-mass | 76.84 | 5945 | 2188 | 26.90 |
| twitter_hate | 1 | 1000 | crc-error-mass | 85.87 | 22050 | 279 | 1.25 |
| twitter_hate | 1 | 1000 | ns-error-mass | 85.69 | 21714 | 615 | 2.75 |

### 5.6 IMDb 修复/破坏分解

IMDb 的分解说明，ns-error-mass 可以修复更多 base-wrong，但也会破坏更多 base-correct；最终结果并不优于 random。这个表只是补充诊断，不是主结论表。

| 问题 | 方法 | 训练数 | Acc | Macro-F1 | base-wrong 数 | LoRA 修复 | LoRA 破坏 | 净修复 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | pool-random | 2500 | 94.37 | 94.37 | 24996 | 23465 | 1283 | 22182 |
| 1 | ns-error-mass | 2500 | 92.27 | 92.25 | 24996 | 24165 | 3034 | 21131 |
| 2 | pool-random | 2500 | 88.69 | 85.72 | 36246 | 33575 | 2981 | 30594 |
| 2 | ns-error-mass | 2500 | 88.70 | 85.54 | 36246 | 33858 | 3260 | 30598 |

### 5.7 TwitterHate 四方法

TwitterHate 的最好结果只比 random 高 0.38 Macro-F1；在 round0 100% pred yes 的背景下，这只能算弱正向或近似 tied。

| 方法 | 训练数 | accept | defer | 训练 base-error % | 评测 pred yes 数 | 评测 pred no 数 | Acc | Macro-F1 | 相对 random |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| crc-error-mass | 2231 | 1106 | 1125 | 27.66 | 6248 | 1187 | 94.84 | 90.57 | +0.21 |
| ns-difficulty-crc-split | 2231 | 1106 | 1125 | 26.45 | 6256 | 1179 | 94.94 | 90.75 | +0.38 |
| ns-difficulty-global | 2231 | 1827 | 404 | 15.82 | 6195 | 1240 | 94.34 | 89.84 | -0.52 |
| random | 2231 | 1840 | 391 | 17.48 | 6144 | 1291 | 94.54 | 90.36 | 0.00 |

### 5.8 Codebase/Twitter 低资源均值

按四个 task/query 平均后，CRC 相对 random 在 train=50、125、250 的 Macro-F1 差值分别为 -2.65、+4.13、-1.33，方向不稳定。

| 训练数 | 方法 | 平均 Acc | 平均 Macro-F1 | 任务数 |
| --- | --- | --- | --- | --- |
| 50 | crc-error-mass | 72.77 | 58.46 | 4 |
| 50 | crc-random delta | -6.30 | -2.65 | 4 |
| 50 | random | 79.07 | 61.11 | 4 |
| 125 | crc-error-mass | 82.48 | 63.06 | 4 |
| 125 | crc-random delta | +2.01 | +4.13 | 4 |
| 125 | random | 80.48 | 58.93 | 4 |
| 250 | crc-error-mass | 84.31 | 69.90 | 4 |
| 250 | crc-random delta | +0.05 | -1.33 | 4 |
| 250 | random | 84.26 | 71.22 | 4 |


## 6. 数据文件

四个主表对应：

- `csv/step1_original_bias.csv`
- `csv/step2_training_composition.csv`
- `csv/step3_final_results.csv`
- `csv/step4_capacity_17b.csv`

补充小表和源汇总对应：

- `csv/round0_yesno.csv`
- `csv/query_bias_profile.csv`
- `csv/base_correct_wrong_training_results.csv`
- `csv/fever06b_counterexample_budget_1500_6000.csv`
- `csv/low_resource_counterexample_50_250.csv`
- `csv/low_resource_counterexample_pairwise.csv`
- `csv/fever06b_formula500_counterexample.csv`
- `csv/balanced_guide_common_results.csv`
- `csv/imdb_full_eval.csv`
- `csv/twitterhate_four_methods.csv`
- `csv/codebase_twitter_low_resource_summary.csv`
- `csv/fever_qwen17b_primary_results.csv`
