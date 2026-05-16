# LROBench CGSD 单次实验报告（seed=1）

## 整体结论

CGSD 在 LROBench seed=1、`T=15`、`alpha=0.07` 下完成三轮主链路。0.6B zero-shot 裸模型在 `U_pool` 上 raw acc 为 78.48%、raw macro F1 为 45.14%；经过 3 轮 DBDS + LoRA 后，raw acc 提升到 86.09%、raw macro F1 提升到 72.86%。LoRA 后 CRC accept 集也更准，accept macro F1 从 Round 0 的 45.70% 提升到 Round 3 的 79.78%。

正式 full-pool CRC 口径下，Round 3 wrong-accept risk 为 3.28%，低于 alpha；最终独立 `D_final` 校准下，defer 率为 33.20%，wrong-accept risk 为 6.30%，仍低于 alpha。最终部署中 253 条 CRC defer 里有 248 条可复用训练阶段标签，新增 teacher-equivalent defer 调用只有 5 次。

Defer-only 真实运行能省 student 调用，但 Round 3 wrong-accept risk 从 full-pool 的 5.25% 升到 10.24%，超过 alpha=0.07，因此不能作为正式 CRC 口径替代全量重预测，只能作为工程优化方向。

消融方面，m=500 单 seed 选择策略对比已完成。Uncertainty 最保守，defer 率最高但 wrong-accept risk 最低；Random 的 raw acc / raw F1 最高；DBDS 当前没有体现优势，主要因为 LROBench 这次 round0 defer 总数少且全部落在边界 band，band 分层选择退化。后续需要在更大数据量/OOD 数据集上，并调小 budget、调整 band 划分后继续验证。


## 实验设置

| 项目 | 设置 |
| --- | --- |
| 数据 | LROBench 合并输入，共 1162 条 |
| 切分 | calibration 200，final calibration 200，pool/deployment 762 |
| alpha | 0.07 |
| temperature | 15 |
| base model | Qwen3-0.6B |
| 标签来源 | groundtruth substitute，本次未调用真实 teacher API |
| embedding | pair embedding，2560 维 |

集合定义来自算法文档：全量输入先一次性划分为 `D_cal`、`D_final` 和 `U_pool`；本次 LROBench 共 1162 条，三者分别为 200、200、762，互不重叠且加起来等于全集。算法主评测分母是 `U_pool`，不是只取全集中的另一小部分。

## 模型与 DBDS 参数

| 参数 | 值 |
| --- | --- |
| 训练方式 | 每轮从 base model 重新 LoRA SFT |
| LoRA rank / alpha / dropout | 1 / 16 / 0.05 |
| LoRA target | `q_proj`, `v_proj` |
| epochs / lr | 3 / 2e-4 |
| batch / grad accum / max length | 1 / 16 / 512 |
| DBDS band 宽度 | delta = 0.1 |
| DBDS band 比例 | B/M/F = 0.6/0.3/0.1 |
| band 内选择 | k-Center greedy |
| easy anchor ratio | 0.1 |

训练样本：

| 选择轮次 | 训练时 defer 候选 | DBDS 请求 | 实际 DBDS | B/M/F 实际数 | anchor | 累计训练行 |
| ---: | ---: | ---: | ---: | --- | ---: | ---: |
| 0 | 503 | 250 | 250 | 151/75/24 | 25 | 275 |
| 1 | 346 | 150 | 150 | 92/46/12 | 15 | 440 |
| 2 | 352 | 100 | 80 | 55/5/20 | 10 | 530 |

说明：训练样本和 LoRA checkpoint 复用已完成的主 run；本次没有重新选样或重新训练，只用现有预测缓存把 CRC 主结果按 `T=15` 重算。第 2 轮 DBDS 请求 100 条，但只选到 80 条 defer 样本。

## 主结果

正式主结果采用每轮 full-pool 重预测：每轮对 calibration、final calibration 和 pool 全量预测，再在 pool 上报告 CRC 决策。

| Round | 模型 | lambda | defer | accept | raw acc | raw F1 | accept acc | accept F1 | defer=oracle F1 | wrong-accept risk |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| - | 0.6B ZS no CRC | - | - | - | 78.48% | 45.14% | - | - | - | - |
| 0 | 0.6B ZS + CRC | 0.52 | 497/762 | 265 | 78.48% | 45.14% | 84.15% | 45.70% | 90.91% | 5.51% |
| 1 | LoRA, 275 rows | 0.52 | 547/762 | 215 | 80.45% | 52.84% | 90.70% | 47.56% | 95.91% | 2.62% |
| 2 | LoRA, 440 rows | 0.53 | 510/762 | 252 | 82.02% | 66.24% | 91.67% | 47.83% | 95.70% | 2.76% |
| 3 | LoRA, 530 rows | 0.53 | 430/762 | 332 | 86.09% | 72.86% | 92.47% | 79.78% | 94.85% | 3.28% |

关键读法：

1. 纯 0.6B no-CRC baseline 是 raw acc 78.48%、raw macro F1 45.14%。
2. Round 3 裸模型质量最好，raw acc 86.09%、raw macro F1 72.86%。
3. 在 `T=15` full-pool 口径下，Round 3 wrong-accept risk 为 3.28%，低于 alpha=0.07。
4. accept F1 从 Round 0 的 45.70% 升到 Round 3 的 79.78%，说明 LoRA 后 CRC 放行集合确实更可靠。

## 最终部署

最终部署使用 Round 3 模型和独立 final calibration，因此 lambda 与 Round 3 中间校准不同。

| 项目 | 值 |
| --- | ---: |
| final lambda | 0.52 |
| CRC accept | 509 |
| CRC defer | 253 |
| CRC defer 率 | 33.20% |
| wrong-accept risk | 6.30% |
| accept 错误率 | 9.43% |
| raw accuracy | 86.09% |
| raw macro F1 | 72.86% |
| accept macro F1 | 75.92% |
| defer=oracle cascade accuracy | 93.70% |
| defer=oracle cascade macro F1 | 89.50% |

最终部署样本来源：

| 来源 | 数量 |
| --- | ---: |
| 训练标签复用 | 530 |
| student accept | 227 |
| 新增 teacher defer | 5 |
| 合计 | 762 |

`CRC defer=253` 不等于新增 teacher 调用 253 次，因为大部分 defer 样本已经在训练阶段有标签。

调用量统计：

| 项目 | 次数 | 说明 |
| --- | ---: | --- |
| student model calls | 4648 | Round 0-3 每轮预测 `D_cal + D_final + U_pool = 1162` |
| `D_cal` teacher-equivalent labels | 200 | CRC 校准标签，一次性使用；usage 中每轮校准会重复读取，不按新增调用重复计费 |
| `D_final` teacher-equivalent labels | 200 | 最终独立校准 |
| 训练 teacher-equivalent labels | 530 | 3 轮 DBDS defer 样本 + easy anchors，累计去重训练行 |
| 部署阶段 CRC defer | 253 | 系统输出中由 teacher/groundtruth 负责的 defer 样本 |
| 部署新增 teacher-equivalent calls | 5 | 其余 248 条 defer 已在训练标签中可复用 |
| 算法累计新增 teacher-equivalent calls | 935 | `200 + 200 + 530 + 5` |

本次没有调用真实 teacher API；上表里的 teacher-equivalent calls 均由 groundtruth substitute 代替，但按算法需要的 teacher 调用位置计数。

### D_final = D_cal 对照

这个对照只改变最终 CRC 校准集：不用独立 final calibration，而是复用 Round 3 的 `D_cal`。测试集仍然是 `U_pool`，不是改到 `D_cal` 上评估。

| final calibration | lambda | U_pool defer | U_pool accept | U_pool wrong-accept risk | U_pool accept 错误率 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 独立 `D_final` | 0.52 | 253/762（33.20%） | 509 | 6.30% | 9.43% |
| 复用 `D_cal` | 0.53 | 430/762（56.43%） | 332 | 3.28% | 7.53% |

## Defer-only 真实运行对照

defer-only real 已真实加载 Round 1-3 LoRA checkpoint 跑过，不是虚构反事实。它每轮只预测 calibration + 上一轮 defer pool，上一轮已 accept 的 pool 样本冻结。

| 口径 | full-pool | defer-only real | 节省 |
| --- | ---: | ---: | ---: |
| 总 student model calls（Round 0-3） | 4648 | 2765 | 1883（40.51%） |
| Round 1-3 pool calls | 2286 | 1003 | 1283（56.12%） |

风险结论保持不变：defer-only real 省调用，但 Round 3 wrong-accept risk 从 full-pool 的 5.25% 升到 10.24%，超过 alpha=0.07。原因是早期错误 accept 被冻结，后续模型无法纠正；所以正式 CRC 结果仍应采用 full-pool 重预测。

## 消融结果

消融设置：LROBench seed=1，固定 T=15、alpha=0.07、m=500、LoRA rank=1、target=`q_proj/v_proj`。复用同一份 round0 split 和 round0 预测缓存；每个配置都重新生成训练行、真实训练 LoRA、重新 full-pool 预测和 CRC 校准。为了提高吞吐，后续消融训练使用等效 batch 16 的 `batch_size=8, gradient_accumulation_steps=2`；早先完成的部分配置使用 `batch_size=1, gradient_accumulation_steps=16`，等效 batch 相同。

### 数据选择策略对比

| 策略 | 训练行 | lambda | defer | raw acc | raw F1 | accept F1 | defer=oracle F1 | wrong-accept risk |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| DBDS | 547 | 0.55 | 494/762（64.83%） | 78.74% | 44.65% | 48.56% | 96.97% | 1.97% |
| Random | 500 | 0.54 | 543/762（71.26%） | 83.33% | 69.91% | 58.09% | 96.76% | 2.10% |
| Uncertainty | 500 | 0.53 | 632/762（82.94%） | 83.33% | 64.79% | 87.18% | 98.81% | 0.79% |
| k-Center | 500 | 0.54 | 526/762（69.03%） | 81.63% | 65.15% | 74.86% | 97.80% | 1.44% |
| Defer-Random | 497 | 0.53 | 619/762（81.23%） | 79.66% | 48.95% | 57.45% | 98.21% | 1.18% |

读法：这一组是 `Tmax=1` 的 m=500 单 seed 选择策略对比，不是多 seed 正式显著性结论。DBDS 请求 500 条 defer 样本，但 T=15 round0 只有 497 条 defer；加 50 条 easy anchor 后训练行是 547。Defer-Random 只能从 497 条 defer 中抽样，所以训练行是 497。当前单 seed 下，DBDS 没有优于 Random / k-Center；Uncertainty 通过更高 defer 率换来了更低 wrong-accept risk。

结果表已写到 `experiments/02_selection_comparison/lrobench/table2_strategy_comparison_m500_seed1.csv`。

### Band 比例消融

| 配置 | 状态 | 训练行 | lambda | defer | raw acc | raw F1 | accept F1 | defer=oracle F1 | wrong-accept risk |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| A1：1/0/0，全边界 | 已完成 | 547 | 0.54 | 597/762（78.35%） | 82.28% | 59.95% | 54.60% | 97.39% | 1.71% |
| A2：0.6/0.3/0.1，默认 | 已完成 | 547 | 0.53 | 357/762（46.85%） | 78.74% | 44.65% | 47.40% | 91.39% | 5.25% |
| A3：0.33/0.34/0.33，均匀 | 运行中 | - | - | - | - | - | - | - | - |
| A4：0/0/1，全 deep | 待完成 | - | - | - | - | - | - | - | - |
| A5：0/1/0，全 middle | 待完成 | - | - | - | - | - | - | - | - |

在当前 T=15 round0 缓存下，DBDS selection 记录显示 497 条 defer 样本全部落在 B band，M/F 为 0。因此 band 比例消融在这个 split 上可能不敏感；A1 和 A2 的实际 distillation band counts 都是 B=497、M=0、F=0。当前 partial 表已写到 `experiments/04_ablation/lrobench/table4a_band_ratio_m500_seed1_partial.csv`。

## 还缺哪些消融

| 模块 | 计划要求 | 当前状态 |
| --- | --- | --- |
| 数据选择对比 | fixed m=500、Tmax=1，对比 Random、Uncertainty、k-Center、Defer-Random、DBDS；5 seeds；paired t-test | 单 seed 已完成；还缺多 seed 和 paired t-test |
| 预算曲线 | DBDS 扫描 m={0,50,100,200,300,500,700,1000,2000}，并叠加 Random/Uncertainty 曲线 | 未完成；当前预算点不完整，也缺 baseline 曲线 |
| Band 比例 | A1-A5：全边界、默认 0.6/0.3/0.1、均匀、全深度 defer、全中间 | 进行中；A1/A2 已完成，A3 正在跑，A4-A5 待完成 |
| Band 宽度 | delta={0.05,0.1,0.15,0.2} | 未完成 |
| Band 内选择 | DBDS 要求每个 band 内用 k-Center greedy；还需与不分 band 的 k-Center、Defer-Random 等策略对比 | 单 seed 已有 DBDS、k-Center、Defer-Random 对比；还缺多 seed |
| 迭代轮次 | Tmax={1,2,3,5} | 未完成；当前只有三轮主链路 |
| LoRA rank | rank={1,2,4,8} | 未完成；当前只有 rank=1 |
| Teacher weighting | beta={0,0.5,1,2} | 未完成 |
| Easy anchor | anchor ratio={0,0.05,0.1,0.2} | 未完成；当前只有 0.1 |
| CRC 保证验证 | 20 次随机划分，验证 wrong-accept risk 均值 | 未完成 |

本报告中的集合分工：

| 口径 | 分母 |
| --- | --- |
| raw acc / macro F1 | pool/deployment 762 条 |
| wrong-accept risk | pool/deployment 762 条 |
| accept 错误率 | CRC accept 子集 |
| lambda 校准 | calibration 200 条，最终部署用 final calibration 200 条 |
| Step2 `n_s` / neighbor support | 用 calibration 200 条建 support bank，再给 U_pool 样本查邻居 |
| 每轮 full-pool student calls | calibration 200 + final calibration 200 + pool 762 = 1162 |

## 文档对应关系

| 报告项 | 本报告对应内容 | 方案/总览中的位置 |
| --- | --- | --- |
| 数据与切分 | 1162 条输入，200/200/762 切分 | 完整方案 §7；实验总览“输入检查” |
| 固定 T=15 | 本报告所有 CRC 主结果用 T=15 缓存重算 | 实验计划“修正 1”；完整方案 §8.3.1 |
| CRC 指标 | lambda、defer、wrong-accept risk、accept 错误率 | 完整方案 §4.1、§8.5 |
| DBDS | defer 集分 B/M/F，band 内 k-Center，easy anchor | 完整方案 §8.4.1 |
| LoRA 设置 | rank=1，target=`q_proj/v_proj`，每轮重训 | 完整方案 §8.4.3 |
| 全量重预测 | 每轮 full-pool 预测作为正式 CRC 口径 | 完整方案 §8.4.4；实验总览“主链路” |
| 最终部署 | final calibration、训练标签复用、新增 teacher defer | 完整方案 §8.5 |
| 未完成消融 | DBDS 对比、band、rank、beta、anchor、CRC 多 seed | 实验计划“实验 2/3/4/6” |
