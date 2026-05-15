# CGSD 代码、实验、文档审计

本文件记录当前代码能完成什么、实验 README 如何运行，以及文档里需要注意的算法口径问题。

## 结论

当前 CLI 已经能按单步方式运行 CGSD 主链路：`prepare -> predict -> calibrate -> select -> train -> predict -> calibrate -> finalize`。每一步都有独立输入、输出、cache、usage 账本，也可以单独展示结果。

文档里有几处算法口径需要修正或明确。它们不会阻止工程实验运行，但会影响“严格 CRC 保证”和成本统计的写法。

## 文档算法问题

| 项 | 问题 | 当前处理 |
| --- | --- | --- |
| 温度选择 | `CGSD_complete_plan_final.md` 原来写在 `D_cal` 上扫描温度并选择 defer 最低的温度，这会让 routing score 依赖校准标签，严格 CRC 口径下有 adaptivity 风险。 | 实验 README 统一要求 `--temperature 15`；代码仍保留 round0 扫描能力，只作为诊断/非严格模式。 |
| 最终 CRC 保证 | 当前默认流程用同一个 `D_cal` 参与每轮 CRC、DBDS defer 集识别和停止判断；训练样本选择因此间接受 `D_cal` 标签影响。若最终仍用同一个 `D_cal` 声明 theorem-level CRC guarantee，独立性条件不严格成立。 | 实验 6 标明：默认结果是工程/经验验证；严格最终保证需要额外最终校准集，并通过 `cgsd_calibrate.py --calibration_predictions_path` 指向独立预测文件。 |
| CRC 风险指标 | 文档有时把 `accept_error_rate = wrong_accept / accepted_count` 写成 `<= alpha`。CRC 公式实际控制的是 `mean(1{accept and wrong})`，分母是校准集总数。 | README 明确 `crc.risk_bound` 是证明口径，`pool_summary.accept_error_rate` 只是诊断口径。 |
| Easy anchor 成本 | 实验方案修正要求 easy anchor 也用 teacher 标签，但部分 teacher call 公式只写 `n_cal + sum m_t + n_def`，漏掉 anchor。 | 实验 README 统一写明默认 anchor 额外消耗 `floor(0.1 * budget)` 个 teacher/groundtruth 标签。 |
| 成本公式 | 文档里的 `C(m) = (m + n_cal + rho*N) * c_T + N*c_S` 是部署口径，但实验运行还有每轮全量 student 推理和训练成本。 | 实验 7 要求分清部署成本和实验总成本；真实统计以 usage JSON 为准。 |
| 0.6B 模型名 | 算法文档和当前代码均统一到 Qwen3-0.6B，默认路径为 `model/qwen3-0.6b`。如果论文或报告要写精确成本、参数量，仍需按实际 checkpoint 复核。 | 实验变量默认使用当前代码路径。 |

## 代码和实验计划一致的部分

1. Round0 zero-shot、后续 round LoRA checkpoint 推理都由 `cgsd_predict.py` 单独启动。
2. 每轮对 `D_cal + U_pool` 全量重推理，符合实验方案“第一版保守全量重推理”。
3. `cgsd_prepare.py` 固定 calibration/pool split，`cgsd_select.py` 会排除 calibration 和已选样本。
4. DBDS 按 B/M/F band 分样，并在 band 内用 k-Center Greedy。
5. Easy anchor 只从非 defer、student 预测正确、高置信样本里选，并复用 teacher/groundtruth 标签进入训练。
6. `cgsd_train_round.py` 每轮从 base model 重新训练 LoRA，不做 continual training。
7. teacher 文件和 groundtruth 替代都能跑；usage 文件分别统计 `teacher_api_file_calls` 和 `groundtruth_substitute_calls`。
8. 每一步都有 `--cache_policy` 和 `--show_result`，可以单独启动、停止、产出、展示。

## 当前不能直接完成的实验项

1. 实验 2 的 Random、Uncertainty、k-Center、Defer-Random 没有独立 baseline selection CLI，需要手工生成 `cgsd_train_rows.jsonl`。
2. 实验 4 的 band 比例消融在算法函数里有参数，但 `cgsd_select.py` 没有暴露 CLI。
3. 实验 5 的 Full GPT-5、4B zero-shot cascade、4B 二次分流不由当前仓库生成，需要外部结果文件。
4. 实验 6 的 20 seed 聚合和违反率统计没有自动脚本。
5. 实验 7 的成本曲线、图表、均值方差聚合没有自动脚本。
6. 实验 8 的 cross-query LoRA 参数平均/迁移没有实现；当前只支持 per-query 独立训练。
7. embedding 生成 pipeline 和真实 teacher API 调用 pipeline 不在当前仓库里；当前代码只读取已经准备好的文件。

## 严格实验建议

1. 所有严格 CRC 实验都显式传 `--temperature 15`。
2. 如果需要 theorem-level final CRC guarantee，准备一个独立最终校准集，不参与 selection、温度选择、早停或模型选择。
3. 报告 `crc.risk_bound`、`wrong_accept_count / total` 和 `pool_summary.accept_error_rate` 时分开命名。
4. teacher 调用量以 usage JSON 为准，公式里要加上 easy anchor、真实 deployment defer 和每轮全量 student 推理。
5. 若使用真实 API teacher，先生成覆盖全量实验样本的 teacher 文件；否则缺失 ID 会按 groundtruth 替代统计。
