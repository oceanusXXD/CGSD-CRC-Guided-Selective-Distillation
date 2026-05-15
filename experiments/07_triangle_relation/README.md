# 实验 7：三角关系验证

当前代码不单独提供三角关系脚本；这个实验由实验 3 的预算曲线和实验 6 的 CRC 保证结果汇总得到。

## 数据前置

1. 先完成实验 3 的每个预算点。
2. 对关键预算点补跑实验 6 的多 seed 验证。
3. 准备 teacher 和 student 的单位成本常量，用于计算总成本。
4. 统一决定成本口径：部署成本只算最终模型推理和最终 defer，实验总成本还要算每轮 predict、teacher 标注、训练和 embedding 准备。

## Baseline 要求和复用

本实验不重新训练模型，直接消费实验 3、实验 6 和可选实验 5 的结果缓存。

1. DBDS 预算曲线来自实验 3：每个 `m` 读取对应输出目录里的 `round_summary.json`、`cgsd_summary.json`、`deployment_decisions.jsonl` 和 usage JSON。
2. CRC 违反率来自实验 6：关键预算点读取多 seed、多 alpha 的 `round_summary.json` 汇总。
3. 端到端外部 baseline 如果要放进同一张成本图，读取实验 5 已整理的 Full GPT-5、4B cascade 和二次分流结果表，不在本实验重新估。
4. 所有曲线必须使用同一套成本常量；如果某个 baseline 只有聚合成本没有 token/call 明细，要单独标记为 external aggregate。

## 运行方式

1. 对每个预算 `m` 跑实验 3，得到 `rho(m)`。
2. 对每个预算 `m` 记录 `accept_error_rate(m)`。
3. 从 usage 文件读取 teacher 调用、student 调用、embedding 使用量和 token 估算。
4. 用外部聚合脚本计算：

```text
C(m) = (m + n_cal + rho(m) * N) * c_T + N * c_S
```

如果没有关闭 easy anchor，把公式里的 `m` 改成 `m + floor(0.1*m)`；如果统计实验总成本，还要把每轮 `predict_usage.json` 的 student 调用全部加进去。

## 需要输出的表

| m | defer_rate | accept_error_rate | final_accuracy | teacher_calls | estimated_cost |
| --- | --- | --- | --- | --- | --- |

`accept_error_rate` 在表里只表示诊断用的 accept 子集条件错误率；CRC 风险列建议额外加 `wrong_accept_rate = wrong_accept_count / total`。

成本表建议额外保留这些列，方便后续统一统计：

| student_calls | teacher_api_file_calls | groundtruth_substitute_calls | estimated_student_prompt_tokens | estimated_student_completion_tokens | estimated_teacher_prompt_tokens | estimated_teacher_completion_tokens | embedding_rows |
| --- | --- | --- | --- | --- | --- | --- | --- |

字段来源：student 调用和 token 来自 `predict_usage.json`；CRC 校准标签来自 `calibrate_usage.json`；DBDS 选择、teacher/groundtruth 训练标签和 embedding 行数来自 `select_usage.json`；训练 token 来自 `train_usage.json`；最终 defer teacher 调用和 token 来自 `finalize_usage.json`。

## 当前代码限制

当前仓库能产生表格所需的原始 JSON；画图、成本常量管理、均值方差聚合还需要额外脚本。
