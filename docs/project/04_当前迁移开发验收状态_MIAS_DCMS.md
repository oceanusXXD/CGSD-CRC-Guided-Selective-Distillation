# MIAS/DCMS 当前迁移开发验收状态

> 日期：2026-07-13
> 范围：新主线文件树迁移、可复用代码迁移、旧树清理、D0-D7 基础开发能力。  
> 说明：本文只记录当前代码与配置层面的完成状态；真实数据集、模型训练、DPO 训练和论文主结果仍需后续实验执行。

---

## 1. 文件树迁移状态

### 已完成

- Active package 已统一为 `mias_dcms/`。
- Runnable entrypoints 已统一保留在 `scripts/`。
- 测试统一保留在 `tests/`。
- MIAS/DCMS 三份项目文档已归档到 `docs/project/`。
- D0 冻结配置已落到 `configs/mias_dcms_freeze.v1.json`。
- 旧版源码包、匿名镜像包和历史结果根目录已从当前工作树移除。
- 已移除已废弃的二分类历史结果与选择器实现。

### 当前主线顶层目录

- `configs/`
- `docs/`
- `experiments/`
- `mias_dcms/`
- `resources/`
- `scripts/`
- `tests/`

---

## 2. 已完成的代码能力

### D0：协议冻结

- `configs/mias_dcms_freeze.v1.json`
  - 固定主任务：Active Preference Acquisition。
  - 固定受控验证任务：Multi-class Active Distillation。
  - 固定历史先导证据：4 个数据源上的 7 个 binary predicates。
  - 固定 preference / multiclass baselines、selection metrics、cost metrics、预算口径、标签隔离字段、DCMS slack grid 与 kappa。

### 统一记录与审计

- `mias_dcms.records`
  - `AcquisitionRecord`
  - `RunRecord`
  - `build_records_from_dcms`
  - `build_run_record`
- `mias_dcms.result_aggregation`
  - run-level 记录完整性校验
  - paper-level metric table 自动聚合
  - cost metrics 完整性检查
- `scripts/aggregate_paper_metrics.py`
  - 从 run-level JSONL 自动生成 paper-level metric table JSON。
- `mias_dcms.budgeting`
  - 监督预算与 evaluation resource 分离统计
  - judge calls、training tokens、selector compute 统一成本报告
  - 方法间监督预算与 train token 公平性比较
- `scripts/audit_budget_report.py`
  - 从 budget JSONL 自动生成公平预算与成本审计 JSON。
- `mias_dcms.auditing`
  - propensity identity
  - acquisition TV
  - maximum propensity ratio
  - selected distribution prediction error
  - `MIASSelectionAudit`
- `scripts/audit_mias_selection.py`
  - 从 sample-level JSONL 生成 selection audit JSON。

### DCMS

- `mias_dcms.soft_groups`
  - 从 cross-fit / ensemble membership draws 生成 soft group mean membership。
  - 输出 robust lower / upper membership interval。
  - 输出 membership calibration 与 interval coverage 报告。
  - 通过 selector label-safety guard 防止 active pool true labels / oracle labels 泄漏。
- `mias_dcms.soft_group_error`
  - 对比 nominal soft membership 约束与 robust interval 约束。
  - 使用 observed membership 计算真实 coverage violation。
  - 输出 robust 是否降低 observed constraint violation 的审计结果。
- `scripts/prepare_soft_group_intervals.py`
  - 从 soft group JSONL 生成 `soft_group_membership.jsonl`、`summary.json` 与可选 `calibration_summary.json`。
- `scripts/audit_soft_group_error.py`
  - 从 soft group candidate JSONL 生成 nominal vs. robust coverage error audit JSON。
- `mias_dcms.selection.dcms`
  - exact small-batch DCMS solver
  - rank normalization
  - robust lower / upper membership intervals
  - slack grid selection
  - utility-retention threshold via `kappa`
  - utility-coverage frontier audit
  - deterministic rounding seed plumbing
  - selected ids、propensity、continuous / rounded moments、constraint violation、solver status 输出
- `scripts/select_dcms.py`
  - 从 candidate JSONL 读取 score 和 group membership。
  - 输出 `selected_ids.json`、`propensity.jsonl`、`selection_summary.json`。
- `scripts/audit_dcms_frontier.py`
  - 在固定 slack grid 上输出 utility retained 与 coverage deviation 的 Pareto/frontier 审计 JSON。

### MIAS 干预工具

- `mias_dcms.interventions`
  - class-logit intercept
  - entropy score recomputation
  - DPO length coefficient intervention
  - normalized length gap
  - fixed-budget response curve audit

### Preference fixed-pool / 标签隔离

- `mias_dcms.preference_pool`
  - selector-safe active pool
  - separate oracle store
  - A/B swap manifest
  - length gap、source pair、A/B position fields
- `scripts/prepare_preference_pool.py`
  - 从 raw preference JSONL 生成 `active_pool.jsonl`、`oracle_store.json`、`swap_manifest.json`、`pool_summary.json`。
- `mias_dcms.preference_split_manifest`
  - 为 preference fixed-pool 生成 seed / active_pool / heldout / test 的稳定无交叉 split ids。
- `scripts/prepare_preference_splits.py`
  - 从 selector-safe active pool 写出 `split_manifest.json` 和 `split_summary.json`。
- `mias_dcms.prompt_clusters`
  - 从预计算 prompt embedding 生成冻结 prompt cluster assignment。
  - 输出 hard cluster、soft cluster membership、cluster probabilities、centroids 和聚类摘要。
- `scripts/prepare_prompt_clusters.py`
  - 为 APL prompt entropy、DCMS prompt cluster moments 和 acquisition audits 生成 selector-safe cluster metadata。
- `mias_dcms.preference_logprob_audit`
  - 审计 active pool 的 policy / reference log-probs。
  - 生成 policy gap、reference gap、implicit reward gap 与 absolute implicit margin。
  - 拒绝缺失或非有限 log-probs，并默认拒绝 implicit margin 全为零的输入。
- `mias_dcms.preference_logprob_generation`
  - 对 selector-safe preference active pool 计算 policy / reference 对两个候选 response 的 causal-LM sequence log-probs。
  - 支持 base model、policy adapter、reference adapter、ChatML prompt formatting、prompt-left truncation 和 token-count/truncation metadata。
- `scripts/generate_preference_logprobs.py`
  - 从 active pool 与 policy/reference model checkpoint 生成 `logprobs.jsonl`，并立即复用 logprob audit 校验 Gate 4 输入。
- `mias_dcms.checkpoint_registry`
  - 注册真实共享初始 DPO policy checkpoint 作为 Gate 4 evidence。
  - 校验 LoRA adapter checkpoint 必需文件，写出 manifest 与文件 SHA256，不伪造缺失 checkpoint。
- `scripts/register_initial_policy_checkpoint.py`
  - 从真实 adapter 目录生成 `dpo.initial_policy_checkpoint` evidence manifest。
  - 仅在 checkpoint 必需文件存在且非空时更新 evidence JSON。
- `scripts/audit_preference_logprobs.py`
  - 从 active-pool logprob JSONL 生成 audited JSONL 与 Gate 4 summary。
- `mias_dcms.preference_selector_audit`
  - 审计 selector score 是否非退化。
  - 复核 top-budget selected ids、selected ids 是否重复、oracle calls 是否等于预算。
  - 报告 score-length correlation、A/B swap score delta 和 selector compute seconds。
- `scripts/audit_preference_selector_scores.py`
  - 从 baseline score JSONL 生成 Gate 5 selector sanity summary。
- `mias_dcms.preference_acquisition_audit`
  - 对 preference selection membership 一次性审计 length / source / prompt cluster 等多属性覆盖。
  - 输出每个属性的 acquisition TV、JS divergence、maximum propensity ratio 和 group propensity 明细。
  - 支持 Random reference membership 作为同预算参考分布。
- `scripts/audit_preference_acquisition.py`
  - 从 preference membership JSONL 和可选 Random reference JSONL 生成 Gate 5 acquisition audit summary。
- `mias_dcms.preference_intervention_audit`
  - 审计 DPO length-gamma intervention 的 propensity response curve。
  - 审计 selector replacement 的 score rank correlation、selected-set overlap 和 attribute coverage delta。
  - 审计 A/B position 的 paired rank correlation、selected ids 和 position propensity。
- `mias_dcms.preference_intervention_inputs`
  - 将 selector-safe active pool、真实 logprobs 和 baseline scores 合并为 Gate 6 干预审计输入。
  - 生成 `base_margin`、固定边界 `length_gap_bin`、source / position / prompt cluster 字段，不读取 oracle labels。
- `scripts/prepare_preference_intervention_inputs.py`
  - CPU-only 生成 length-gamma、selector replacement、A/B position 审计所需 JSONL。
- `scripts/audit_preference_intervention.py`
  - 从 intervention JSONL 生成 Gate 6 length-gamma / selector replacement / A/B position audit summary。
- `mias_dcms.preference_evaluation`
  - 计算 held-out preference accuracy。
  - 计算 worst-group preference accuracy。
  - 计算 raw judge win rate 与 length-controlled win rate。
  - 计算 capability regression，供 DPO 主结果表和 run-level aggregation 使用。
- `scripts/audit_preference_evaluation.py`
  - 从 held-out preference predictions、judge rows 和 capability rows 生成结构化 evaluation metrics JSON。
- `mias_dcms.preference_scoring`
  - Reward Margin selector score
  - APL selector score
  - ActiveDPO fixed-pool adaptation score：policy-reference gap proxy、可选 pair-token length normalization、selector-safe prompt novelty component
  - hidden preference labels 输入保护
- `scripts/score_preference_baselines.py`
  - 从 scored preference pool 生成 selector-safe baseline score JSONL 与 summary。
  - 支持通过 `--metadata_path` 合并 frozen prompt cluster metadata，并输出 ActiveDPO gradient / length-normalized / novelty components。
- `scripts/select_preference_baseline.py`
  - 从 Reward Margin / APL / ActiveDPO baseline score 生成 selected ids、membership 和 selection summary。
- `mias_dcms.preference_dcms_inputs`
  - 将 selector-safe preference baseline scores 转为 DCMS candidate rows。
  - 支持 categorical observable groups 与 soft group membership。
- `scripts/prepare_preference_dcms_inputs.py`
  - 生成可被 `scripts/select_dcms.py` 直接消费的 `score` / `groups` JSONL。
- `mias_dcms.preference_reveal`
  - 选中后才揭示 oracle preference label。
  - 生成 DPO train rows，并保持未选 active-pool 样本不暴露 oracle label。
- `scripts/reveal_preference_labels.py`
  - 将 selected ids、active pool 和 oracle store 合并为 `revealed_rows.jsonl` / `dpo_train_rows.jsonl` / `summary.json`。
- `mias_dcms.preference_run_summary`
  - 汇总 preference selection、reveal、DPO train rows、cost metrics 为 run-level record。
- `scripts/build_preference_run_summary.py`
  - 生成可进入 paper-level aggregation 的 preference acquisition `RunRecord` JSON。
- `mias_dcms.experiment_run_matrix`
  - 生成 DPO 主实验的 planned run matrix，覆盖 dataset / model / budget / seed / method 的笛卡尔组合。
  - 为每个 planned run 固定 stable `run_id`、预期 artifact path、training config hash、judge config hash 和总 `config_hash`。
  - 验证缺失 planned run、重复 run id、artifact path 缺失，以及同一 dataset / model / budget / seed 下跨方法 training / judge config drift。
- `scripts/build_experiment_run_matrix.py`
  - 从 `configs/dpo_run_matrix.example.json` 风格配置写出 `run_matrix.jsonl` 与 readiness summary。
  - 作为真实 DPO 主实验前的 preflight，不会训练模型或生成实验结果。
- `mias_dcms.preference_experiment_preflight`
  - 审计 preference / DPO 主实验训练前输入产物是否一致。
  - 检查 active pool 是否包含 hidden label 泄漏字段，oracle store 与 logprobs 是否覆盖 active pool ids。
  - 检查 split manifest 的 active ids 是否匹配，seed / active / heldout / test split 是否交叉。
  - 检查 run matrix 是否覆盖 expected methods / seeds，并核对 data_config 中 active pool、oracle store、logprobs 路径。
- `scripts/audit_preference_experiment_preflight.py`
  - 从 active pool、oracle store、logprobs、split manifest 和 run matrix 写出训练前 readiness report。
  - 在 hidden-label 泄漏、缺失 oracle/logprob、split overlap 或 run-matrix 输入路径不一致时返回非零。
- `mias_dcms.dpo_execution_manifest`
  - 将 planned run matrix 转换为 DPO 主实验执行 manifest。
  - 为每个 planned run 固定 selection、reveal、training、evaluation、summary 的阶段顺序、输入产物、输出产物和依赖关系。
  - 为 selection、reveal、training、evaluation、summary stage 生成可调度 commands；evaluation 若缺真实评估输入则保持无命令 blocked 状态。
  - failed run 保留在 manifest 中，但不生成 actionable stages，防止静默删除失败运行。
  - 验证 duplicate run id、stage order drift 和 required artifact 缺失。
- `scripts/build_dpo_execution_manifest.py`
  - 从 `run_matrix.jsonl` 写出 `execution_manifest.json`。
  - 在执行 manifest 缺失关键产物或结构不一致时返回非零。
- `scripts/train_preference_dpo_run.py`
  - 从 revealed `dpo_train_rows.jsonl` 训练单个 preference DPO run，输出 `training_summary.json`、policy adapter 和 `cost_report.json`。
- `scripts/build_dpo_run_record.py`
  - 从 selection / reveal / training / evaluation / cost 产物生成单 run `run_record.json`，避免 summary stage 使用空指标。
- `mias_dcms.dpo_execution_status`
  - 根据 execution manifest 和已存在产物审计每个 DPO run 的执行状态。
  - 将 run 标记为 blocked / in_progress / complete / failed，并记录 next stage、present inputs、missing outputs。
  - failed run 保留 failure reason；缺失 failure reason 会进入 issues，避免失败运行被静默吞掉。
- `scripts/audit_dpo_execution_status.py`
  - 从 `execution_manifest.json` 写出 execution status report。
  - 全部 planned run 产物齐全时返回 0；仍有 blocked / in_progress / failed run 时返回非零。
- `scripts/audit_experiment_gate_readiness.py`
  - 支持 `--require_existing_paths` 真实路径校验。
  - stdout 输出紧凑 readiness 摘要，完整 gate 明细保留在 JSON report 中。
- `mias_dcms.dpo_run_record_collection`
  - 从 execution manifest 和实际 DPO 执行产物收集 run-level records。
  - 保留 completed / failed / incomplete run 状态，缺失产物或 Gate 8-10 必需指标时进入 readiness issues。
  - 输出结构对齐 `validate_dpo_run_pack` 所需的 selection / training / evaluation / cost metrics。
- `scripts/collect_dpo_run_records.py`
  - 从 `execution_manifest.json` 写出 `run_records.jsonl` 和 collection readiness report。
  - 全部 run records 完整且没有 failed / incomplete run 时返回 0；缺失产物或指标时返回非零。
- `mias_dcms.dpo_run_pack`
  - 审计 DPO 主结果 run pack 是否覆盖 Random / Reward Margin / APL / ActiveDPO / APL+DCMS / ActiveDPO+DCMS。
  - 检查 dataset / model / budget / seed / method 组合是否缺失或重复。
  - 要求 failed run 显式保留并记录 failure reason。
  - 检查 selection / training / evaluation / cost metrics 是否满足 Gate 8-10 汇总前要求。
  - 审计论文 figure / table artifact manifest 的输入文件、聚合规则、seed 数、error bar 与 failed-run policy。
- `scripts/validate_dpo_run_pack.py`
  - 从 run-level JSONL 和可选 paper artifact manifest 生成 DPO run-pack readiness report。
- `mias_dcms.experiment_gate_readiness`
  - 将 Gate 0-10 的真实实验验收项整理为可执行 readiness audit。
  - 从声明的 evidence manifest 判断每个 Gate 的 ready / blocked 状态和缺失证据。
  - 防止把单元测试通过误报为真实 fixed-pool、DPO 训练、主实验或论文产物已完成。
- `scripts/audit_experiment_gate_readiness.py`
  - 从 evidence JSON 写出 Gate 0-10 readiness report。
  - 任一 Gate 缺少真实实验证据时返回非零，作为后续实验执行前后的总门禁。
- `mias_dcms.run_metric_comparison`
  - 对 run-level records 做 baseline vs treatment paired seed comparison。
  - 输出 evaluation / selection / training / cost metrics 的 delta mean、bootstrap CI 和 paired permutation p-value。
  - 显式报告 expected seed 缺失、paired seed 不足和缺失指标，避免静默删除失败或缺失 run。
- `scripts/compare_run_metrics.py`
  - 从 run-level JSONL 生成 Gate 9 统计比较 report，供主表和论文 claim freeze 使用。
- `mias_dcms.paper_claim_audit`
  - 审计 claim-to-evidence mapping 是否满足指定 claim type 的证据类型要求。
  - 检查支持证据的 seed 数和 failed-run policy。
  - 扫描论文文本或 claim text 中禁止使用的过度主张，例如“首次发现 active learning sampling bias”或“DCMS 无条件提高下游性能”。
- `scripts/audit_paper_claims.py`
  - 从 claims JSON、evidence JSON、requirements JSON 和可选论文正文生成 Gate 10 claim-freeze audit report。
- `mias_dcms.result_freeze_pack`
  - 审计 D8 / Gate 10 result freeze pack 是否包含 `results_manifest`、主表、附表、主图数据和 claim-evidence map。
  - 检查每个冻结产物是否记录输入结果文件、聚合规则、seed 数、error bar 和 failed-run policy。
  - 检查 frozen protocol 是否覆盖预注册 metrics / baselines、judge version 和只允许 bug-fixes / supplement 的 freeze policy。
- `scripts/validate_result_freeze_pack.py`
  - 从 freeze-pack JSON 生成 result-freeze readiness report，作为论文结果冻结前的自动门禁。
- `mias_dcms.intervention_statistics`
  - 审计干预响应曲线是否覆盖至少 5 个 intervention strength。
  - 输出 Spearman monotonicity、slope、bootstrap slope CI，并保留每个 setting 的完整响应范围。
  - 显式报告 expected setting 缺失、failed setting 和 failed setting 缺少原因，避免只展示成功 setting。
- `scripts/audit_intervention_statistics.py`
  - 从 intervention curve JSONL 生成 Gate 9 / D8 intervention statistics report。
- `mias_dcms.paper_artifacts`
  - 从 run-level records、intervention statistics、matched-utility summary 和 claim audit 生成冻结版 Fig. 1-3 / Table 1-3 JSON 数据包。
  - 生成 `results_manifest`、`claim_evidence_map`、主表、附表、主图数据和 frozen protocol，可直接进入 `mias_dcms.result_freeze_pack` 验证。
  - 在 claim audit 未通过或 expected baselines 缺失时拒绝生成论文冻结产物。
- `scripts/build_paper_artifacts.py`
  - 将审计后的实验结果写出为 `freeze_pack.json`、`results_manifest.json`、`claim_evidence_map.json`、`main_tables/*.json` 和 `figure_data/*.json`。

### Multiclass protocol

- `mias_dcms.multiclass_protocol`
  - fixed seed / active / test split
  - pool class prior
  - split disjointness validation
- `scripts/prepare_multiclass_splits.py`
  - 从 multiclass JSONL 生成 `split_ids.json`、`pool_prior.json`、`summary.json`。

### Baseline selector tools

- `mias_dcms.selectors`
  - seeded random without replacement
  - moment-matched Random selection
  - top-budget selection
  - entropy uncertainty
  - margin uncertainty
  - selector input label-safety check
- `mias_dcms.sampling_diagnostics`
  - Random / Entropy / Margin / BADGE / GALAXY classification selectors
  - `Entropy+DCMS` / `BADGE+DCMS` wrappers using soft class-posterior moments
  - continuous propensity, rounded moments, selected slack and utility-retention artifacts
  - accepts `cross_fitted_class_posterior`; raw scored probabilities are explicitly marked as a smoke-only proxy
- `scripts/select_moment_matched_random.py`
  - 从 candidate JSONL 读取 group membership。
  - 输出 moment-matched Random 的 `selected_ids.json`、`membership.jsonl`、`selection_summary.json`。
- `scripts/benchmark_pipeline.py diagnose-classification`
  - 支持 `random,entropy,margin,badge,galaxy,entropy+dcms,badge+dcms`。
  - DCMS 选择前只读取 selector-safe rows；真实 class label 只用于 post-selection shift report。

### 统计与 composition 审计

- `mias_dcms.statistics`
  - bootstrap mean CI
  - paired mean delta
  - paired permutation test
  - method-level metric summary
- `mias_dcms.composition`
  - utility quantile profile
  - matched-utility report
  - coverage deviation against target moments
- `scripts/audit_matched_utility.py`
  - 对 baseline / treatment 两组 selection 做 utility matching 与 group coverage 对比。

---

## 3. 当前测试覆盖

当前测试覆盖：

- codebase cleanliness / legacy import guard
- D0 freeze protocol
- acquisition records and run records
- fair budget / cost reporting
- run-level result aggregation / paper table generation
- MIAS audit metrics
- DCMS solver, soft group intervals, soft group error audit, slack, robust intervals, utility-coverage frontier, rounding seed
- intervention utilities
- preference fixed pool
- preference logprob audit / selector score sanity audit / first-round acquisition audit / DPO intervention audit / DPO evaluation metric audit / baseline scoring / top-budget selection / DCMS input preparation / post-selection oracle reveal / run-level summary / DPO run-pack validation
- paired run-metric statistical comparison
- paper claim-to-evidence audit / banned overclaim detection
- result freeze-pack validation
- intervention response statistics / hidden failed-setting guard
- frozen paper artifact generation
- multiclass split / prior protocol
- selector utilities
- CLI scripts:
  - `aggregate_paper_metrics.py`
  - `compare_run_metrics.py`
  - `audit_paper_claims.py`
  - `validate_result_freeze_pack.py`
  - `build_paper_artifacts.py`
  - `audit_budget_report.py`
  - `audit_dcms_frontier.py`
  - `audit_mias_selection.py`
  - `audit_intervention_response.py`
  - `audit_intervention_statistics.py`
  - `audit_matched_utility.py`
  - `audit_soft_group_error.py`
  - `audit_preference_logprobs.py`
  - `audit_preference_acquisition.py`
  - `audit_preference_selector_scores.py`
  - `audit_preference_intervention.py`
  - `audit_preference_evaluation.py`
  - `prepare_preference_pool.py`
  - `score_preference_baselines.py`
  - `select_preference_baseline.py`
  - `prepare_preference_dcms_inputs.py`
  - `prepare_preference_splits.py`
  - `reveal_preference_labels.py`
  - `build_preference_run_summary.py`
  - `build_experiment_run_matrix.py`
  - `audit_preference_experiment_preflight.py`
  - `build_dpo_execution_manifest.py`
  - `train_preference_dpo_run.py`
  - `build_dpo_run_record.py`
  - `audit_dpo_execution_status.py`
  - `run_dpo_manifest_stage.py`
  - `collect_dpo_run_records.py`
  - `validate_dpo_run_pack.py`
  - `audit_experiment_gate_readiness.py`
  - `prepare_multiclass_splits.py`
  - `prepare_soft_group_intervals.py`
  - `select_dcms.py`
  - `select_moment_matched_random.py`
- statistics / composition utilities

最近一次验证：

```text
full regression suite: 222 passed, 236 subtests passed in 109.94s
focused selector/multiclass/data/gate verification: 17 passed in 7.12s
compileall: passed; pip check: passed; git diff --check: passed
stale refs: none in active mainline scan
old dirs: absent src, absent code, absent result
```

当前 Gate readiness evidence：

- `configs/experiment_gate_evidence.current.json`
  - `protocol.freeze`: `configs/mias_dcms_freeze.v1.json`
  - `preference.active_pool`: `experiments/inputs/preference/helpsteer2_preference/active_pool.jsonl`
  - `preference.oracle_store`: `experiments/inputs/preference/helpsteer2_preference/oracle_store.json`
  - `preference.split_manifest`: `experiments/inputs/preference/helpsteer2_preference/split_manifest.json`
- `experiments/reports/gate_readiness.current.json`
  - 当前只证明 Gate 0 ready。
- `experiments/reports/gate_readiness.strict.current.json`
  - `ready_gates: [gate_0_protocol_freeze]`；其余 10 个 Gate blocked，缺失证据数 37。
  - Gate 4 已有真实 HelpSteer2 active pool / oracle store / split manifest；仍缺真实 policy/reference logprobs 和初始 DPO policy checkpoint。

当前 DPO mainline planning artifacts：

- `configs/dpo_run_matrix.current.json`
  - 绑定当前 HelpSteer2-Preference fixed-pool、oracle store、split manifest 和预期 logprobs 路径。
  - 覆盖 `Random`、`Reward Margin`、`APL`、`ActiveDPO`、`APL+DCMS`、`ActiveDPO+DCMS` × 5 seeds × budget 100，共 30 个 planned runs。
- `experiments/runs/dpo_main/current/run_matrix.jsonl`
- `experiments/runs/dpo_main/current/execution_manifest.json`
- `experiments/reports/dpo_main/current/run_matrix_summary.json`
  - `is_ready: true`
  - `expected_run_count: 30`
  - `planned_run_count: 30`
- `experiments/reports/dpo_main/current/random_selection_stage_report.json`
  - 当前主线未执行 selection stage；fixed pool 已存在，但所有方法仍缺主线 logprobs 输入。
- `experiments/reports/dpo_main/current/random_reveal_stage_report.json`
  - 当前仅保留历史 stage report；其路径不构成当前主线完成证据。
  - stage runner 默认只保存 stdout/stderr preview 与 parsed JSON summary，不再把完整 child stdout 塞进报告。
  - manifest 已具备 training、evaluation、summary stage 命令；当前 `execution_status.json` 显示 30 个 planned runs 均在 selection 阶段因缺失输入而 blocked。
  - evaluation stage 通过 `evaluation_config` 模板生成 `heldout_preference_predictions.jsonl`、`judge_rows.jsonl`、`capability_rows.jsonl` 和 `aulc_rows.jsonl` 输入路径；这些主线评估输入尚未生成。
- `experiments/reports/dpo_main/current/execution_status.json`
  - `is_complete: false`
  - `run_count: 30`
  - `in_progress_run_count: 0`
  - `blocked_run_count: 30`
  - `next_stage_counts: {selection: 30}`
  - 30 个主线 run 均因缺失主线 logprobs 输入停在 selection stage；fixed pool 本身已存在。
  - 当前 manifest command counts：selection 65、reveal 30、training 30、evaluation 30、summary 30。
  - `scripts/audit_dpo_execution_status.py` 默认 stdout 为 compact summary；完整 run 明细仍写入 `execution_status.json`。

---

## 4. 尚未完成的真实实验 Gate

以下项目需要真实数据、模型 checkpoint、训练或外部 judge 才能验收，当前不能只靠单元测试声明通过：

### Gate 1：原二分类重审计

- 恢复全部 setting 的 sample-level logits / selected ids / teacher labels。
- 修正 guide / seed / active label budget。
- 重新计算 4 数据源 / 7 predicates 的机制统计和下游指标。

### Gate 2-3：多分类 MIAS 因果验证

- 准备 AG News / TREC 固定池。
- 训练初始模型并保存完整 logits。
- `scripts/benchmark_pipeline.py` 已实现 Random / Entropy / Margin / BADGE / GALAXY，以及 `Entropy+DCMS` / `BADGE+DCMS`；当前只有 24-row AG News 和 36-row TREC smoke，不能声称 Gate 2-3 完成。
- 跑完整 AG News / TREC、多 seed、多模型的采集统计。
- 跑 class-intercept alpha response curves。
- 验证 propensity identity 和多 seed / 多模型稳定性。

### Gate 4-6：Preference / DPO 主任务

- 主线 HelpSteer2-Preference fixed pool 已生成：17,354 paired rows，seed / active / heldout / test 为 1,000 / 10,000 / 2,000 / 4,000；TL;DR fixed pool 仍需准备。
- 当前 DPO run matrix / execution manifest 已生成 30 个 planned runs。
- 训练初始 DPO policy。
- 保存真实 policy / reference log-probs，并用 `scripts/audit_preference_logprobs.py` 做完整性和 implicit margin 审计。
- Reward Margin / APL / ActiveDPO fixed-pool CPU selector score 已实现；ActiveDPO 当前明确为固定池适配，不宣称完全复现原论文 online acquisition 设置。
- 跑真实 length-gamma、selector replacement、A/B swap intervention；当前已有对应结构化审计工具，仍需真实 selector outputs 和多 seed / 多 setting 结果。

### Gate 7：DCMS 算法正确性完整实验

- 当前已覆盖 A1-A5 的基础单元行为，并具备 soft group interval construction、calibration / coverage reporting、soft group error audit、utility-coverage frontier、matched-utility composition 审计与基础统计工具。
- 仍需真实 cross-fitted soft group estimator 训练和真实数据 soft group error 实验。

### Gate 8-10：下游训练、统计、论文图表

- 多分类训练曲线。
- DPO 主表；当前已有 held-out preference accuracy、worst-group、length-controlled win rate 和 capability regression 的结构化指标生成工具，仍需真实评测数据。
- matched-utility composition intervention。
- 统计 CI / permutation test。
- 论文 Fig. 1-3、Table 1-3 和 AAAI 主张审查。

---

## 5. 当前结论

当前仓库已经完成新主线迁移和基础开发层：

- 文件树已切换到 `mias_dcms/` 主线。
- 旧目录和旧路径引用已清理。
- 可复用旧代码已迁移为 active package 和 scripts。
- MIAS/DCMS 的基础算法、记录、审计、干预、固定池、preference logprob audit、selector score sanity audit、first-round acquisition audit、DPO intervention audit、DPO evaluation metric audit、preference baseline scoring / selection / DCMS input preparation / post-selection oracle reveal / run-level summary、DPO run-matrix preflight、preference experiment artifact preflight、DPO execution manifest、DPO execution status audit、current DPO planning artifacts、split、selector、soft group intervals/calibration/error audit、统计、composition、预算成本报告、run-level 聚合、DCMS frontier 和 CLI 工具已建立并通过测试。

尚不能声明“完整 AAAI 实验方案完成”，因为真实数据、模型训练、DPO 主实验、统计检验和论文图表仍未执行。

## 6. 2026-07-13 修复与 CPU smoke 验证

本轮修复了 ActiveDPO logprob token-count 字段兼容、AULC evaluation 输入、DPO
update-step / initial-adapter 配置传递、selector hidden-field guard、paired A/B
swap id、prompt split leakage guard，以及大池 DCMS continuous relaxation + rounding
路径。

可复现的合成 CPU smoke 结果位于：
`experiments/reports/smoke_mias_dcms.current.json`。

该 smoke 使用 60 个合成 preference pairs，验证了 10 个样本的 selection/reveal、
ActiveDPO+DCMS、非二值 propensity、rounded group moments、A/B paired audit、
preference metrics 和 AULC；它不是 Gate 4-10 的真实数据证据。

## 7. 2026-07-13 真实模型合成池 smoke

`scripts/run_qwen_preference_smoke.py` 在本地 Qwen3-0.6B 上完成了一条小规模端到端链：

- 48 行 selector-safe active pool，8 / 24 / 8 / 8 的 seed / active / held-out / test split；
- 8 个 seed rows 训练共享初始 LoRA，并完成 checkpoint registration；
- 真实 policy/reference log-prob、Random / Reward Margin / APL / ActiveDPO / 两个 DCMS selector；
- 训练一次 `ActiveDPO+DCMS`，held-out preference accuracy `0.75`、worst-group `0.50`、
  length-controlled win rate `0.6667`、AULC `0.75`。

完整报告：`experiments/reports/real_smoke_qwen06b.json`。
该实验使用合成 pool 和固定合成 oracle labels，capability regression 是 preferred-response
log-prob proxy；`paper_evidence: false`，不能替代 HelpSteer2 / TL;DR 主实验或 Gate 4-10。

## 8. 2026-07-13 AG News selector/DCMS smoke

使用真实 AG News 数据和本地 Qwen3-0.6B 的 24-row scored pool，运行了三个预算
（4 / 8 / 12）的 `Random`、`Entropy`、`BADGE`、`GALAXY`、`Entropy+DCMS` 和
`BADGE+DCMS`。完整产物位于：
`experiments/runs/benchmark_shift/ag_news_qwen06b_smoke_dcms/`。

- 每行保存了模型概率和 frozen representation；zero-shot pool accuracy 为 `0.8333`。
- 两个 DCMS 方法都输出 `selected_ids`、连续 `q_propensity`、rounded moments、slack trace
  和 utility-retained；预算 4 的 propensity 总和为 `4.0`。
- TREC 另完成 36-row smoke（6 类各 6 条），Qwen3-0.6B zero-shot accuracy 为 `0.5556`，
  产物位于 `experiments/runs/benchmark_shift/trec_qwen06b_smoke_dcms/`；同样覆盖三个预算和六种方法。
- 这些 smoke 的 DCMS membership 使用 scored probability proxy；没有 cross-fitted estimator、
  多 seed、多模型或完整 AG News / TREC 训练，因此 `paper_evidence: false`，Gate 2-3 仍 blocked。
