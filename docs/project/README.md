# MIAS/DCMS Project Tree

This directory contains the planning and acceptance documents for the MIAS/DCMS path. The source documents do not prescribe a complete repository tree, so the current tree is organized around their stage outputs and leakage-isolation requirements.

## Project Documents

- `01_最终任务_算法与AAAI论文路径_MIAS_DCMS.md`: final research positioning, algorithm scope, and AAAI paper path.
- `02_开发与实验执行文档_MIAS_DCMS.md`: execution phases D0-D8 and required artifacts.
- `03_开发与实验验收清单_MIAS_DCMS.md`: gate-by-gate acceptance checklist.
- `04_当前迁移开发验收状态_MIAS_DCMS.md`: current code migration and development acceptance status.
- `05_下一阶段实验规划_MIAS_DCMS.md`: evidence-based ordering, run matrix, stop/go rules, and completion criteria for the next experiments.
- `06_Qwen06B_T4_串行执行队列.md`: task-isolated binary, multiclass, and DPO execution order for the Qwen3-0.6B T4 study.

## Current File Tree Contract

- `mias_dcms/`: active reusable package code only.
- `scripts/`: active runnable entrypoints only.
- `tests/`: tests for the active code tree.
- `docs/project/`: MIAS/DCMS planning and acceptance documents.
- `docs/paper/`: active paper draft materials.
- `docs/archive/`: old paper drafts and superseded narratives.
- `experiments/inputs/`: durable input surfaces such as source data, split ids, embeddings, and reusable caches.
- `experiments/runs/`: local or compact run outputs.
- `experiments/reports/`: compact reports and provenance.
- `experiments/reports/binary_legacy/`: archived binary-task tables and figures.
- `resources/`: supporting non-code resources.

## Boundary Rules

- Active code may depend only on `mias_dcms/` and `scripts/`; superseded package trees are not part of the active import surface.
- Legacy binary evidence stays under `experiments/reports/binary_legacy/`.
- Hidden-label, oracle-label, selector-score, and evaluator-score artifacts should remain separated when new experiment outputs are added.
- Gate evidence should point to real compact artifacts; checkpoint evidence is registered through `scripts/register_initial_policy_checkpoint.py` only after required files exist.
- Generated caches, checkpoints, model weights, and large local run outputs remain ignored unless they are compact reproducibility artifacts.
