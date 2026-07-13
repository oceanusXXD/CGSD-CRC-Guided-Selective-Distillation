from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import PurePosixPath
from typing import Any


DPO_EXECUTION_STAGES = ("selection", "reveal", "training", "evaluation", "summary")

REQUIRED_DPO_EXECUTION_ARTIFACTS = (
    "active_pool_path",
    "oracle_store_path",
    "logprobs_path",
    "selection_summary_path",
    "selected_ids_path",
    "revealed_rows_path",
    "dpo_train_rows_path",
    "training_summary_path",
    "evaluation_metrics_path",
    "cost_report_path",
)


@dataclass(frozen=True)
class DPOExecutionManifestValidationReport:
    run_count: int
    issue_count: int
    issues: list[dict[str, Any]] = field(default_factory=list)

    @property
    def is_ready(self) -> bool:
        return not self.issues

    def as_dict(self) -> dict[str, Any]:
        return {
            "is_ready": self.is_ready,
            "run_count": self.run_count,
            "issue_count": self.issue_count,
            "issues": [dict(issue) for issue in self.issues],
        }


def build_dpo_execution_manifest(rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    run_rows = [dict(row) for row in rows]
    manifest_runs = [_build_run_plan(row) for row in run_rows]
    manifest = {
        "stage_order": list(DPO_EXECUTION_STAGES),
        "run_count": len(manifest_runs),
        "runs": manifest_runs,
    }
    report = validate_dpo_execution_manifest(manifest)
    return {**manifest, **report.as_dict()}


def validate_dpo_execution_manifest(manifest: Mapping[str, Any]) -> DPOExecutionManifestValidationReport:
    issues: list[dict[str, Any]] = []
    runs = manifest.get("runs", [])
    if not isinstance(runs, Sequence) or isinstance(runs, (str, bytes)):
        return DPOExecutionManifestValidationReport(
            run_count=0,
            issue_count=1,
            issues=[{"code": "invalid_runs"}],
        )

    run_ids: list[str] = []
    for run_index, run in enumerate(runs):
        if not isinstance(run, Mapping):
            issues.append({"code": "invalid_run", "run_index": run_index})
            continue
        run_id = str(run.get("run_id", ""))
        run_ids.append(run_id)
        _validate_required_artifacts(run, run_index=run_index, run_id=run_id, issues=issues)
        if str(run.get("run_status")) == "failed":
            if not str(run.get("failure_reason", "")).strip():
                issues.append({"code": "failed_run_missing_reason", "run_index": run_index, "run_id": run_id})
            if run.get("stages") not in ([], None):
                issues.append({"code": "failed_run_has_actionable_stages", "run_index": run_index, "run_id": run_id})
            continue
        stages = run.get("stages", [])
        if not isinstance(stages, Sequence) or isinstance(stages, (str, bytes)):
            issues.append({"code": "invalid_stages", "run_index": run_index, "run_id": run_id})
            continue
        observed_stage_order = [
            str(stage.get("stage"))
            for stage in stages
            if isinstance(stage, Mapping) and stage.get("stage") is not None
        ]
        if observed_stage_order != list(DPO_EXECUTION_STAGES):
            issues.append(
                {
                    "code": "stage_order_mismatch",
                    "run_index": run_index,
                    "run_id": run_id,
                    "observed": observed_stage_order,
                    "expected": list(DPO_EXECUTION_STAGES),
                }
            )

    for run_id in sorted({value for value in run_ids if run_ids.count(value) > 1}):
        issues.append({"code": "duplicate_run_id", "run_id": run_id})

    return DPOExecutionManifestValidationReport(
        run_count=len(runs),
        issue_count=len(issues),
        issues=issues,
    )


def _build_run_plan(row: Mapping[str, Any]) -> dict[str, Any]:
    artifacts = dict(row.get("artifacts") or {})
    data_config = dict(row.get("data_config") or {})
    run_id = str(row.get("run_id"))
    run_dir = _run_dir_from_artifacts(artifacts)
    run_status = str(row.get("run_status", "planned"))
    run_plan = {
        "run_id": run_id,
        "dataset": str(row.get("dataset")),
        "model": str(row.get("model")),
        "budget": int(row.get("budget", 0)),
        "seed": int(row.get("seed", 0)),
        "method": str(row.get("method")),
        "run_status": run_status,
        "failure_reason": str(row.get("failure_reason", "")),
        "config_hash": str(row.get("config_hash", "")),
        "training_config_hash": str(row.get("training_config_hash", "")),
        "judge_config_hash": str(row.get("judge_config_hash", "")),
        "artifacts": {**artifacts, "run_record_path": str(run_dir / "run_record.json")},
        "data_config": data_config,
    }
    if run_status == "failed":
        run_plan["stages"] = []
        return run_plan
    run_plan["stages"] = _stages_for_run(run_plan["artifacts"], data_config, row)
    return run_plan


def _stages_for_run(
    artifacts: Mapping[str, Any],
    data_config: Mapping[str, Any],
    row: Mapping[str, Any],
) -> list[dict[str, Any]]:
    method = str(row.get("method"))
    budget = int(row.get("budget", 0))
    seed = int(row.get("seed", 0))
    run_dir = _run_dir_from_artifacts(artifacts)
    return [
        {
            "stage": "selection",
            "depends_on": [],
            "status": "blocked",
            "blocker": "awaiting_execution",
            "inputs": {
                "active_pool_path": str(data_config.get("active_pool_path") or artifacts.get("active_pool_path")),
                "logprobs_path": str(data_config.get("logprobs_path") or artifacts.get("logprobs_path")),
            },
            "outputs": {
                "selected_ids_path": str(artifacts.get("selected_ids_path")),
                "selection_summary_path": str(artifacts.get("selection_summary_path")),
            },
            "commands": _selection_commands(
                method=method,
                budget=budget,
                seed=seed,
                run_dir=run_dir,
                artifacts=artifacts,
                data_config=data_config,
            ),
        },
        {
            "stage": "reveal",
            "depends_on": ["selection"],
            "status": "blocked",
            "blocker": "awaiting_selection",
            "inputs": {
                "active_pool_path": str(data_config.get("active_pool_path") or artifacts.get("active_pool_path")),
                "oracle_store_path": str(data_config.get("oracle_store_path") or artifacts.get("oracle_store_path")),
                "selected_ids_path": str(artifacts.get("selected_ids_path")),
            },
            "outputs": {
                "revealed_rows_path": str(artifacts.get("revealed_rows_path")),
                "dpo_train_rows_path": str(artifacts.get("dpo_train_rows_path")),
            },
            "commands": [
                " ".join(
                    [
                        "python",
                        "scripts/reveal_preference_labels.py",
                        "--active_pool_path",
                        _quote(str(data_config.get("active_pool_path") or artifacts.get("active_pool_path"))),
                        "--oracle_store_path",
                        _quote(str(data_config.get("oracle_store_path") or artifacts.get("oracle_store_path"))),
                        "--selected_ids_path",
                        _quote(str(artifacts.get("selected_ids_path"))),
                        "--output_dir",
                        _quote(str(run_dir)),
                        "--round_index",
                        "0",
                        "--method",
                        _quote(_normalize_method_for_command(method)),
                    ]
                )
            ],
        },
        {
            "stage": "training",
            "depends_on": ["reveal"],
            "status": "blocked",
            "blocker": "awaiting_reveal",
            "inputs": {
                "dpo_train_rows_path": str(artifacts.get("dpo_train_rows_path")),
            },
            "outputs": {
                "training_summary_path": str(artifacts.get("training_summary_path")),
                "cost_report_path": str(artifacts.get("cost_report_path")),
            },
            "commands": _training_commands(
                method=method,
                budget=budget,
                seed=seed,
                run_dir=run_dir,
                artifacts=artifacts,
                row=row,
            ),
        },
        {
            "stage": "evaluation",
            "depends_on": ["training"],
            "status": "blocked",
            "blocker": "awaiting_training",
            "inputs": _evaluation_inputs(artifacts=artifacts, row=row),
            "outputs": {
                "evaluation_metrics_path": str(artifacts.get("evaluation_metrics_path")),
            },
            "commands": _evaluation_commands(
                artifacts=artifacts,
                row=row,
            ),
        },
        {
            "stage": "summary",
            "depends_on": ["evaluation"],
            "status": "blocked",
            "blocker": "awaiting_evaluation",
            "inputs": {
                "selection_summary_path": str(artifacts.get("selection_summary_path")),
                "reveal_summary_path": str(run_dir / "summary.json"),
                "revealed_rows_path": str(artifacts.get("revealed_rows_path")),
                "dpo_train_rows_path": str(artifacts.get("dpo_train_rows_path")),
                "training_summary_path": str(artifacts.get("training_summary_path")),
                "evaluation_metrics_path": str(artifacts.get("evaluation_metrics_path")),
                "cost_report_path": str(artifacts.get("cost_report_path")),
            },
            "outputs": {
                "run_record_path": str(artifacts.get("run_record_path")),
            },
            "commands": _summary_commands(
                method=method,
                budget=budget,
                seed=seed,
                artifacts=artifacts,
                row=row,
            ),
        },
    ]


def _selection_commands(
    *,
    method: str,
    budget: int,
    seed: int,
    run_dir: PurePosixPath,
    artifacts: Mapping[str, Any],
    data_config: Mapping[str, Any],
) -> list[str]:
    active_pool_path = str(data_config.get("active_pool_path") or artifacts.get("active_pool_path"))
    logprobs_path = str(data_config.get("logprobs_path") or artifacts.get("logprobs_path"))
    normalized = _normalize_method_for_command(method)
    if normalized == "random":
        return [
            " ".join(
                [
                    "python",
                    "scripts/select_preference_random.py",
                    "--input_path",
                    _quote(active_pool_path),
                    "--output_dir",
                    _quote(str(run_dir)),
                    "--budget",
                    str(int(budget)),
                    "--seed",
                    str(int(seed)),
                ]
            )
        ]

    if normalized in {"reward_margin", "apl", "active_dpo"}:
        scored_path = str(run_dir / "baseline_scores.jsonl")
        score_command = _score_preference_command(
            input_path=logprobs_path,
            output_path=scored_path,
            method=normalized,
            data_config=data_config,
        )
        return [
            score_command,
            " ".join(
                [
                    "python",
                    "scripts/select_preference_baseline.py",
                    "--input_path",
                    _quote(scored_path),
                    "--output_dir",
                    _quote(str(run_dir)),
                    "--method",
                    _quote(normalized),
                    "--budget",
                    str(int(budget)),
                ]
            ),
        ]

    if normalized in {"apl_dcms", "active_dpo_dcms"}:
        base_method = "apl" if normalized == "apl_dcms" else "active_dpo"
        scored_path = str(run_dir / "baseline_scores.jsonl")
        dcms_candidates_path = str(run_dir / "dcms_candidates.jsonl")
        score_command = _score_preference_command(
            input_path=logprobs_path,
            output_path=scored_path,
            method=base_method,
            data_config=data_config,
        )
        return [
            score_command,
            " ".join(
                [
                    "python",
                    "scripts/prepare_preference_dcms_inputs.py",
                    "--input_path",
                    _quote(scored_path),
                    "--output_path",
                    _quote(dcms_candidates_path),
                    "--method",
                    _quote(base_method),
                    "--group_fields",
                    "ab_position,source_pair",
                ]
            ),
            " ".join(
                [
                    "python",
                    "scripts/select_dcms.py",
                    "--input_path",
                    _quote(dcms_candidates_path),
                    "--output_dir",
                    _quote(str(run_dir)),
                    "--budget",
                    str(int(budget)),
                    "--target_moments",
                    "pool",
                    "--slack_grid",
                    "0,0.01,0.02,0.05,0.1,0.2,0.5",
                    "--kappa",
                    "0.05",
                    "--rounding_seed",
                    str(int(seed)),
                    "--use_rank_normalization",
                ]
            ),
        ]
    return [f"# unsupported selection method: {method}"]


def _score_preference_command(
    *,
    input_path: str,
    output_path: str,
    method: str,
    data_config: Mapping[str, Any],
) -> str:
    parts = [
        "python",
        "scripts/score_preference_baselines.py",
        "--input_path",
        _quote(input_path),
        "--output_path",
        _quote(output_path),
        "--methods",
        method,
    ]
    prompt_clusters_path = data_config.get("prompt_clusters_path")
    if prompt_clusters_path:
        parts.extend(["--metadata_path", _quote(str(prompt_clusters_path))])
    if method == "active_dpo":
        if _as_bool(data_config.get("active_dpo_length_normalize")):
            parts.append("--active_dpo_length_normalize")
        if "active_dpo_novelty_weight" in data_config:
            parts.extend(["--active_dpo_novelty_weight", str(float(data_config["active_dpo_novelty_weight"]))])
    return " ".join(parts)


def _training_commands(
    *,
    method: str,
    budget: int,
    seed: int,
    run_dir: PurePosixPath,
    artifacts: Mapping[str, Any],
    row: Mapping[str, Any],
) -> list[str]:
    training_config = dict(row.get("training_config") or {})
    model_name = str(training_config.get("model_name_or_path") or row.get("model") or "")
    return [
        " ".join(
            [
                "python",
                "scripts/train_preference_dpo_run.py",
                "--dpo_train_rows_path",
                _quote(str(artifacts.get("dpo_train_rows_path"))),
                "--output_dir",
                _quote(str(run_dir / "policy_adapter")),
                "--training_summary_path",
                _quote(str(artifacts.get("training_summary_path"))),
                "--cost_report_path",
                _quote(str(artifacts.get("cost_report_path"))),
                "--model_name_or_path",
                _quote(model_name),
                "--training_config_json",
                _quote(_json_arg(training_config)),
                "--seed",
                str(int(seed)),
                "--method",
                _quote(_normalize_method_for_command(method)),
                "--budget",
                str(int(budget)),
            ]
        )
    ]


def _evaluation_inputs(*, artifacts: Mapping[str, Any], row: Mapping[str, Any]) -> dict[str, str]:
    evaluation_config = dict(row.get("evaluation_config") or {})
    inputs = {"training_summary_path": str(artifacts.get("training_summary_path"))}
    for field in ("preference_predictions_path", "judge_rows_path", "capability_rows_path"):
        path = _evaluation_input_path(evaluation_config, field=field, row=row, artifacts=artifacts)
        if path:
            inputs[field] = path
    return inputs


def _evaluation_commands(*, artifacts: Mapping[str, Any], row: Mapping[str, Any]) -> list[str]:
    evaluation_config = dict(row.get("evaluation_config") or {})
    preference_predictions_path = _evaluation_input_path(
        evaluation_config, field="preference_predictions_path", row=row, artifacts=artifacts
    )
    judge_rows_path = _evaluation_input_path(
        evaluation_config, field="judge_rows_path", row=row, artifacts=artifacts
    )
    capability_rows_path = _evaluation_input_path(
        evaluation_config, field="capability_rows_path", row=row, artifacts=artifacts
    )
    if not any((preference_predictions_path, judge_rows_path, capability_rows_path)):
        return []
    command = [
        "python",
        "scripts/audit_preference_evaluation.py",
        "--output_path",
        _quote(str(artifacts.get("evaluation_metrics_path"))),
    ]
    if preference_predictions_path:
        command.extend(["--preference_predictions_path", _quote(preference_predictions_path)])
    if judge_rows_path:
        command.extend(["--judge_rows_path", _quote(judge_rows_path)])
    if capability_rows_path:
        command.extend(["--capability_rows_path", _quote(capability_rows_path)])
    return [" ".join(command)]


def _evaluation_input_path(
    evaluation_config: Mapping[str, Any],
    *,
    field: str,
    row: Mapping[str, Any],
    artifacts: Mapping[str, Any],
) -> str:
    direct = str(evaluation_config.get(field, ""))
    if direct:
        return _format_stage_template(direct, row=row, artifacts=artifacts)
    template = str(evaluation_config.get(f"{field}_template", ""))
    if template:
        return _format_stage_template(template, row=row, artifacts=artifacts)
    return ""


def _format_stage_template(template: str, *, row: Mapping[str, Any], artifacts: Mapping[str, Any]) -> str:
    run_dir = _run_dir_from_artifacts(artifacts)
    context = {
        "run_dir": str(run_dir),
        "dataset": str(row.get("dataset", "")),
        "model": str(row.get("model", "")),
        "budget": str(row.get("budget", "")),
        "seed": str(row.get("seed", "")),
        "method": str(row.get("method", "")),
        "run_id": str(row.get("run_id", "")),
    }
    return str(template).format(**context)


def _summary_commands(
    *,
    method: str,
    budget: int,
    seed: int,
    artifacts: Mapping[str, Any],
    row: Mapping[str, Any],
) -> list[str]:
    return [
        " ".join(
            [
                "python",
                "scripts/build_dpo_run_record.py",
                "--selection_summary_path",
                _quote(str(artifacts.get("selection_summary_path"))),
                "--reveal_summary_path",
                _quote(str(_run_dir_from_artifacts(artifacts) / "summary.json")),
                "--training_rows_path",
                _quote(str(artifacts.get("dpo_train_rows_path"))),
                "--training_summary_path",
                _quote(str(artifacts.get("training_summary_path"))),
                "--evaluation_metrics_path",
                _quote(str(artifacts.get("evaluation_metrics_path"))),
                "--cost_report_path",
                _quote(str(artifacts.get("cost_report_path"))),
                "--output_path",
                _quote(str(artifacts.get("run_record_path"))),
                "--dataset",
                _quote(str(row.get("dataset"))),
                "--model",
                _quote(str(row.get("model"))),
                "--method",
                _quote(str(method)),
                "--budget",
                str(int(budget)),
                "--seed",
                str(int(seed)),
                "--config_hash",
                _quote(str(row.get("config_hash", ""))),
            ]
        )
    ]


def _normalize_method_for_command(method: str) -> str:
    key = str(method).strip().lower().replace("+", "_").replace("-", "_").replace(" ", "_")
    aliases = {
        "random": "random",
        "reward_margin": "reward_margin",
        "apl": "apl",
        "active_dpo": "active_dpo",
        "activedpo": "active_dpo",
        "apl_dcms": "apl_dcms",
        "active_dpo_dcms": "active_dpo_dcms",
        "activedpo_dcms": "active_dpo_dcms",
    }
    return aliases.get(key, key)


def _quote(value: str) -> str:
    text = str(value)
    if not text:
        return "''"
    if all(ch.isalnum() or ch in "/._:-=," for ch in text):
        return text
    return "'" + text.replace("'", "'\"'\"'") + "'"


def _json_arg(payload: Mapping[str, Any]) -> str:
    return _json_dumps_compact(payload)


def _json_dumps_compact(payload: Mapping[str, Any]) -> str:
    import json

    return json.dumps(dict(payload), ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def _validate_required_artifacts(
    run: Mapping[str, Any],
    *,
    run_index: int,
    run_id: str,
    issues: list[dict[str, Any]],
) -> None:
    artifacts = run.get("artifacts")
    if not isinstance(artifacts, Mapping):
        issues.append({"code": "missing_artifacts", "run_index": run_index, "run_id": run_id})
        return
    for artifact_name in REQUIRED_DPO_EXECUTION_ARTIFACTS:
        if not str(artifacts.get(artifact_name, "")).strip():
            issues.append(
                {
                    "code": "missing_required_artifact",
                    "run_index": run_index,
                    "run_id": run_id,
                    "artifact": artifact_name,
                }
            )


def _run_dir_from_artifacts(artifacts: Mapping[str, Any]) -> PurePosixPath:
    for value in artifacts.values():
        text = str(value)
        if text:
            return PurePosixPath(text).parent
    return PurePosixPath(".")
