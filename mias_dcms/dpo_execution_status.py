from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class DPOExecutionStatusReport:
    run_count: int
    completed_run_count: int
    in_progress_run_count: int
    blocked_run_count: int
    failed_run_count: int
    runs: list[dict[str, Any]]
    issues: list[dict[str, Any]] = field(default_factory=list)

    @property
    def is_complete(self) -> bool:
        return self.run_count > 0 and self.completed_run_count == self.run_count and not self.issues

    def as_dict(self) -> dict[str, Any]:
        return {
            "is_complete": self.is_complete,
            "run_count": self.run_count,
            "completed_run_count": self.completed_run_count,
            "in_progress_run_count": self.in_progress_run_count,
            "blocked_run_count": self.blocked_run_count,
            "failed_run_count": self.failed_run_count,
            "runs": [dict(run) for run in self.runs],
            "issues": [dict(issue) for issue in self.issues],
        }


def audit_dpo_execution_status(
    manifest: Mapping[str, Any],
    *,
    existing_paths: Iterable[str],
) -> DPOExecutionStatusReport:
    existing = {str(path) for path in existing_paths}
    source_runs = manifest.get("runs", [])
    if not isinstance(source_runs, Sequence) or isinstance(source_runs, (str, bytes)):
        return DPOExecutionStatusReport(
            run_count=0,
            completed_run_count=0,
            in_progress_run_count=0,
            blocked_run_count=0,
            failed_run_count=0,
            runs=[],
            issues=[{"code": "invalid_runs"}],
        )

    runs: list[dict[str, Any]] = []
    issues: list[dict[str, Any]] = []
    for run_index, source_run in enumerate(source_runs):
        if not isinstance(source_run, Mapping):
            issues.append({"code": "invalid_run", "run_index": run_index})
            continue
        run = _audit_run(source_run, existing_paths=existing, run_index=run_index)
        runs.append(run)
        if run["run_status"] == "failed" and not str(run.get("failure_reason", "")).strip():
            issues.append(
                {
                    "code": "failed_run_missing_reason",
                    "run_index": run_index,
                    "run_id": run.get("run_id"),
                }
            )

    completed_run_count = sum(1 for run in runs if run.get("execution_status") == "complete")
    failed_run_count = sum(1 for run in runs if run.get("execution_status") == "failed")
    in_progress_run_count = sum(1 for run in runs if run.get("execution_status") == "in_progress")
    blocked_run_count = sum(1 for run in runs if run.get("execution_status") == "blocked")
    return DPOExecutionStatusReport(
        run_count=len(runs),
        completed_run_count=completed_run_count,
        in_progress_run_count=in_progress_run_count,
        blocked_run_count=blocked_run_count,
        failed_run_count=failed_run_count,
        runs=runs,
        issues=issues,
    )


def _audit_run(
    source_run: Mapping[str, Any],
    *,
    existing_paths: set[str],
    run_index: int,
) -> dict[str, Any]:
    run = dict(source_run)
    run_status = str(run.get("run_status", "planned"))
    if run_status == "failed":
        run["execution_status"] = "failed"
        run["next_stage"] = None
        run["stages"] = []
        return run

    source_stages = run.get("stages", [])
    stages: list[dict[str, Any]] = []
    next_stage: str | None = None
    any_complete = False
    completed_stage_names: set[str] = set()
    for stage_index, source_stage in enumerate(source_stages):
        if not isinstance(source_stage, Mapping):
            stages.append(
                {
                    "stage": f"invalid_stage_{stage_index}",
                    "status": "blocked",
                    "blocker": "invalid_stage",
                    "present_inputs": [],
                    "missing_inputs": [],
                    "present_outputs": [],
                    "missing_outputs": [],
                }
            )
            if next_stage is None:
                next_stage = f"invalid_stage_{stage_index}"
            continue
        stage_name = str(source_stage.get("stage", f"stage_{stage_index}"))
        dependencies = [str(value) for value in source_stage.get("depends_on", [])]
        missing_dependencies = [value for value in dependencies if value not in completed_stage_names]
        dependency_blocker = (
            f"awaiting_{missing_dependencies[0]}" if missing_dependencies else None
        )
        stage = _audit_stage(
            source_stage,
            existing_paths=existing_paths,
            dependency_blocker=dependency_blocker,
        )
        if stage["status"] == "complete":
            any_complete = True
            completed_stage_names.add(stage_name)
        elif next_stage is None:
            next_stage = str(stage.get("stage"))
        stages.append(stage)

    all_complete = bool(stages) and all(stage["status"] == "complete" for stage in stages)
    run["stages"] = stages
    run["next_stage"] = None if all_complete else next_stage
    if all_complete:
        run["execution_status"] = "complete"
    elif any_complete:
        run["execution_status"] = "in_progress"
    else:
        run["execution_status"] = "blocked"
    return run


def _audit_stage(
    source_stage: Mapping[str, Any],
    *,
    existing_paths: set[str],
    dependency_blocker: str | None = None,
) -> dict[str, Any]:
    stage = dict(source_stage)
    inputs = _string_mapping(stage.get("inputs"))
    outputs = _string_mapping(stage.get("outputs"))
    present_inputs = sorted(path for path in inputs.values() if path in existing_paths)
    missing_inputs = sorted(path for path in inputs.values() if path and path not in existing_paths)
    present_outputs = sorted(path for path in outputs.values() if path in existing_paths)
    missing_outputs = sorted(path for path in outputs.values() if path and path not in existing_paths)
    stage["present_inputs"] = present_inputs
    stage["missing_inputs"] = missing_inputs
    stage["present_outputs"] = present_outputs
    stage["missing_outputs"] = missing_outputs
    if dependency_blocker is not None:
        stage["status"] = "blocked"
        stage["blocker"] = dependency_blocker
    elif outputs and not missing_outputs and not missing_inputs:
        stage["status"] = "complete"
        stage["blocker"] = None
    else:
        stage["status"] = "blocked"
        source_blocker = str(source_stage.get("blocker") or "")
        if missing_inputs and source_blocker == "awaiting_execution":
            stage["blocker"] = "missing_inputs"
        elif source_blocker:
            stage["blocker"] = source_blocker
        else:
            stage["blocker"] = "missing_inputs" if missing_inputs else "awaiting_execution"
    return stage


def _string_mapping(value: Any) -> dict[str, str]:
    if not isinstance(value, Mapping):
        return {}
    return {str(key): str(path) for key, path in value.items() if str(path)}
