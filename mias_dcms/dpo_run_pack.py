from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product
from typing import Any, Iterable, Mapping, Sequence


DPO_MAIN_METHODS = (
    "Random",
    "Reward Margin",
    "APL",
    "ActiveDPO",
    "APL+DCMS",
    "ActiveDPO+DCMS",
)

REQUIRED_DPO_SELECTION_METRICS = (
    "acquisition_tv",
    "utility_retained",
    "max_constraint_violation",
)

REQUIRED_DPO_TRAINING_METRICS = (
    "dpo_train_row_count",
    "update_steps",
    "training_token_budget",
)

REQUIRED_DPO_EVALUATION_METRICS = (
    "preference_accuracy",
    "worst_group_preference_accuracy",
    "length_controlled_win_rate",
    "capability_regression",
    "aulc",
)

REQUIRED_DPO_COST_METRICS = (
    "seed_label_count",
    "active_label_count",
    "evaluation_label_count",
    "judge_calls",
    "train_tokens",
    "selector_compute_seconds",
    "oracle_label_calls",
)


@dataclass(frozen=True)
class DPORunPackValidationReport:
    expected_run_count: int
    observed_run_count: int
    completed_run_count: int
    failed_run_count: int
    missing_run_count: int
    covered_methods: list[str]
    covered_datasets: list[str]
    covered_models: list[str]
    covered_budgets: list[int]
    covered_seeds: list[int]
    issues: list[dict[str, Any]] = field(default_factory=list)

    @property
    def is_ready(self) -> bool:
        return not self.issues

    def as_dict(self) -> dict[str, Any]:
        return {
            "is_ready": self.is_ready,
            "expected_run_count": self.expected_run_count,
            "observed_run_count": self.observed_run_count,
            "completed_run_count": self.completed_run_count,
            "failed_run_count": self.failed_run_count,
            "missing_run_count": self.missing_run_count,
            "covered_methods": list(self.covered_methods),
            "covered_datasets": list(self.covered_datasets),
            "covered_models": list(self.covered_models),
            "covered_budgets": list(self.covered_budgets),
            "covered_seeds": list(self.covered_seeds),
            "issues": [dict(issue) for issue in self.issues],
        }


def validate_dpo_run_pack(
    rows: Iterable[Mapping[str, Any]],
    *,
    expected_datasets: Sequence[str],
    expected_models: Sequence[str],
    expected_budgets: Sequence[int],
    expected_seeds: Sequence[int],
    required_methods: Sequence[str] = DPO_MAIN_METHODS,
    required_selection_metrics: Sequence[str] = REQUIRED_DPO_SELECTION_METRICS,
    required_training_metrics: Sequence[str] = REQUIRED_DPO_TRAINING_METRICS,
    required_evaluation_metrics: Sequence[str] = REQUIRED_DPO_EVALUATION_METRICS,
    required_cost_metrics: Sequence[str] = REQUIRED_DPO_COST_METRICS,
) -> DPORunPackValidationReport:
    run_rows = [dict(row) for row in rows]
    issues: list[dict[str, Any]] = []
    by_key: dict[str, list[dict[str, Any]]] = {}

    for row_index, row in enumerate(run_rows):
        key = _run_key_from_row(row)
        by_key.setdefault(key, []).append(row)
        if len(by_key[key]) > 1:
            issues.append({"code": "duplicate_run", "row_index": row_index, "run_key": key})

        status = str(row.get("run_status", "completed"))
        if status == "completed":
            _validate_completed_row(
                row,
                row_index=row_index,
                run_key=key,
                issues=issues,
                required_selection_metrics=required_selection_metrics,
                required_training_metrics=required_training_metrics,
                required_evaluation_metrics=required_evaluation_metrics,
                required_cost_metrics=required_cost_metrics,
            )
        elif status == "failed":
            issue = {"code": "failed_run", "row_index": row_index, "run_key": key}
            reason = row.get("failure_reason")
            if reason:
                issue["failure_reason"] = str(reason)
            issues.append(issue)
            if not str(reason or "").strip():
                issues.append(
                    {
                        "code": "failed_run_missing_reason",
                        "row_index": row_index,
                        "run_key": key,
                    }
                )
        else:
            issues.append(
                {
                    "code": "invalid_run_status",
                    "row_index": row_index,
                    "run_key": key,
                    "run_status": status,
                }
            )

    expected_keys = [
        _run_key(dataset=dataset, model=model, budget=budget, seed=seed, method=method)
        for dataset, model, budget, seed, method in product(
            [str(value) for value in expected_datasets],
            [str(value) for value in expected_models],
            [int(value) for value in expected_budgets],
            [int(value) for value in expected_seeds],
            [str(value) for value in required_methods],
        )
    ]

    missing_keys = [key for key in expected_keys if key not in by_key]
    for key in missing_keys:
        issues.append({"code": "missing_run", "run_key": key})

    completed_run_count = sum(1 for row in run_rows if str(row.get("run_status", "completed")) == "completed")
    failed_run_count = sum(1 for row in run_rows if str(row.get("run_status", "completed")) == "failed")

    return DPORunPackValidationReport(
        expected_run_count=len(expected_keys),
        observed_run_count=len(run_rows),
        completed_run_count=completed_run_count,
        failed_run_count=failed_run_count,
        missing_run_count=len(missing_keys),
        covered_methods=_covered_methods(run_rows, required_methods),
        covered_datasets=_sorted_strings(row.get("dataset") for row in run_rows),
        covered_models=_sorted_strings(row.get("model") for row in run_rows),
        covered_budgets=_sorted_ints(row.get("budget") for row in run_rows),
        covered_seeds=_sorted_ints(row.get("seed") for row in run_rows),
        issues=issues,
    )


def validate_paper_artifact_manifest(
    manifest: Mapping[str, Any],
    *,
    expected_figures: Sequence[str],
    expected_tables: Sequence[str],
    expected_seed_count: int | None = None,
) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []
    if not isinstance(manifest.get("results_manifest"), Mapping):
        issues.append({"code": "missing_results_manifest"})

    for group_name, expected_names in (("figures", expected_figures), ("tables", expected_tables)):
        artifacts = manifest.get(group_name)
        if not isinstance(artifacts, Mapping):
            artifacts = {}
            issues.append({"code": "missing_artifact_group", "group": group_name})
        for artifact_name in expected_names:
            artifact = artifacts.get(artifact_name)
            if not isinstance(artifact, Mapping):
                issues.append(
                    {
                        "code": "missing_artifact",
                        "group": group_name,
                        "artifact": artifact_name,
                    }
                )
                continue
            _validate_artifact(
                artifact,
                group_name=group_name,
                artifact_name=str(artifact_name),
                expected_seed_count=expected_seed_count,
                issues=issues,
            )
    return issues


def _validate_completed_row(
    row: Mapping[str, Any],
    *,
    row_index: int,
    run_key: str,
    issues: list[dict[str, Any]],
    required_selection_metrics: Sequence[str],
    required_training_metrics: Sequence[str],
    required_evaluation_metrics: Sequence[str],
    required_cost_metrics: Sequence[str],
) -> None:
    if int(row.get("selected_count", -1)) != int(row.get("budget", -2)):
        issues.append({"code": "selected_count_budget_mismatch", "row_index": row_index, "run_key": run_key})
    if not str(row.get("config_hash", "")).strip():
        issues.append({"code": "missing_config_hash", "row_index": row_index, "run_key": run_key})

    metric_groups = (
        ("selection_metrics", required_selection_metrics),
        ("training_metrics", required_training_metrics),
        ("evaluation_metrics", required_evaluation_metrics),
        ("cost_metrics", required_cost_metrics),
    )
    for group_name, required_names in metric_groups:
        metrics = row.get(group_name)
        if not isinstance(metrics, Mapping):
            issues.append(
                {
                    "code": "missing_metric_group",
                    "row_index": row_index,
                    "run_key": run_key,
                    "metric_group": group_name,
                }
            )
            continue
        for metric_name in required_names:
            metric_path = f"{group_name}.{metric_name}"
            if metric_name not in metrics:
                issues.append(
                    {
                        "code": "missing_metric",
                        "row_index": row_index,
                        "run_key": run_key,
                        "metric": metric_path,
                    }
                )
                continue
            try:
                value = float(metrics[metric_name])
            except (TypeError, ValueError):
                issues.append(
                    {
                        "code": "non_numeric_metric",
                        "row_index": row_index,
                        "run_key": run_key,
                        "metric": metric_path,
                    }
                )
                continue
            if group_name == "cost_metrics" and value < 0.0:
                issues.append(
                    {
                        "code": "negative_cost_metric",
                        "row_index": row_index,
                        "run_key": run_key,
                        "metric": metric_path,
                    }
                )


def _validate_artifact(
    artifact: Mapping[str, Any],
    *,
    group_name: str,
    artifact_name: str,
    expected_seed_count: int | None,
    issues: list[dict[str, Any]],
) -> None:
    if not artifact.get("input_result_files"):
        issues.append(_artifact_issue("artifact_missing_inputs", group_name, artifact_name))
    if not str(artifact.get("aggregation_rule", "")).strip():
        issues.append(_artifact_issue("artifact_missing_aggregation_rule", group_name, artifact_name))
    if not str(artifact.get("error_bar", "")).strip():
        issues.append(_artifact_issue("artifact_missing_error_bar", group_name, artifact_name))
    if "includes_failed_runs" not in artifact:
        issues.append(_artifact_issue("artifact_missing_failed_run_policy", group_name, artifact_name))
    if expected_seed_count is not None:
        seed_count = artifact.get("seed_count")
        if int(seed_count if seed_count is not None else -1) != int(expected_seed_count):
            issue = _artifact_issue("artifact_seed_count_mismatch", group_name, artifact_name)
            issue["expected_seed_count"] = int(expected_seed_count)
            issue["actual_seed_count"] = seed_count
            issues.append(issue)


def _artifact_issue(code: str, group_name: str, artifact_name: str) -> dict[str, Any]:
    return {"code": code, "group": group_name, "artifact": artifact_name}


def _run_key_from_row(row: Mapping[str, Any]) -> str:
    return _run_key(
        dataset=str(row.get("dataset", "")),
        model=str(row.get("model", "")),
        budget=int(row.get("budget", -1)),
        seed=int(row.get("seed", -1)),
        method=str(row.get("method", "")),
    )


def _run_key(*, dataset: str, model: str, budget: int, seed: int, method: str) -> str:
    return f"{dataset}|{model}|{int(budget)}|{int(seed)}|{method}"


def _sorted_strings(values: Iterable[Any]) -> list[str]:
    return sorted({str(value) for value in values if value is not None})


def _sorted_ints(values: Iterable[Any]) -> list[int]:
    return sorted({int(value) for value in values if value is not None})


def _covered_methods(rows: Sequence[Mapping[str, Any]], required_methods: Sequence[str]) -> list[str]:
    observed = {str(row.get("method")) for row in rows if row.get("method") is not None}
    ordered = [str(method) for method in required_methods if str(method) in observed]
    extras = sorted(observed.difference(ordered))
    return [*ordered, *extras]
