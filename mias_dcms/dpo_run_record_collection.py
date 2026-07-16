from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from mias_dcms.dpo_run_pack import (
    REQUIRED_DPO_COST_METRICS,
    REQUIRED_DPO_EVALUATION_METRICS,
    REQUIRED_DPO_SELECTION_METRICS,
    REQUIRED_DPO_TRAINING_METRICS,
)
from mias_dcms.preference_run_summary import build_preference_run_record, estimate_preference_train_tokens


SUMMARY_ARTIFACTS = (
    "selection_summary_path",
    "revealed_rows_path",
    "dpo_train_rows_path",
    "training_summary_path",
    "evaluation_metrics_path",
    "cost_report_path",
)


@dataclass(frozen=True)
class DPORunRecordCollectionReport:
    completed_run_count: int
    failed_run_count: int
    incomplete_run_count: int
    records: list[dict[str, Any]] = field(default_factory=list)
    issues: list[dict[str, Any]] = field(default_factory=list)

    @property
    def is_ready(self) -> bool:
        return not self.issues and self.failed_run_count == 0 and self.incomplete_run_count == 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "is_ready": self.is_ready,
            "record_count": len(self.records),
            "completed_run_count": self.completed_run_count,
            "failed_run_count": self.failed_run_count,
            "incomplete_run_count": self.incomplete_run_count,
            "issue_count": len(self.issues),
            "issues": [dict(issue) for issue in self.issues],
        }


def collect_dpo_run_records(
    manifest: Mapping[str, Any],
    *,
    artifact_payloads: Mapping[str, Any],
) -> DPORunRecordCollectionReport:
    records: list[dict[str, Any]] = []
    issues: list[dict[str, Any]] = []
    completed_run_count = 0
    failed_run_count = 0
    incomplete_run_count = 0

    runs = manifest.get("runs", [])
    if not isinstance(runs, list):
        return DPORunRecordCollectionReport(
            completed_run_count=0,
            failed_run_count=0,
            incomplete_run_count=0,
            issues=[{"code": "invalid_runs"}],
        )

    for run_index, run in enumerate(runs):
        if not isinstance(run, Mapping):
            issues.append({"code": "invalid_run", "run_index": run_index})
            continue
        run_status = str(run.get("run_status", "planned"))
        if run_status == "failed":
            failed_run_count += 1
            records.append(_failed_record(run))
            continue

        artifacts = run.get("artifacts", {})
        if not isinstance(artifacts, Mapping):
            artifacts = {}
        missing_artifacts = _missing_payload_issues(
            run,
            artifacts=artifacts,
            artifact_payloads=artifact_payloads,
            run_index=run_index,
        )
        issues.extend(missing_artifacts)
        if missing_artifacts:
            incomplete_run_count += 1
            records.append(_incomplete_record(run, artifacts=artifacts, artifact_payloads=artifact_payloads))
            continue

        record, metric_issues = _completed_record_or_metric_issues(
            run,
            artifacts=artifacts,
            artifact_payloads=artifact_payloads,
            run_index=run_index,
        )
        issues.extend(metric_issues)
        if metric_issues:
            incomplete_run_count += 1
            record["run_status"] = "incomplete"
        else:
            completed_run_count += 1
            record["run_status"] = "completed"
        records.append(record)

    return DPORunRecordCollectionReport(
        completed_run_count=completed_run_count,
        failed_run_count=failed_run_count,
        incomplete_run_count=incomplete_run_count,
        records=records,
        issues=issues,
    )


def _failed_record(run: Mapping[str, Any]) -> dict[str, Any]:
    record = _base_record(run)
    record.update(
        {
            "run_status": "failed",
            "failure_reason": str(run.get("failure_reason", "")),
            "selected_count": 0,
            "selection_metrics": {},
            "training_metrics": {},
            "evaluation_metrics": {},
            "cost_metrics": {},
        }
    )
    return record


def _incomplete_record(
    run: Mapping[str, Any],
    *,
    artifacts: Mapping[str, Any],
    artifact_payloads: Mapping[str, Any],
) -> dict[str, Any]:
    selection_summary = _payload(artifacts, "selection_summary_path", artifact_payloads, default={})
    training_rows = _payload(artifacts, "dpo_train_rows_path", artifact_payloads, default=[])
    training_summary = _payload(artifacts, "training_summary_path", artifact_payloads, default={})
    evaluation_payload = _payload(artifacts, "evaluation_metrics_path", artifact_payloads, default={})
    cost_report = _payload(artifacts, "cost_report_path", artifact_payloads, default={})
    selected_count = _selected_count(selection_summary, run)
    record = _base_record(run)
    record.update(
        {
            "run_status": "incomplete",
            "selected_count": selected_count,
            "selection_metrics": _selection_metrics(selection_summary),
            "training_metrics": _training_metrics(training_summary),
            "evaluation_metrics": _evaluation_metrics(evaluation_payload),
            "cost_metrics": _cost_metrics(
                cost_report,
                selected_count=selected_count,
                training_rows=_as_rows(training_rows),
            ),
        }
    )
    return record


def _completed_record_or_metric_issues(
    run: Mapping[str, Any],
    *,
    artifacts: Mapping[str, Any],
    artifact_payloads: Mapping[str, Any],
    run_index: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    selection_summary = _as_mapping(_payload(artifacts, "selection_summary_path", artifact_payloads, default={}))
    revealed_rows = _as_rows(_payload(artifacts, "revealed_rows_path", artifact_payloads, default=[]))
    training_rows = _as_rows(_payload(artifacts, "dpo_train_rows_path", artifact_payloads, default=[]))
    training_summary = _as_mapping(_payload(artifacts, "training_summary_path", artifact_payloads, default={}))
    evaluation_payload = _as_mapping(_payload(artifacts, "evaluation_metrics_path", artifact_payloads, default={}))
    cost_report = _as_mapping(_payload(artifacts, "cost_report_path", artifact_payloads, default={}))

    selected_count = _selected_count(selection_summary, run)
    training_metrics = _training_metrics(training_summary)
    reveal_summary = {
        "revealed_count": selected_count,
        "dpo_train_row_count": training_metrics.get("dpo_train_row_count", len(training_rows)),
        "unrevealed_count": max(0, selected_count - len(revealed_rows)),
    }
    run_record = build_preference_run_record(
        dataset=str(run.get("dataset", "")),
        model=str(run.get("model", "")),
        method=str(run.get("method", "")),
        budget=int(run.get("budget", 0)),
        seed=int(run.get("seed", 0)),
        config_hash=str(run.get("config_hash", "")),
        selection_summary=selection_summary,
        reveal_summary=reveal_summary,
        training_rows=training_rows,
        training_metrics=training_metrics,
        evaluation_metrics=_evaluation_metrics(evaluation_payload),
        seed_label_count=int(cost_report.get("seed_label_count", 0)),
        evaluation_label_count=int(cost_report.get("evaluation_label_count", 0)),
        judge_calls=int(cost_report.get("judge_calls", 0)),
        selector_compute_seconds=float(cost_report.get("selector_compute_seconds", 0.0)),
        train_tokens=(
            int(cost_report["train_tokens"])
            if cost_report.get("train_tokens") is not None
            else None
        ),
        oracle_label_calls=(
            int(cost_report["oracle_label_calls"])
            if cost_report.get("oracle_label_calls") is not None
            else None
        ),
    )
    record = run_record.as_dict()
    metric_issues = _metric_issues(
        run,
        run_index=run_index,
        selection_metrics=record["selection_metrics"],
        training_metrics=record["training_metrics"],
        evaluation_metrics=record["evaluation_metrics"],
        cost_metrics=record["cost_metrics"],
        explicit_cost_report=cost_report,
    )
    return record, metric_issues


def _missing_payload_issues(
    run: Mapping[str, Any],
    *,
    artifacts: Mapping[str, Any],
    artifact_payloads: Mapping[str, Any],
    run_index: int,
) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []
    for artifact_name in SUMMARY_ARTIFACTS:
        path = str(artifacts.get(artifact_name, ""))
        if not path or path not in artifact_payloads:
            issues.append(
                {
                    "code": "missing_required_artifact_payload",
                    "run_index": run_index,
                    "run_id": str(run.get("run_id", "")),
                    "method": str(run.get("method", "")),
                    "artifact": artifact_name,
                    "path": path,
                }
            )
    return issues


def _metric_issues(
    run: Mapping[str, Any],
    *,
    run_index: int,
    selection_metrics: Mapping[str, Any],
    training_metrics: Mapping[str, Any],
    evaluation_metrics: Mapping[str, Any],
    cost_metrics: Mapping[str, Any],
    explicit_cost_report: Mapping[str, Any],
) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []
    metric_groups: tuple[tuple[str, Mapping[str, Any], tuple[str, ...]], ...] = (
        ("selection_metrics", selection_metrics, REQUIRED_DPO_SELECTION_METRICS),
        ("training_metrics", training_metrics, REQUIRED_DPO_TRAINING_METRICS),
        ("evaluation_metrics", evaluation_metrics, REQUIRED_DPO_EVALUATION_METRICS),
        ("cost_metrics", cost_metrics, REQUIRED_DPO_COST_METRICS),
    )
    for group_name, observed, required_metrics in metric_groups:
        for metric in required_metrics:
            if metric not in observed or observed.get(metric) is None:
                issues.append(_missing_metric_issue(run, run_index, f"{group_name}.{metric}"))

    for metric in ("seed_label_count", "evaluation_label_count", "judge_calls", "selector_compute_seconds"):
        if metric not in explicit_cost_report or explicit_cost_report.get(metric) is None:
            issues.append(_missing_metric_issue(run, run_index, f"cost_metrics.{metric}"))
    return _dedupe_issues(issues)


def _missing_metric_issue(run: Mapping[str, Any], run_index: int, metric: str) -> dict[str, Any]:
    return {
        "code": "missing_required_metric",
        "run_index": run_index,
        "run_id": str(run.get("run_id", "")),
        "method": str(run.get("method", "")),
        "metric": metric,
    }


def _base_record(run: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "dataset": str(run.get("dataset", "")),
        "model": str(run.get("model", "")),
        "method": str(run.get("method", "")),
        "budget": int(run.get("budget", 0)),
        "seed": int(run.get("seed", 0)),
        "config_hash": str(run.get("config_hash", "")),
    }


def _payload(
    artifacts: Mapping[str, Any],
    artifact_name: str,
    artifact_payloads: Mapping[str, Any],
    *,
    default: Any,
) -> Any:
    path = str(artifacts.get(artifact_name, ""))
    return artifact_payloads.get(path, default)


def _selected_count(selection_summary: Any, run: Mapping[str, Any]) -> int:
    if isinstance(selection_summary, Mapping) and selection_summary.get("selected_count") is not None:
        return int(selection_summary["selected_count"])
    return int(run.get("budget", 0))


def _selection_metrics(selection_summary: Any) -> dict[str, Any]:
    summary = _as_mapping(selection_summary)
    metrics = dict(_as_mapping(summary.get("selection_metrics")))
    for field in ("pool_size", "selected_score_min", "selected_score_max", "utility_retained", "max_constraint_violation"):
        if field in summary and summary[field] is not None:
            metrics[field] = summary[field]
    return metrics


def _training_metrics(training_summary: Any) -> dict[str, Any]:
    summary = _as_mapping(training_summary)
    if isinstance(summary.get("training_metrics"), Mapping):
        return dict(summary["training_metrics"])
    return dict(summary)


def _evaluation_metrics(evaluation_payload: Any) -> dict[str, Any]:
    payload = _as_mapping(evaluation_payload)
    if isinstance(payload.get("evaluation_metrics"), Mapping):
        return dict(payload["evaluation_metrics"])
    return dict(payload)


def _cost_metrics(cost_report: Any, *, selected_count: int, training_rows: list[dict[str, Any]]) -> dict[str, Any]:
    report = _as_mapping(cost_report)
    return {
        "seed_label_count": int(report.get("seed_label_count", 0)),
        "active_label_count": int(selected_count),
        "evaluation_label_count": int(report.get("evaluation_label_count", 0)),
        "judge_calls": int(report.get("judge_calls", 0)),
        "train_tokens": int(
            report["train_tokens"]
            if report.get("train_tokens") is not None
            else estimate_preference_train_tokens(training_rows)
        ),
        "selector_compute_seconds": float(report.get("selector_compute_seconds", 0.0)),
        "oracle_label_calls": int(
            report["oracle_label_calls"]
            if report.get("oracle_label_calls") is not None
            else selected_count
        ),
    }


def _as_mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _as_rows(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [dict(row) for row in value if isinstance(row, Mapping)]


def _dedupe_issues(issues: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[tuple[tuple[str, str], ...]] = set()
    deduped: list[dict[str, Any]] = []
    for issue in issues:
        key = tuple(sorted((str(k), str(v)) for k, v in issue.items()))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(issue)
    return deduped
