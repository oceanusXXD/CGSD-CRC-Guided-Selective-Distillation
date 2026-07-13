from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence


ALLOWED_FREEZE_POLICIES = ("bug-fixes-only", "supplement-only")


@dataclass(frozen=True)
class ResultFreezePackValidationReport:
    main_table_count: int
    appendix_table_count: int
    figure_data_count: int
    issues: list[dict[str, Any]] = field(default_factory=list)

    @property
    def is_ready(self) -> bool:
        return not self.issues

    def as_dict(self) -> dict[str, Any]:
        return {
            "is_ready": self.is_ready,
            "main_table_count": self.main_table_count,
            "appendix_table_count": self.appendix_table_count,
            "figure_data_count": self.figure_data_count,
            "issues": [dict(issue) for issue in self.issues],
        }


def validate_result_freeze_pack(
    pack: Mapping[str, Any],
    *,
    expected_main_tables: Sequence[str],
    expected_figures: Sequence[str],
    expected_metrics: Sequence[str],
    expected_baselines: Sequence[str],
) -> ResultFreezePackValidationReport:
    issues: list[dict[str, Any]] = []

    results_manifest = pack.get("results_manifest")
    if not isinstance(results_manifest, Mapping):
        issues.append({"code": "missing_results_manifest"})
    else:
        _validate_artifact(results_manifest, artifact_name="results_manifest", issues=issues)

    main_tables = pack.get("main_tables")
    if not isinstance(main_tables, Mapping):
        main_tables = {}
        issues.append({"code": "missing_main_tables"})
    for table_name in expected_main_tables:
        artifact = main_tables.get(table_name)
        if not isinstance(artifact, Mapping):
            issues.append({"code": "missing_main_table", "artifact": str(table_name)})
            continue
        _validate_artifact(artifact, artifact_name=str(table_name), issues=issues)

    appendix_tables = pack.get("appendix_tables")
    if appendix_tables is None:
        appendix_tables = {}
    if not isinstance(appendix_tables, Mapping):
        appendix_tables = {}
        issues.append({"code": "invalid_appendix_tables"})
    for table_name, artifact in appendix_tables.items():
        if not isinstance(artifact, Mapping):
            issues.append({"code": "invalid_appendix_table", "artifact": str(table_name)})
            continue
        _validate_artifact(artifact, artifact_name=str(table_name), issues=issues)

    figure_data = pack.get("figure_data")
    if not isinstance(figure_data, Mapping):
        figure_data = {}
        issues.append({"code": "missing_figure_data_group"})
    for figure_name in expected_figures:
        artifact = figure_data.get(figure_name)
        if not isinstance(artifact, Mapping):
            issues.append({"code": "missing_figure_data", "artifact": str(figure_name)})
            continue
        _validate_artifact(artifact, artifact_name=str(figure_name), issues=issues)

    claim_evidence_map = pack.get("claim_evidence_map")
    if not isinstance(claim_evidence_map, Mapping):
        issues.append({"code": "missing_claim_evidence_map"})
    else:
        _validate_artifact(claim_evidence_map, artifact_name="claim_evidence_map", issues=issues)

    frozen_protocol = pack.get("frozen_protocol")
    if not isinstance(frozen_protocol, Mapping):
        issues.append({"code": "missing_frozen_protocol"})
    else:
        _validate_frozen_protocol(
            frozen_protocol,
            expected_metrics=expected_metrics,
            expected_baselines=expected_baselines,
            issues=issues,
        )

    return ResultFreezePackValidationReport(
        main_table_count=len(main_tables),
        appendix_table_count=len(appendix_tables),
        figure_data_count=len(figure_data),
        issues=issues,
    )


def _validate_artifact(
    artifact: Mapping[str, Any],
    *,
    artifact_name: str,
    issues: list[dict[str, Any]],
) -> None:
    if not str(artifact.get("path", "")).strip():
        issues.append({"code": "artifact_missing_path", "artifact": artifact_name})
    if not artifact.get("input_result_files"):
        issues.append({"code": "artifact_missing_input_files", "artifact": artifact_name})
    if not str(artifact.get("aggregation_rule", "")).strip():
        issues.append({"code": "artifact_missing_aggregation_rule", "artifact": artifact_name})
    if "seed_count" not in artifact:
        issues.append({"code": "artifact_missing_seed_count", "artifact": artifact_name})
    if not str(artifact.get("error_bar", "")).strip():
        issues.append({"code": "artifact_missing_error_bar", "artifact": artifact_name})
    if "includes_failed_runs" not in artifact:
        issues.append({"code": "artifact_missing_failed_run_policy", "artifact": artifact_name})


def _validate_frozen_protocol(
    protocol: Mapping[str, Any],
    *,
    expected_metrics: Sequence[str],
    expected_baselines: Sequence[str],
    issues: list[dict[str, Any]],
) -> None:
    metrics = {str(value) for value in protocol.get("metrics", [])}
    baselines = {str(value) for value in protocol.get("baselines", [])}
    for metric in expected_metrics:
        if str(metric) not in metrics:
            issues.append({"code": "missing_frozen_metric", "metric": str(metric)})
    for baseline in expected_baselines:
        if str(baseline) not in baselines:
            issues.append({"code": "missing_frozen_baseline", "baseline": str(baseline)})
    if not str(protocol.get("judge_version", "")).strip():
        issues.append({"code": "missing_judge_version"})
    if str(protocol.get("freeze_policy", "")) not in ALLOWED_FREEZE_POLICIES:
        issues.append(
            {
                "code": "invalid_freeze_policy",
                "freeze_policy": str(protocol.get("freeze_policy", "")),
                "allowed": list(ALLOWED_FREEZE_POLICIES),
            }
        )
