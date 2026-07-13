from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from mias_dcms.preference_logprob_audit import audit_preference_logprobs
from mias_dcms.selectors import FORBIDDEN_SELECTOR_INPUT_FIELDS


@dataclass(frozen=True)
class PreferenceExperimentPreflightInputs:
    active_pool: Iterable[Mapping[str, Any]]
    oracle_store: Mapping[str, Mapping[str, Any]]
    logprob_rows: Iterable[Mapping[str, Any]]
    split_manifest: Mapping[str, Any]
    run_matrix: Iterable[Mapping[str, Any]]
    expected_active_pool_path: str | None = None
    expected_oracle_store_path: str | None = None
    expected_logprobs_path: str | None = None
    expected_methods: Sequence[str] = ()
    expected_seeds: Sequence[int] = ()
    id_field: str = "sample_id"


@dataclass(frozen=True)
class PreferenceExperimentPreflightReport:
    active_pool_count: int
    oracle_label_count: int
    logprob_count: int
    planned_run_count: int
    covered_methods: list[str]
    covered_seeds: list[int]
    logprob_summary: dict[str, Any]
    issues: list[dict[str, Any]] = field(default_factory=list)

    @property
    def is_ready(self) -> bool:
        return not self.issues

    def as_dict(self) -> dict[str, Any]:
        return {
            "is_ready": self.is_ready,
            "active_pool_count": self.active_pool_count,
            "oracle_label_count": self.oracle_label_count,
            "logprob_count": self.logprob_count,
            "planned_run_count": self.planned_run_count,
            "covered_methods": list(self.covered_methods),
            "covered_seeds": list(self.covered_seeds),
            "logprob_summary": dict(self.logprob_summary),
            "issues": [dict(issue) for issue in self.issues],
        }


def audit_preference_experiment_preflight(
    inputs: PreferenceExperimentPreflightInputs,
) -> PreferenceExperimentPreflightReport:
    active_rows = [dict(row) for row in inputs.active_pool]
    oracle_store = {str(sample_id): dict(row) for sample_id, row in inputs.oracle_store.items()}
    logprob_rows = [dict(row) for row in inputs.logprob_rows]
    run_rows = [dict(row) for row in inputs.run_matrix]
    issues: list[dict[str, Any]] = []

    active_ids = _ids_from_rows(active_rows, id_field=inputs.id_field, issue_code="active_pool_missing_id", issues=issues)
    logprob_ids = _ids_from_rows(
        logprob_rows,
        id_field=inputs.id_field,
        issue_code="logprob_missing_id",
        issues=issues,
    )
    oracle_ids = set(oracle_store)

    _audit_hidden_label_boundary(active_rows, issues=issues, id_field=inputs.id_field)
    _audit_id_coverage(
        active_ids=active_ids,
        oracle_ids=oracle_ids,
        logprob_ids=set(logprob_ids),
        issues=issues,
    )
    _audit_split_manifest(inputs.split_manifest, active_ids=active_ids, issues=issues)
    _audit_run_matrix(
        run_rows,
        expected_methods=inputs.expected_methods,
        expected_seeds=inputs.expected_seeds,
        expected_active_pool_path=inputs.expected_active_pool_path,
        expected_oracle_store_path=inputs.expected_oracle_store_path,
        expected_logprobs_path=inputs.expected_logprobs_path,
        issues=issues,
    )

    logprob_summary: dict[str, Any]
    try:
        _audited_rows, logprob_summary = audit_preference_logprobs(logprob_rows, id_field=inputs.id_field)
    except ValueError as exc:
        logprob_summary = {"error": str(exc), "implicit_margin_not_all_zero": False}
        issues.append({"code": "logprob_audit_failed", "message": str(exc)})

    return PreferenceExperimentPreflightReport(
        active_pool_count=len(active_rows),
        oracle_label_count=len(oracle_store),
        logprob_count=len(logprob_rows),
        planned_run_count=len(run_rows),
        covered_methods=sorted({str(row.get("method")) for row in run_rows if row.get("method") is not None}),
        covered_seeds=_sorted_ints(row.get("seed") for row in run_rows),
        logprob_summary=logprob_summary,
        issues=issues,
    )


def _audit_hidden_label_boundary(
    active_rows: Sequence[Mapping[str, Any]],
    *,
    issues: list[dict[str, Any]],
    id_field: str,
) -> None:
    for row_index, row in enumerate(active_rows):
        leaked = sorted(FORBIDDEN_SELECTOR_INPUT_FIELDS.intersection(row))
        if leaked:
            issues.append(
                {
                    "code": "hidden_label_leakage",
                    "row_index": row_index,
                    "sample_id": _row_id(row, id_field=id_field, fallback=str(row_index)),
                    "fields": leaked,
                }
            )


def _audit_id_coverage(
    *,
    active_ids: set[str],
    oracle_ids: set[str],
    logprob_ids: set[str],
    issues: list[dict[str, Any]],
) -> None:
    for sample_id in sorted(active_ids - oracle_ids):
        issues.append({"code": "oracle_missing_active_id", "sample_id": sample_id})
    for sample_id in sorted(active_ids - logprob_ids):
        issues.append({"code": "logprob_missing_active_id", "sample_id": sample_id})
    for sample_id in sorted(oracle_ids - active_ids):
        issues.append({"code": "oracle_extra_id", "sample_id": sample_id})
    for sample_id in sorted(logprob_ids - active_ids):
        issues.append({"code": "logprob_extra_id", "sample_id": sample_id})


def _audit_split_manifest(
    split_manifest: Mapping[str, Any],
    *,
    active_ids: set[str],
    issues: list[dict[str, Any]],
) -> None:
    split_fields = ("seed_ids", "active_pool_ids", "heldout_ids", "test_ids")
    split_sets: dict[str, set[str]] = {}
    for field_name in split_fields:
        values = split_manifest.get(field_name, [])
        if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
            issues.append({"code": "invalid_split_field", "field": field_name})
            values = []
        split_sets[field_name] = {str(value) for value in values}

    manifest_active_ids = split_sets["active_pool_ids"]
    for sample_id in sorted(active_ids - manifest_active_ids):
        issues.append({"code": "split_missing_active_id", "sample_id": sample_id})
    for sample_id in sorted(manifest_active_ids - active_ids):
        issues.append({"code": "split_extra_active_id", "sample_id": sample_id})

    for left_index, left_name in enumerate(split_fields):
        for right_name in split_fields[left_index + 1 :]:
            overlap = sorted(split_sets[left_name].intersection(split_sets[right_name]))
            for sample_id in overlap:
                issues.append(
                    {
                        "code": "split_overlap",
                        "sample_id": sample_id,
                        "left_split": left_name,
                        "right_split": right_name,
                    }
                )


def _audit_run_matrix(
    run_rows: Sequence[Mapping[str, Any]],
    *,
    expected_methods: Sequence[str],
    expected_seeds: Sequence[int],
    expected_active_pool_path: str | None,
    expected_oracle_store_path: str | None,
    expected_logprobs_path: str | None,
    issues: list[dict[str, Any]],
) -> None:
    covered_methods = {str(row.get("method")) for row in run_rows}
    for method in expected_methods:
        if str(method) not in covered_methods:
            issues.append({"code": "run_matrix_missing_method", "method": str(method)})

    covered_seeds = {int(row.get("seed")) for row in run_rows if _is_int_like(row.get("seed"))}
    for seed in expected_seeds:
        if int(seed) not in covered_seeds:
            issues.append({"code": "run_matrix_missing_seed", "seed": int(seed)})

    expected_paths = {
        "active_pool_path": expected_active_pool_path,
        "oracle_store_path": expected_oracle_store_path,
        "logprobs_path": expected_logprobs_path,
    }
    for row_index, row in enumerate(run_rows):
        artifacts = row.get("artifacts")
        data_config = row.get("data_config")
        artifacts = artifacts if isinstance(artifacts, Mapping) else {}
        data_config = data_config if isinstance(data_config, Mapping) else {}
        for field_name, expected_path in expected_paths.items():
            if expected_path is None:
                continue
            data_config_path = str(data_config.get(field_name) or "")
            if data_config_path and data_config_path != str(expected_path):
                issues.append(
                    {
                        "code": f"run_matrix_{field_name}_mismatch",
                        "row_index": row_index,
                        "source": "data_config",
                        "expected": str(expected_path),
                        "actual": data_config_path,
                    }
                )


def _ids_from_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    id_field: str,
    issue_code: str,
    issues: list[dict[str, Any]],
) -> set[str]:
    ids: list[str] = []
    for row_index, row in enumerate(rows):
        sample_id = _row_id(row, id_field=id_field)
        if sample_id is None:
            issues.append({"code": issue_code, "row_index": row_index})
            continue
        ids.append(sample_id)
    duplicates = sorted({sample_id for sample_id in ids if ids.count(sample_id) > 1})
    for sample_id in duplicates:
        issues.append({"code": "duplicate_sample_id", "sample_id": sample_id})
    return set(ids)


def _row_id(row: Mapping[str, Any], *, id_field: str, fallback: str | None = None) -> str | None:
    value = row.get(id_field, row.get("id", fallback))
    if value is None:
        return None
    return str(value)


def _is_int_like(value: Any) -> bool:
    try:
        int(value)
    except (TypeError, ValueError):
        return False
    return True


def _sorted_ints(values: Iterable[Any]) -> list[int]:
    parsed: set[int] = set()
    for value in values:
        if _is_int_like(value):
            parsed.add(int(value))
    return sorted(parsed)
