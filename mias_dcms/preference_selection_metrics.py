from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from mias_dcms.preference_acquisition_audit import audit_preference_acquisition
from mias_dcms.preference_pool import length_gap_bin, normalized_response_length_gap
from mias_dcms.selection import rank_normalize_utilities
from mias_dcms.selectors import assert_selector_rows_are_label_safe


DEFAULT_PREFERENCE_AUDIT_GROUP_FIELDS = (
    "length_gap_bin",
    "source_pair",
    "prompt_cluster",
    "ab_position",
    "length_by_prompt_cluster",
)


def build_preference_selection_metrics(
    rows: Iterable[Mapping[str, Any]],
    *,
    selected_ids: Sequence[str] | None = None,
    selected_field: str = "selected",
    score_field: str | None = None,
    method: str = "selector",
    group_fields: Sequence[str] = DEFAULT_PREFERENCE_AUDIT_GROUP_FIELDS,
    constraint_violation: float = 0.0,
    utility_retained: float | None = None,
) -> dict[str, Any]:
    """Build selector metrics without reading hidden preference labels."""
    source_rows = [dict(row) for row in rows]
    if not source_rows:
        raise ValueError("preference selection rows must not be empty")
    assert_selector_rows_are_label_safe(source_rows)

    selected_set = {str(value) for value in selected_ids} if selected_ids is not None else None
    audit_rows: list[dict[str, Any]] = []
    for row in source_rows:
        sample_id = str(row.get("sample_id", row.get("id", "")))
        if not sample_id:
            raise ValueError("selector row is missing sample_id/id")
        payload = dict(row)
        if selected_set is not None:
            payload[selected_field] = int(sample_id in selected_set)
        elif selected_field not in payload:
            raise ValueError(f"rows are missing selected field {selected_field!r}")
        for field in group_fields:
            value = _group_value(payload, str(field))
            if value is not None:
                payload[str(field)] = value
        audit_rows.append(payload)

    available_fields = tuple(
        str(field)
        for field in group_fields
        if all(_group_value(row, str(field)) is not None for row in audit_rows)
    )
    selection_metrics: dict[str, Any] = {
        "acquisition_tv": 0.0,
        "maximum_propensity_ratio": 0.0,
        "total_absolute_prediction_error": 0.0,
        "utility_retained": 1.0 if utility_retained is None else float(utility_retained),
        "max_constraint_violation": float(constraint_violation),
        "audited_group_fields": list(available_fields),
        "utility_retained_not_applicable": utility_retained is None,
    }
    if available_fields:
        audit = audit_preference_acquisition(
            audit_rows,
            method=str(method),
            group_fields=available_fields,
            selected_field=selected_field,
        )
        selection_metrics.update(
            {
                "acquisition_tv": float(audit["max_acquisition_tv"]),
                "maximum_propensity_ratio": float(audit["max_propensity_ratio"]),
                "total_absolute_prediction_error": max(
                    float(report["total_absolute_prediction_error"])
                    for report in audit["by_group_field"].values()
                ),
            }
        )
    if score_field is not None:
        selection_metrics["score_field"] = str(score_field)
    return selection_metrics


def utility_retained_from_scores(
    rows: Iterable[Mapping[str, Any]],
    *,
    selected_ids: Sequence[str],
    score_field: str,
) -> float:
    source_rows = [dict(row) for row in rows]
    scores = [float(row[score_field]) for row in source_rows]
    budget = len(selected_ids)
    if budget <= 0:
        return 1.0
    utilities = rank_normalize_utilities(scores)
    top_utility = sum(sorted(utilities, reverse=True)[:budget])
    selected = {str(value) for value in selected_ids}
    selected_utility = sum(
        utility
        for row, utility in zip(source_rows, utilities, strict=True)
        if str(row.get("sample_id", row.get("id"))) in selected
    )
    return selected_utility / top_utility if top_utility > 0.0 else 1.0


def materialize_preference_group_fields(
    row: Mapping[str, Any],
    *,
    group_fields: Sequence[str] = DEFAULT_PREFERENCE_AUDIT_GROUP_FIELDS,
) -> dict[str, Any]:
    """Copy a selector-safe row while making derivable audit fields explicit."""
    payload = dict(row)
    for field in group_fields:
        value = _group_value(payload, str(field))
        if value is not None:
            payload[str(field)] = value
    return payload


def _group_value(row: Mapping[str, Any], field: str) -> str | None:
    if field in row and row[field] is not None:
        return str(row[field])
    if field == "length_gap_bin":
        if row.get("length_gap") is not None:
            return length_gap_bin(float(row["length_gap"]))
        if row.get("response_a") is not None and row.get("response_b") is not None:
            gap = normalized_response_length_gap(str(row["response_a"]), str(row["response_b"]))
            return length_gap_bin(gap)
    if field in {"length_by_prompt_cluster", "length_gap_by_prompt_cluster"}:
        length_value = _group_value(row, "length_gap_bin")
        prompt_value = _group_value(row, "prompt_cluster")
        if length_value is not None and prompt_value is not None:
            return f"{length_value}|{prompt_value}"
    if field == "prompt_cluster" and row.get("prompt_cluster_id") is not None:
        return str(row["prompt_cluster_id"])
    groups = row.get("groups")
    if isinstance(groups, Mapping):
        prefix = f"{field}="
        candidates = sorted(
            str(key)[len(prefix) :]
            for key, value in groups.items()
            if str(key).startswith(prefix) and float(value) > 0.0
        )
        if candidates:
            return candidates[0]
    return None
