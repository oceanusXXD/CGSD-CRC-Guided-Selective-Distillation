from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
import math
from typing import Any

from mias_dcms.auditing import mias_selection_audit
from mias_dcms.selectors import assert_selector_rows_are_label_safe


def audit_preference_acquisition(
    rows: Iterable[Mapping[str, Any]],
    *,
    method: str,
    group_fields: Sequence[str],
    selected_field: str = "selected",
    random_reference_rows: Iterable[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    source_rows = [dict(row) for row in rows]
    if not source_rows:
        raise ValueError("preference acquisition rows must not be empty")
    assert_selector_rows_are_label_safe(source_rows)

    fields = tuple(str(field) for field in group_fields if str(field).strip())
    if not fields:
        raise ValueError("group_fields must not be empty")

    by_group_field = {
        field: _audit_one_field(source_rows, group_field=field, selected_field=selected_field)
        for field in fields
    }
    selected_size = sum(1 for row in source_rows if bool(row[selected_field]))
    if selected_size == 0:
        raise ValueError("at least one row must be selected")

    payload: dict[str, Any] = {
        "method": str(method),
        "pool_size": len(source_rows),
        "selected_size": selected_size,
        "selected_field": selected_field,
        "group_fields": list(fields),
        "by_group_field": by_group_field,
        "max_acquisition_tv": max(field_report["acquisition_tv"] for field_report in by_group_field.values()),
        "max_propensity_ratio": max(
            field_report["maximum_propensity_ratio"] for field_report in by_group_field.values()
        ),
        "random_reference_present": random_reference_rows is not None,
    }

    if random_reference_rows is not None:
        reference_rows = [dict(row) for row in random_reference_rows]
        assert_selector_rows_are_label_safe(reference_rows)
        payload["random_reference"] = {
            "pool_size": len(reference_rows),
            "selected_size": sum(1 for row in reference_rows if bool(row[selected_field])),
            "by_group_field": {
                field: _audit_one_field(reference_rows, group_field=field, selected_field=selected_field)
                for field in fields
            },
        }
    return payload


def _audit_one_field(
    rows: list[dict[str, Any]],
    *,
    group_field: str,
    selected_field: str,
) -> dict[str, Any]:
    audit = mias_selection_audit(rows, group_field=group_field, selected_field=selected_field)
    audit_payload = audit.as_dict()
    pool_distribution = {
        group: group_report["pool_share"] for group, group_report in audit_payload["groups"].items()
    }
    selected_distribution = {
        group: group_report["actual_selected_share"] for group, group_report in audit_payload["groups"].items()
    }
    return {
        "group_field": group_field,
        "pool_size": audit.pool_size,
        "selected_size": audit.selected_size,
        "acquisition_tv": audit.acquisition_tv,
        "acquisition_js": _jensen_shannon_divergence(pool_distribution, selected_distribution),
        "maximum_propensity_ratio": audit.maximum_propensity_ratio,
        "total_absolute_prediction_error": audit.total_absolute_prediction_error,
        "groups": audit_payload["groups"],
    }


def _jensen_shannon_divergence(
    left: Mapping[str, float],
    right: Mapping[str, float],
) -> float:
    groups = sorted(set(left) | set(right))
    left_values = [float(left.get(group, 0.0)) for group in groups]
    right_values = [float(right.get(group, 0.0)) for group in groups]
    midpoint = [(left_value + right_value) / 2.0 for left_value, right_value in zip(left_values, right_values)]
    return 0.5 * _kl_divergence(left_values, midpoint) + 0.5 * _kl_divergence(right_values, midpoint)


def _kl_divergence(left: Sequence[float], right: Sequence[float]) -> float:
    total = 0.0
    for left_value, right_value in zip(left, right, strict=True):
        if left_value > 0.0:
            if right_value <= 0.0:
                raise ValueError("right distribution has zero mass where left has positive mass")
            total += left_value * math.log(left_value / right_value)
    return total
