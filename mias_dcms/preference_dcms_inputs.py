from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from mias_dcms.preference_pool import length_gap_bin, normalized_response_length_gap
from mias_dcms.preference_selection_metrics import materialize_preference_group_fields
from mias_dcms.selectors import assert_selector_rows_are_label_safe


def build_preference_dcms_candidate_rows(
    rows: Iterable[dict[str, Any]],
    *,
    method: str,
    group_fields: Sequence[str] = (),
    group_field: str | None = None,
    id_field: str = "sample_id",
    score_field: str | None = None,
    selection_group_field: str | None = None,
    audit_group_fields: Sequence[str] = (),
) -> list[dict[str, Any]]:
    source_rows = [dict(row) for row in rows]
    assert_selector_rows_are_label_safe(source_rows)
    normalized_method = _normalize_method(method)
    resolved_score_field = score_field or f"{normalized_method}_score"
    if group_field is None and not group_fields:
        raise ValueError("group_field or group_fields must be provided")

    candidates: list[dict[str, Any]] = []
    for row in source_rows:
        sample_id = _row_id(row, id_field=id_field)
        candidate = {
            "sample_id": sample_id,
            "score": _score(row, score_field=resolved_score_field, method=normalized_method),
            "method": normalized_method,
            "source_score_field": resolved_score_field,
            "groups": _groups(row, group_field=group_field, group_fields=group_fields),
        }
        if normalized_method == "gradient_dpo":
            candidate.update(
                {
                    key: value
                    for key, value in row.items()
                    if str(key).startswith("gradient_dpo_") and value is not None
                }
            )
        if selection_group_field:
            group_value = row.get(selection_group_field)
            if group_value is None or not str(group_value):
                raise ValueError(
                    f"row {sample_id!r} is missing selection group field {selection_group_field!r}"
                )
            candidate[str(selection_group_field)] = str(group_value)
        materialized = materialize_preference_group_fields(row, group_fields=audit_group_fields)
        for field in audit_group_fields:
            if field in materialized and materialized[field] is not None:
                candidate[str(field)] = materialized[field]
        candidates.append(candidate)
    return candidates


def _normalize_method(method: str) -> str:
    normalized = str(method).strip().lower().replace("-", "_")
    aliases = {
        "reward_margin": "reward_margin",
        "margin": "reward_margin",
        "apl": "apl",
        "active_dpo": "active_dpo",
        "activedpo": "active_dpo",
        "gradient_dpo": "gradient_dpo",
        "gradientdpo": "gradient_dpo",
    }
    if normalized not in aliases:
        raise ValueError(f"unsupported preference DCMS method: {method!r}")
    return aliases[normalized]


def _row_id(row: Mapping[str, Any], *, id_field: str) -> str:
    value = row.get(id_field, row.get("id"))
    if value is None:
        raise ValueError(f"row is missing id field {id_field!r} and fallback 'id'")
    return str(value)


def _score(row: Mapping[str, Any], *, score_field: str, method: str) -> float:
    if score_field in row and row[score_field] is not None:
        return float(row[score_field])
    selector_scores = row.get("selector_scores")
    if isinstance(selector_scores, Mapping) and method in selector_scores:
        return float(selector_scores[method])
    sample_id = row.get("sample_id", row.get("id", "<unknown>"))
    raise ValueError(f"row {sample_id!r} is missing score field {score_field!r}")


def _groups(
    row: Mapping[str, Any],
    *,
    group_field: str | None,
    group_fields: Sequence[str],
) -> dict[str, float]:
    if group_field is not None:
        value = row.get(group_field)
        if not isinstance(value, Mapping):
            raise ValueError(f"group field {group_field!r} must contain an object")
        return {str(key): float(item) for key, item in value.items()}

    groups: dict[str, float] = {}
    for field in group_fields:
        value = _derived_group_value(row, field)
        groups[f"{field}={value}"] = 1.0
    return groups


def _derived_group_value(row: Mapping[str, Any], field: str) -> str:
    if field in row and row[field] is not None:
        return str(row[field])
    if field == "length_gap_bin":
        if "length_gap" in row and row["length_gap"] is not None:
            return length_gap_bin(float(row["length_gap"]))
        if "response_a" in row and "response_b" in row:
            gap = normalized_response_length_gap(str(row["response_a"]), str(row["response_b"]))
            return length_gap_bin(gap)
    if field in {"length_by_prompt_cluster", "length_gap_by_prompt_cluster"}:
        length_value = _derived_group_value(row, "length_gap_bin")
        prompt_value = _derived_group_value(row, "prompt_cluster")
        return f"{length_value}|{prompt_value}"
    if field == "prompt_cluster" and row.get("prompt_cluster_id") is not None:
        return str(row["prompt_cluster_id"])
    sample_id = row.get("sample_id", row.get("id", "<unknown>"))
    raise ValueError(f"row {sample_id!r} is missing group field {field!r}")
