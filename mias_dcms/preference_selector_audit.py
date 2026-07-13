from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping
import math
from typing import Any

from mias_dcms.selectors import assert_selector_rows_are_label_safe, select_top_budget


def audit_preference_selector_scores(
    rows: Iterable[Mapping[str, Any]],
    *,
    method: str,
    budget: int,
    id_field: str = "sample_id",
    score_field: str | None = None,
    length_field: str = "length_gap",
    swap_pair_field: str = "swap_pair_id",
    position_field: str = "ab_position",
    selector_compute_seconds: float = 0.0,
    require_non_degenerate: bool = True,
) -> dict[str, Any]:
    source_rows = [dict(row) for row in rows]
    if not source_rows:
        raise ValueError("selector rows must not be empty")
    assert_selector_rows_are_label_safe(source_rows)

    normalized_method = _normalize_method(method)
    resolved_score_field = score_field or f"{normalized_method}_score"
    sample_ids = [_row_id(row, id_field=id_field) for row in source_rows]
    scores = [
        _score_from_row(row, method=normalized_method, score_field=resolved_score_field)
        for row in source_rows
    ]
    if len(set(sample_ids)) != len(sample_ids):
        raise ValueError("sample ids must be unique")

    variance = _population_variance(scores)
    score_not_all_equal = variance > 1e-12
    if require_non_degenerate and not score_not_all_equal:
        raise ValueError("score is degenerate: all selector scores are equal")

    selected_ids = select_top_budget(sample_ids=sample_ids, scores=scores, budget=int(budget))
    repeated_selected_ids = len(set(selected_ids)) != len(selected_ids)

    summary: dict[str, Any] = {
        "method": normalized_method,
        "score_field": resolved_score_field,
        "pool_size": len(source_rows),
        "budget": int(budget),
        "score_min": min(scores),
        "score_max": max(scores),
        "score_mean": _mean(scores),
        "score_variance": variance,
        "score_not_all_equal": score_not_all_equal,
        "selected_ids": selected_ids,
        "selected_count": len(selected_ids),
        "top_budget_reproducible": selected_ids == select_top_budget(
            sample_ids=sample_ids,
            scores=scores,
            budget=int(budget),
        ),
        "selected_ids_have_duplicates": repeated_selected_ids,
        "expected_oracle_calls_after_reveal": len(selected_ids),
        "oracle_calls_equal_budget": len(selected_ids) == int(budget),
        "selector_compute_seconds": float(selector_compute_seconds),
    }

    length_values = _optional_numeric_values(source_rows, length_field)
    if length_values is not None:
        summary["length_field"] = length_field
        summary["score_length_correlation"] = _pearson_correlation(scores, length_values)

    summary.update(
        _ab_swap_summary(
            source_rows,
            method=normalized_method,
            score_field=resolved_score_field,
            swap_pair_field=swap_pair_field,
            position_field=position_field,
        )
    )
    return summary


def _normalize_method(method: str) -> str:
    return str(method).strip().lower().replace("-", "_")


def _row_id(row: Mapping[str, Any], *, id_field: str) -> str:
    value = row.get(id_field, row.get("id"))
    if value is None:
        raise ValueError(f"row is missing id field {id_field!r} and fallback 'id'")
    return str(value)


def _score_from_row(row: Mapping[str, Any], *, method: str, score_field: str) -> float:
    if score_field in row and row[score_field] is not None:
        return float(row[score_field])
    selector_scores = row.get("selector_scores")
    if isinstance(selector_scores, Mapping) and method in selector_scores:
        return float(selector_scores[method])
    sample_id = row.get("sample_id", row.get("id", "<unknown>"))
    raise ValueError(f"row {sample_id!r} is missing score field {score_field!r}")


def _optional_numeric_values(
    rows: list[Mapping[str, Any]],
    field: str,
) -> list[float] | None:
    if not field:
        return None
    values: list[float] = []
    for row in rows:
        if field not in row or row[field] is None:
            return None
        values.append(float(row[field]))
    return values


def _ab_swap_summary(
    rows: list[Mapping[str, Any]],
    *,
    method: str,
    score_field: str,
    swap_pair_field: str,
    position_field: str,
) -> dict[str, Any]:
    pairs: dict[str, dict[str, float]] = defaultdict(dict)
    for row in rows:
        if swap_pair_field not in row or position_field not in row:
            continue
        pair_id = str(row[swap_pair_field])
        position = str(row[position_field]).strip().lower()
        if position in {"original", "swapped"}:
            pairs[pair_id][position] = _score_from_row(row, method=method, score_field=score_field)

    deltas = [
        abs(pair_scores["original"] - pair_scores["swapped"])
        for pair_scores in pairs.values()
        if "original" in pair_scores and "swapped" in pair_scores
    ]
    return {
        "ab_swap_pair_count": len(deltas),
        "ab_swap_mean_abs_score_delta": _mean(deltas) if deltas else 0.0,
        "ab_swap_max_abs_score_delta": max(deltas) if deltas else 0.0,
    }


def _mean(values: list[float]) -> float:
    if not values:
        raise ValueError("values must not be empty")
    return sum(values) / len(values)


def _population_variance(values: list[float]) -> float:
    mean = _mean(values)
    return sum((value - mean) ** 2 for value in values) / len(values)


def _pearson_correlation(left: list[float], right: list[float]) -> float:
    if len(left) != len(right):
        raise ValueError("correlation inputs must have equal length")
    if len(left) < 2:
        return 0.0
    left_mean = _mean(left)
    right_mean = _mean(right)
    numerator = sum((x - left_mean) * (y - right_mean) for x, y in zip(left, right, strict=True))
    left_denominator = math.sqrt(sum((x - left_mean) ** 2 for x in left))
    right_denominator = math.sqrt(sum((y - right_mean) ** 2 for y in right))
    denominator = left_denominator * right_denominator
    if denominator == 0.0:
        return 0.0
    return numerator / denominator
