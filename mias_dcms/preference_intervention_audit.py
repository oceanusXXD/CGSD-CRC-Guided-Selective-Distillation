from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from mias_dcms.preference_acquisition_audit import audit_preference_acquisition
from mias_dcms.selectors import assert_selector_rows_are_label_safe, select_top_budget


def audit_length_gamma_intervention(
    rows: Iterable[Mapping[str, Any]],
    *,
    gammas: Sequence[float],
    budget: int,
    target_length_bin: str,
    id_field: str = "sample_id",
    base_margin_field: str = "base_margin",
    length_gap_field: str = "length_gap",
    length_bin_field: str = "length_gap_bin",
    linked_group_fields: Sequence[str] = (),
) -> dict[str, Any]:
    source_rows = [dict(row) for row in rows]
    _validate_safe_rows(source_rows)
    gamma_values = [float(gamma) for gamma in gammas]
    if not gamma_values:
        raise ValueError("gammas must not be empty")

    sample_ids = [_row_id(row, id_field=id_field) for row in source_rows]
    base_scores = [float(row[base_margin_field]) for row in source_rows]
    length_gaps = [float(row[length_gap_field]) for row in source_rows]
    target = str(target_length_bin)
    target_pool_count = sum(str(row[length_bin_field]) == target for row in source_rows)

    points: list[dict[str, Any]] = []
    for gamma in gamma_values:
        scores = [
            base_score + gamma * length_gap
            for base_score, length_gap in zip(base_scores, length_gaps, strict=True)
        ]
        selected_ids = select_top_budget(sample_ids=sample_ids, scores=scores, budget=int(budget))
        selected_id_set = set(selected_ids)
        selected_rows = [row for row in source_rows if _row_id(row, id_field=id_field) in selected_id_set]
        target_selected_count = sum(str(row[length_bin_field]) == target for row in selected_rows)
        point = {
            "gamma": gamma,
            "budget": int(budget),
            "selected_ids": selected_ids,
            "target_length_bin": target,
            "target_length_bin_pool_count": target_pool_count,
            "target_length_bin_selected_count": target_selected_count,
            "target_length_bin_propensity": target_selected_count / target_pool_count
            if target_pool_count
            else 0.0,
            "length_bin_distribution": _distribution(
                [str(row[length_bin_field]) for row in selected_rows]
            ),
            "linked_group_distribution": {
                str(field): _distribution([str(row[field]) for row in selected_rows])
                for field in linked_group_fields
                if all(field in row for row in selected_rows)
            },
        }
        points.append(point)

    gamma_zero_matches_base = True
    if 0.0 in gamma_values:
        zero_scores = [
            base_score + 0.0 * length_gap
            for base_score, length_gap in zip(base_scores, length_gaps, strict=True)
        ]
        gamma_zero_matches_base = zero_scores == base_scores

    return {
        "mode": "length_gamma",
        "pool_size": len(source_rows),
        "budget": int(budget),
        "gammas": gamma_values,
        "gamma_grid_has_negative_zero_positive": (
            any(gamma < 0.0 for gamma in gamma_values)
            and any(gamma == 0.0 for gamma in gamma_values)
            and any(gamma > 0.0 for gamma in gamma_values)
        ),
        "gamma_zero_matches_base_score": gamma_zero_matches_base,
        "target_length_bin": target,
        "points": points,
        "target_propensity_slope": _slope(
            gamma_values,
            [float(point["target_length_bin_propensity"]) for point in points],
        ),
    }


def audit_selector_replacement(
    rows: Iterable[Mapping[str, Any]],
    *,
    selector_a_score_field: str,
    selector_b_score_field: str,
    budget: int,
    group_fields: Sequence[str],
    id_field: str = "sample_id",
) -> dict[str, Any]:
    source_rows = [dict(row) for row in rows]
    _validate_safe_rows(source_rows)
    sample_ids = [_row_id(row, id_field=id_field) for row in source_rows]
    scores_a = [float(row[selector_a_score_field]) for row in source_rows]
    scores_b = [float(row[selector_b_score_field]) for row in source_rows]
    selected_a = select_top_budget(sample_ids=sample_ids, scores=scores_a, budget=int(budget))
    selected_b = select_top_budget(sample_ids=sample_ids, scores=scores_b, budget=int(budget))

    selected_a_set = set(selected_a)
    selected_b_set = set(selected_b)
    attribute_delta: dict[str, Any] = {}
    selector_a_propensities: dict[str, Any] = {}
    selector_b_propensities: dict[str, Any] = {}
    for field in group_fields:
        field_name = str(field)
        rows_a = [
            {**row, "selected": _row_id(row, id_field=id_field) in selected_a_set}
            for row in source_rows
        ]
        rows_b = [
            {**row, "selected": _row_id(row, id_field=id_field) in selected_b_set}
            for row in source_rows
        ]
        audit_a = audit_preference_acquisition(
            rows_a,
            method="selector_a",
            group_fields=(field_name,),
        )
        audit_b = audit_preference_acquisition(
            rows_b,
            method="selector_b",
            group_fields=(field_name,),
        )
        dist_a = _selected_distribution(rows_a, group_field=field_name)
        dist_b = _selected_distribution(rows_b, group_field=field_name)
        attribute_delta[field_name] = {
            "selector_a_distribution": dist_a,
            "selector_b_distribution": dist_b,
            "selected_distribution_tv": _distribution_tv(dist_a, dist_b),
        }
        selector_a_propensities[field_name] = audit_a["by_group_field"][field_name]["groups"]
        selector_b_propensities[field_name] = audit_b["by_group_field"][field_name]["groups"]

    return {
        "mode": "selector_replacement",
        "pool_size": len(source_rows),
        "budget": int(budget),
        "selector_a_score_field": selector_a_score_field,
        "selector_b_score_field": selector_b_score_field,
        "selector_a_selected_ids": selected_a,
        "selector_b_selected_ids": selected_b,
        "score_rank_correlation": _spearman_rank_correlation(scores_a, scores_b),
        "selected_set_overlap": len(selected_a_set & selected_b_set) / int(budget)
        if int(budget)
        else 0.0,
        "attribute_coverage_delta": attribute_delta,
        "selector_a_group_propensities": selector_a_propensities,
        "selector_b_group_propensities": selector_b_propensities,
    }


def audit_ab_position_intervention(
    rows: Iterable[Mapping[str, Any]],
    *,
    score_field: str,
    budget: int,
    id_field: str = "sample_id",
    pair_field: str = "swap_pair_id",
    position_field: str = "ab_position",
) -> dict[str, Any]:
    source_rows = [dict(row) for row in rows]
    _validate_safe_rows(source_rows)
    sample_ids = [_row_id(row, id_field=id_field) for row in source_rows]
    scores = [float(row[score_field]) for row in source_rows]
    selected_ids = select_top_budget(sample_ids=sample_ids, scores=scores, budget=int(budget))
    selected_id_set = set(selected_ids)
    selected_rows = [row for row in source_rows if _row_id(row, id_field=id_field) in selected_id_set]

    pair_scores: dict[str, dict[str, float]] = defaultdict(dict)
    for row in source_rows:
        pair = str(row[pair_field])
        position = str(row[position_field]).strip().lower()
        if position in {"original", "swapped"}:
            pair_scores[pair][position] = float(row[score_field])
    paired = [
        values
        for values in pair_scores.values()
        if "original" in values and "swapped" in values
    ]
    original_scores = [values["original"] for values in paired]
    swapped_scores = [values["swapped"] for values in paired]

    position_pool_counts: dict[str, int] = defaultdict(int)
    position_selected_counts: dict[str, int] = defaultdict(int)
    for row in source_rows:
        position_pool_counts[str(row[position_field])] += 1
    for row in selected_rows:
        position_selected_counts[str(row[position_field])] += 1
    position_propensity = {
        position: position_selected_counts[position] / count if count else 0.0
        for position, count in sorted(position_pool_counts.items())
    }

    pool_distribution = {
        position: count / len(source_rows)
        for position, count in sorted(position_pool_counts.items())
    }
    selected_distribution = _distribution([str(row[position_field]) for row in selected_rows])

    return {
        "mode": "ab_position",
        "pool_size": len(source_rows),
        "budget": int(budget),
        "pair_count": len(paired),
        "selected_ids": selected_ids,
        "original_swapped_rank_correlation": _spearman_rank_correlation(
            original_scores,
            swapped_scores,
        ),
        "position_propensity": position_propensity,
        "position_acquisition_tv": _distribution_tv(pool_distribution, selected_distribution),
    }


def _validate_safe_rows(rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("rows must not be empty")
    assert_selector_rows_are_label_safe(rows)


def _row_id(row: Mapping[str, Any], *, id_field: str) -> str:
    value = row.get(id_field, row.get("id"))
    if value is None:
        raise ValueError(f"row is missing id field {id_field!r} and fallback 'id'")
    return str(value)


def _distribution(values: Sequence[str]) -> dict[str, float]:
    if not values:
        return {}
    counts: dict[str, int] = defaultdict(int)
    for value in values:
        counts[str(value)] += 1
    return {key: counts[key] / len(values) for key in sorted(counts)}


def _selected_distribution(rows: list[dict[str, Any]], *, group_field: str) -> dict[str, float]:
    return _distribution([str(row[group_field]) for row in rows if bool(row["selected"])])


def _distribution_tv(left: Mapping[str, float], right: Mapping[str, float]) -> float:
    keys = set(left) | set(right)
    return 0.5 * sum(abs(float(left.get(key, 0.0)) - float(right.get(key, 0.0))) for key in keys)


def _slope(xs: Sequence[float], ys: Sequence[float]) -> float:
    if len(xs) != len(ys):
        raise ValueError("slope inputs must have equal length")
    if len(xs) < 2:
        return 0.0
    x_mean = sum(xs) / len(xs)
    y_mean = sum(ys) / len(ys)
    denominator = sum((x - x_mean) ** 2 for x in xs)
    if denominator == 0.0:
        return 0.0
    numerator = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys, strict=True))
    return numerator / denominator


def _spearman_rank_correlation(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right):
        raise ValueError("rank correlation inputs must have equal length")
    if len(left) < 2:
        return 0.0
    return _pearson(_ranks(left), _ranks(right))


def _ranks(values: Sequence[float]) -> list[float]:
    indexed = sorted(enumerate(float(value) for value in values), key=lambda item: (item[1], item[0]))
    ranks = [0.0] * len(indexed)
    index = 0
    while index < len(indexed):
        end = index + 1
        while end < len(indexed) and indexed[end][1] == indexed[index][1]:
            end += 1
        average_rank = (index + end - 1) / 2.0
        for sorted_index in range(index, end):
            original_index = indexed[sorted_index][0]
            ranks[original_index] = average_rank
        index = end
    return ranks


def _pearson(left: Sequence[float], right: Sequence[float]) -> float:
    left_mean = sum(left) / len(left)
    right_mean = sum(right) / len(right)
    numerator = sum((x - left_mean) * (y - right_mean) for x, y in zip(left, right, strict=True))
    left_denominator = sum((x - left_mean) ** 2 for x in left) ** 0.5
    right_denominator = sum((y - right_mean) ** 2 for y in right) ** 0.5
    denominator = left_denominator * right_denominator
    if denominator == 0.0:
        return 0.0
    return numerator / denominator
