from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping
from typing import Any


def preference_accuracy(
    rows: Iterable[Mapping[str, Any]],
    *,
    label_field: str = "oracle_preference",
    prediction_field: str = "predicted_preference",
) -> float:
    row_list = list(rows)
    if not row_list:
        raise ValueError("preference rows must not be empty")
    correct = sum(
        1
        for row in row_list
        if _normalized_value(row, label_field) == _normalized_value(row, prediction_field)
    )
    return correct / len(row_list)


def worst_group_preference_accuracy(
    rows: Iterable[Mapping[str, Any]],
    *,
    group_field: str,
    label_field: str = "oracle_preference",
    prediction_field: str = "predicted_preference",
) -> float:
    row_list = list(rows)
    if not row_list:
        raise ValueError("preference rows must not be empty")
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in row_list:
        grouped[_normalized_value(row, group_field)].append(row)
    return min(
        preference_accuracy(
            group_rows,
            label_field=label_field,
            prediction_field=prediction_field,
        )
        for group_rows in grouped.values()
    )


def length_controlled_win_rate(
    rows: Iterable[Mapping[str, Any]],
    *,
    win_field: str = "judge_win",
    length_bin_field: str = "length_gap_bin",
) -> float:
    row_list = list(rows)
    if not row_list:
        raise ValueError("judge rows must not be empty")
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in row_list:
        grouped[_normalized_value(row, length_bin_field)].append(float(row[win_field]))
    return _mean([_mean(values) for values in grouped.values()])


def raw_judge_win_rate(
    rows: Iterable[Mapping[str, Any]],
    *,
    win_field: str = "judge_win",
) -> float:
    row_list = list(rows)
    if not row_list:
        raise ValueError("judge rows must not be empty")
    return _mean([float(row[win_field]) for row in row_list])


def capability_regression(
    rows: Iterable[Mapping[str, Any]],
    *,
    baseline_field: str = "baseline_score",
    policy_field: str = "policy_score",
) -> float:
    row_list = list(rows)
    if not row_list:
        raise ValueError("capability rows must not be empty")
    return _mean([float(row[baseline_field]) - float(row[policy_field]) for row in row_list])


def area_under_learning_curve(
    rows: Iterable[Mapping[str, Any]],
    *,
    x_field: str = "budget",
    y_field: str = "performance",
) -> float:
    """Return a normalized trapezoidal area for a budget/performance curve.

    AULC is defined over the observed budget range. With one point, the point's
    performance is returned; with repeated budgets, the best value at each
    budget is used only after requiring a deterministic ordering.
    """
    row_list = list(rows)
    if not row_list:
        raise ValueError("AULC rows must not be empty")
    points = sorted(
        (float(row[x_field]), float(row[y_field])) for row in row_list
    )
    collapsed: dict[float, float] = {}
    for budget, performance in points:
        collapsed[budget] = max(float(performance), collapsed.get(budget, float("-inf")))
    curve = sorted(collapsed.items())
    if len(curve) == 1:
        return curve[0][1]
    x_values = [point[0] for point in curve]
    y_values = [point[1] for point in curve]
    width = x_values[-1] - x_values[0]
    if width <= 0.0:
        return _mean(y_values)
    area = sum(
        (x_values[index] - x_values[index - 1])
        * (y_values[index] + y_values[index - 1])
        / 2.0
        for index in range(1, len(curve))
    )
    return area / width


def build_preference_evaluation_metrics(
    *,
    preference_rows: Iterable[Mapping[str, Any]] | None = None,
    judge_rows: Iterable[Mapping[str, Any]] | None = None,
    capability_rows: Iterable[Mapping[str, Any]] | None = None,
    aulc_rows: Iterable[Mapping[str, Any]] | None = None,
    group_field: str = "observable_group",
    length_bin_field: str = "length_gap_bin",
    label_field: str = "oracle_preference",
    prediction_field: str = "predicted_preference",
    win_field: str = "judge_win",
    baseline_field: str = "baseline_score",
    policy_field: str = "policy_score",
    aulc_x_field: str = "budget",
    aulc_y_field: str = "performance",
) -> dict[str, float | int]:
    metrics: dict[str, float | int] = {}

    if preference_rows is not None:
        preference_list = list(preference_rows)
        metrics["preference_accuracy"] = preference_accuracy(
            preference_list,
            label_field=label_field,
            prediction_field=prediction_field,
        )
        metrics["worst_group_preference_accuracy"] = worst_group_preference_accuracy(
            preference_list,
            group_field=group_field,
            label_field=label_field,
            prediction_field=prediction_field,
        )
        metrics["preference_eval_count"] = len(preference_list)

    if judge_rows is not None:
        judge_list = list(judge_rows)
        metrics["raw_judge_win_rate"] = raw_judge_win_rate(
            judge_list,
            win_field=win_field,
        )
        metrics["length_controlled_win_rate"] = length_controlled_win_rate(
            judge_list,
            win_field=win_field,
            length_bin_field=length_bin_field,
        )
        metrics["judge_eval_count"] = len(judge_list)

    if capability_rows is not None:
        capability_list = list(capability_rows)
        metrics["capability_regression"] = capability_regression(
            capability_list,
            baseline_field=baseline_field,
            policy_field=policy_field,
        )
        metrics["capability_eval_count"] = len(capability_list)

    if aulc_rows is not None:
        aulc_list = list(aulc_rows)
        metrics["aulc"] = area_under_learning_curve(
            aulc_list,
            x_field=aulc_x_field,
            y_field=aulc_y_field,
        )
        metrics["aulc_point_count"] = len(aulc_list)

    if not metrics:
        raise ValueError("at least one evaluation input must be provided")
    return metrics


def _normalized_value(row: Mapping[str, Any], field: str) -> str:
    if field not in row or row[field] is None:
        raise ValueError(f"row is missing required field {field!r}")
    return str(row[field])


def _mean(values: list[float]) -> float:
    if not values:
        raise ValueError("values must not be empty")
    return sum(values) / len(values)
