from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping, Sequence


@dataclass(frozen=True)
class ResponseCurvePoint:
    value: float
    budget: int
    selected_ids: list[str]
    target_group: str
    target_group_pool_count: int
    target_group_selected_count: int
    target_group_propensity: float
    selected_group_distribution: dict[str, float]


@dataclass(frozen=True)
class ResponseCurve:
    points: list[ResponseCurvePoint]

    def as_dict(self) -> dict[str, object]:
        return {
            "points": [
                {
                    "value": point.value,
                    "budget": point.budget,
                    "selected_ids": list(point.selected_ids),
                    "target_group": point.target_group,
                    "target_group_pool_count": point.target_group_pool_count,
                    "target_group_selected_count": point.target_group_selected_count,
                    "target_group_propensity": point.target_group_propensity,
                    "selected_group_distribution": dict(point.selected_group_distribution),
                }
                for point in self.points
            ]
        }


def apply_class_intercept(
    logits: Sequence[Sequence[float]],
    *,
    target_class: int,
    alpha: float,
) -> list[list[float]]:
    shifted: list[list[float]] = []
    for row in logits:
        values = [float(value) for value in row]
        if target_class < 0 or target_class >= len(values):
            raise ValueError("target_class is out of range for logits row")
        values[target_class] += float(alpha)
        shifted.append(values)
    return shifted


def entropy_scores_from_logits(logits: Sequence[Sequence[float]]) -> list[float]:
    return [_entropy_from_row(row) for row in logits]


def normalized_length_gap(*, response_a_length: int, response_b_length: int) -> float:
    total = int(response_a_length) + int(response_b_length)
    if total <= 0:
        return 0.0
    return (int(response_a_length) - int(response_b_length)) / total


def apply_length_coefficient(
    base_margins: Sequence[float],
    normalized_gaps: Sequence[float],
    *,
    gamma: float,
) -> list[float]:
    if len(base_margins) != len(normalized_gaps):
        raise ValueError("base_margins and normalized_gaps must have equal length")
    return [float(margin) + float(gamma) * float(gap) for margin, gap in zip(base_margins, normalized_gaps)]


def fixed_budget_response_curve(
    *,
    sample_ids: Sequence[str],
    groups: Sequence[str],
    score_by_value: Mapping[float, Sequence[float]],
    budget: int,
    target_group: str,
) -> ResponseCurve:
    if len(sample_ids) != len(groups):
        raise ValueError("sample_ids and groups must have equal length")
    if budget < 0 or budget > len(sample_ids):
        raise ValueError("budget must be between 0 and sample count")
    ids = [str(sample_id) for sample_id in sample_ids]
    group_values = [str(group) for group in groups]
    target = str(target_group)
    target_pool_count = sum(group == target for group in group_values)

    points: list[ResponseCurvePoint] = []
    for value in sorted(score_by_value):
        scores = [float(score) for score in score_by_value[value]]
        if len(scores) != len(ids):
            raise ValueError("each score list must match sample count")
        selected_indexes = sorted(
            range(len(ids)),
            key=lambda index: (-scores[index], ids[index]),
        )[:budget]
        selected_ids = [ids[index] for index in selected_indexes]
        selected_groups = [group_values[index] for index in selected_indexes]
        target_selected_count = sum(group == target for group in selected_groups)
        distribution = _distribution(selected_groups)
        propensity = target_selected_count / target_pool_count if target_pool_count else 0.0
        points.append(
            ResponseCurvePoint(
                value=float(value),
                budget=int(budget),
                selected_ids=selected_ids,
                target_group=target,
                target_group_pool_count=target_pool_count,
                target_group_selected_count=target_selected_count,
                target_group_propensity=propensity,
                selected_group_distribution=distribution,
            )
        )
    return ResponseCurve(points=points)


def _entropy_from_row(row: Sequence[float]) -> float:
    values = [float(value) for value in row]
    if not values:
        raise ValueError("logits rows must not be empty")
    max_value = max(values)
    exp_values = [math.exp(value - max_value) for value in values]
    total = sum(exp_values)
    probabilities = [value / total for value in exp_values]
    return -sum(probability * math.log(probability) for probability in probabilities if probability > 0.0)


def _distribution(groups: Sequence[str]) -> dict[str, float]:
    if not groups:
        return {}
    counts: dict[str, int] = {}
    for group in groups:
        counts[group] = counts.get(group, 0) + 1
    return {group: count / len(groups) for group, count in sorted(counts.items())}
