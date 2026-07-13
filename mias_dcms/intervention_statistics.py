from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
import random
from typing import Any, Iterable, Mapping, Sequence


@dataclass(frozen=True)
class InterventionStatisticsReport:
    completed_setting_count: int
    failed_setting_count: int
    by_setting: dict[str, dict[str, Any]]
    issues: list[dict[str, Any]] = field(default_factory=list)

    @property
    def is_ready(self) -> bool:
        return not self.issues

    def as_dict(self) -> dict[str, Any]:
        return {
            "is_ready": self.is_ready,
            "completed_setting_count": self.completed_setting_count,
            "failed_setting_count": self.failed_setting_count,
            "by_setting": {key: dict(value) for key, value in self.by_setting.items()},
            "issues": [dict(issue) for issue in self.issues],
        }


def audit_intervention_response_statistics(
    rows: Iterable[Mapping[str, Any]],
    *,
    expected_settings: Sequence[str],
    minimum_values: int = 5,
    setting_field: str = "setting",
    status_field: str = "status",
    failure_reason_field: str = "failure_reason",
    intervention_value_field: str = "intervention_value",
    response_field: str = "target_group_propensity",
    confidence: float = 0.95,
    resamples: int = 1000,
    seed: int = 0,
) -> InterventionStatisticsReport:
    source_rows = [dict(row) for row in rows]
    issues: list[dict[str, Any]] = []
    rows_by_setting: dict[str, list[dict[str, Any]]] = defaultdict(list)
    failed_settings: set[str] = set()

    for row_index, row in enumerate(source_rows):
        setting = str(row.get(setting_field, ""))
        if not setting:
            issues.append({"code": "missing_setting", "row_index": row_index})
            continue
        rows_by_setting[setting].append(row)
        if str(row.get(status_field, "completed")) == "failed":
            failed_settings.add(setting)
            issue = {"code": "failed_setting", "setting": setting, "row_index": row_index}
            reason = str(row.get(failure_reason_field, "") or "").strip()
            if reason:
                issue["failure_reason"] = reason
            issues.append(issue)
            if not reason:
                issues.append(
                    {
                        "code": "failed_setting_missing_reason",
                        "setting": setting,
                        "row_index": row_index,
                    }
                )

    for expected_setting in expected_settings:
        if str(expected_setting) not in rows_by_setting:
            issues.append({"code": "missing_expected_setting", "setting": str(expected_setting)})

    by_setting: dict[str, dict[str, Any]] = {}
    for setting, setting_rows in sorted(rows_by_setting.items()):
        completed_rows = [
            row
            for row in setting_rows
            if str(row.get(status_field, "completed")) == "completed"
        ]
        if not completed_rows:
            continue
        values: list[float] = []
        responses: list[float] = []
        for row_index, row in enumerate(completed_rows):
            try:
                values.append(float(row[intervention_value_field]))
                responses.append(float(row[response_field]))
            except (KeyError, TypeError, ValueError):
                issues.append(
                    {
                        "code": "invalid_response_curve_point",
                        "setting": setting,
                        "row_index": row_index,
                    }
                )

        if len(values) != len(responses) or not values:
            continue
        unique_value_count = len(set(values))
        if unique_value_count < int(minimum_values):
            issues.append(
                {
                    "code": "insufficient_intervention_values",
                    "setting": setting,
                    "intervention_value_count": unique_value_count,
                    "minimum_values": int(minimum_values),
                }
            )

        slope = _slope(values, responses)
        ci_low, ci_high = _bootstrap_slope_ci(
            values,
            responses,
            confidence=confidence,
            resamples=resamples,
            seed=seed,
        )
        by_setting[setting] = {
            "setting": setting,
            "completed_point_count": len(values),
            "intervention_value_count": unique_value_count,
            "intervention_value_min": min(values),
            "intervention_value_max": max(values),
            "response_min": min(responses),
            "response_max": max(responses),
            "spearman_monotonicity": _spearman_rank_correlation(values, responses),
            "slope": slope,
            "slope_ci_low": ci_low,
            "slope_ci_high": ci_high,
            "confidence": float(confidence),
            "resamples": int(resamples),
        }

    return InterventionStatisticsReport(
        completed_setting_count=len(by_setting),
        failed_setting_count=len(failed_settings),
        by_setting=by_setting,
        issues=issues,
    )


def _bootstrap_slope_ci(
    xs: Sequence[float],
    ys: Sequence[float],
    *,
    confidence: float,
    resamples: int,
    seed: int,
) -> tuple[float, float]:
    if len(xs) != len(ys):
        raise ValueError("xs and ys must have equal length")
    if len(xs) < 2:
        return (0.0, 0.0)
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be between 0 and 1")
    if resamples <= 0:
        raise ValueError("resamples must be positive")
    rng = random.Random(seed)
    slopes = []
    for _ in range(int(resamples)):
        indexes = [rng.randrange(len(xs)) for _ in xs]
        slopes.append(_slope([xs[index] for index in indexes], [ys[index] for index in indexes]))
    alpha = 1.0 - confidence
    return (_quantile(slopes, alpha / 2.0), _quantile(slopes, 1.0 - alpha / 2.0))


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
    return sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys, strict=True)) / denominator


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
            ranks[indexed[sorted_index][0]] = average_rank
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


def _quantile(values: Sequence[float], probability: float) -> float:
    if not values:
        raise ValueError("values must not be empty")
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    position = probability * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight
