from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping, Sequence

from mias_dcms.selectors import assert_selector_rows_are_label_safe


@dataclass(frozen=True)
class SoftGroupIntervalRow:
    sample_id: str
    group_membership: dict[str, float]
    membership_lower: dict[str, float]
    membership_upper: dict[str, float]
    draw_count: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "group_membership": dict(self.group_membership),
            "membership_lower": dict(self.membership_lower),
            "membership_upper": dict(self.membership_upper),
            "draw_count": self.draw_count,
        }


@dataclass(frozen=True)
class SoftGroupIntervalReport:
    groups: list[str]
    sample_count: int
    confidence: float
    rows: list[SoftGroupIntervalRow]

    def as_dict(self) -> dict[str, Any]:
        return {
            "groups": list(self.groups),
            "sample_count": self.sample_count,
            "confidence": self.confidence,
            "rows": [row.as_dict() for row in self.rows],
        }


@dataclass(frozen=True)
class SoftGroupCalibrationReport:
    groups: list[str]
    sample_count: int
    per_group: dict[str, dict[str, float]]
    overall_brier_score: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "groups": list(self.groups),
            "sample_count": self.sample_count,
            "per_group": {group: dict(values) for group, values in self.per_group.items()},
            "overall_brier_score": self.overall_brier_score,
        }


@dataclass(frozen=True)
class IntervalCoverageReport:
    groups: list[str]
    sample_count: int
    per_group: dict[str, dict[str, float]]
    overall_coverage_rate: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "groups": list(self.groups),
            "sample_count": self.sample_count,
            "per_group": {group: dict(values) for group, values in self.per_group.items()},
            "overall_coverage_rate": self.overall_coverage_rate,
        }


def build_soft_group_intervals(
    *,
    sample_ids: Sequence[str],
    membership_draws: Sequence[Sequence[Mapping[str, float]]],
    confidence: float = 0.95,
) -> SoftGroupIntervalReport:
    ids = [str(sample_id) for sample_id in sample_ids]
    if len(ids) != len(membership_draws):
        raise ValueError("sample_ids and membership_draws must have equal length")
    if len(set(ids)) != len(ids):
        raise ValueError("sample_ids must be unique")
    if not 0.0 < float(confidence) <= 1.0:
        raise ValueError("confidence must be in (0, 1]")

    groups = _all_groups(membership_draws)
    rows: list[SoftGroupIntervalRow] = []
    lower_probability = (1.0 - float(confidence)) / 2.0
    upper_probability = 1.0 - lower_probability
    for sample_id, draws in zip(ids, membership_draws):
        if not draws:
            raise ValueError(f"sample {sample_id!r} has no membership draws")
        values_by_group = {
            group: [_validated_membership(draw.get(group, 0.0), sample_id=sample_id, group=group) for draw in draws]
            for group in groups
        }
        rows.append(
            SoftGroupIntervalRow(
                sample_id=sample_id,
                group_membership={
                    group: _mean(values)
                    for group, values in values_by_group.items()
                },
                membership_lower={
                    group: _quantile(values, lower_probability)
                    for group, values in values_by_group.items()
                },
                membership_upper={
                    group: _quantile(values, upper_probability)
                    for group, values in values_by_group.items()
                },
                draw_count=len(draws),
            )
        )
    return SoftGroupIntervalReport(
        groups=groups,
        sample_count=len(rows),
        confidence=float(confidence),
        rows=rows,
    )


def soft_group_calibration_report(
    *,
    predicted_memberships: Sequence[Mapping[str, float]],
    observed_memberships: Sequence[Mapping[str, float]],
) -> SoftGroupCalibrationReport:
    if len(predicted_memberships) != len(observed_memberships):
        raise ValueError("predicted_memberships and observed_memberships must have equal length")
    if not predicted_memberships:
        raise ValueError("memberships must not be empty")

    groups = _groups_from_memberships(predicted_memberships, observed_memberships)
    per_group: dict[str, dict[str, float]] = {}
    all_squared_errors: list[float] = []
    for group in groups:
        predicted_values = [
            _validated_membership(row.get(group, 0.0), sample_id=str(index), group=group)
            for index, row in enumerate(predicted_memberships)
        ]
        observed_values = [
            _validated_membership(row.get(group, 0.0), sample_id=str(index), group=group)
            for index, row in enumerate(observed_memberships)
        ]
        squared_errors = [
            (predicted - observed) ** 2
            for predicted, observed in zip(predicted_values, observed_values)
        ]
        all_squared_errors.extend(squared_errors)
        per_group[group] = {
            "mean_predicted": _mean(predicted_values),
            "mean_observed": _mean(observed_values),
            "mean_error": _mean([predicted - observed for predicted, observed in zip(predicted_values, observed_values)]),
            "brier_score": _mean(squared_errors),
        }
    return SoftGroupCalibrationReport(
        groups=groups,
        sample_count=len(predicted_memberships),
        per_group=per_group,
        overall_brier_score=_mean(all_squared_errors),
    )


def interval_coverage_report(
    interval_rows: Sequence[SoftGroupIntervalRow],
    *,
    observed_memberships: Sequence[Mapping[str, float]],
) -> IntervalCoverageReport:
    if len(interval_rows) != len(observed_memberships):
        raise ValueError("interval_rows and observed_memberships must have equal length")
    if not interval_rows:
        raise ValueError("interval_rows must not be empty")

    groups = _groups_from_interval_rows(interval_rows, observed_memberships)
    per_group: dict[str, dict[str, float]] = {}
    all_covered: list[float] = []
    for group in groups:
        covered: list[float] = []
        for index, (row, observed) in enumerate(zip(interval_rows, observed_memberships)):
            lower = _validated_membership(row.membership_lower.get(group, 0.0), sample_id=row.sample_id, group=group)
            upper = _validated_membership(row.membership_upper.get(group, 0.0), sample_id=row.sample_id, group=group)
            value = _validated_membership(observed.get(group, 0.0), sample_id=str(index), group=group)
            if lower > upper:
                raise ValueError(f"sample {row.sample_id!r} group {group!r} lower bound exceeds upper bound")
            covered.append(1.0 if lower - 1e-12 <= value <= upper + 1e-12 else 0.0)
        all_covered.extend(covered)
        per_group[group] = {
            "coverage_rate": _mean(covered),
            "covered_count": float(sum(covered)),
            "total_count": float(len(covered)),
        }
    return IntervalCoverageReport(
        groups=groups,
        sample_count=len(interval_rows),
        per_group=per_group,
        overall_coverage_rate=_mean(all_covered),
    )


def build_soft_group_intervals_from_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    id_field: str = "sample_id",
    draws_field: str = "ensemble_memberships",
    confidence: float = 0.95,
) -> SoftGroupIntervalReport:
    row_dicts = [dict(row) for row in rows]
    assert_selector_rows_are_label_safe(row_dicts)
    sample_ids: list[str] = []
    membership_draws: list[list[dict[str, float]]] = []
    for index, row in enumerate(row_dicts):
        sample_id = row.get(id_field)
        if sample_id is None:
            raise ValueError(f"row {index} missing id field {id_field!r}")
        raw_draws = row.get(draws_field)
        if not isinstance(raw_draws, list):
            raise ValueError(f"row {sample_id!r} field {draws_field!r} must contain a list")
        sample_ids.append(str(sample_id))
        membership_draws.append([_coerce_membership_draw(draw, sample_id=str(sample_id)) for draw in raw_draws])
    return build_soft_group_intervals(
        sample_ids=sample_ids,
        membership_draws=membership_draws,
        confidence=confidence,
    )


def _coerce_membership_draw(draw: Any, *, sample_id: str) -> dict[str, float]:
    if not isinstance(draw, dict):
        raise ValueError(f"sample {sample_id!r} membership draw must be an object")
    return {str(group): float(value) for group, value in draw.items()}


def _all_groups(membership_draws: Sequence[Sequence[Mapping[str, float]]]) -> list[str]:
    groups: set[str] = set()
    for draws in membership_draws:
        for draw in draws:
            groups.update(str(group) for group in draw)
    if not groups:
        raise ValueError("membership_draws must contain at least one group")
    return sorted(groups)


def _groups_from_memberships(
    first: Sequence[Mapping[str, float]],
    second: Sequence[Mapping[str, float]],
) -> list[str]:
    groups: set[str] = set()
    for collection in (first, second):
        for row in collection:
            groups.update(str(group) for group in row)
    if not groups:
        raise ValueError("memberships must contain at least one group")
    return sorted(groups)


def _groups_from_interval_rows(
    interval_rows: Sequence[SoftGroupIntervalRow],
    observed_memberships: Sequence[Mapping[str, float]],
) -> list[str]:
    groups: set[str] = set()
    for row in interval_rows:
        groups.update(row.group_membership)
        groups.update(row.membership_lower)
        groups.update(row.membership_upper)
    for row in observed_memberships:
        groups.update(str(group) for group in row)
    if not groups:
        raise ValueError("interval rows must contain at least one group")
    return sorted(groups)


def _validated_membership(value: float, *, sample_id: str, group: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"sample {sample_id!r} group {group!r} membership must be finite")
    if number < 0.0 or number > 1.0:
        raise ValueError(f"sample {sample_id!r} group {group!r} membership must be in [0, 1]")
    return number


def _mean(values: Sequence[float]) -> float:
    if not values:
        raise ValueError("values must not be empty")
    return sum(values) / len(values)


def _quantile(values: Sequence[float], probability: float) -> float:
    if not values:
        raise ValueError("values must not be empty")
    if not 0.0 <= probability <= 1.0:
        raise ValueError("probability must be between 0 and 1")
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    position = probability * (len(ordered) - 1)
    lower_index = int(position)
    upper_index = min(lower_index + 1, len(ordered) - 1)
    weight = position - lower_index
    return ordered[lower_index] * (1.0 - weight) + ordered[upper_index] * weight
