from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any, Iterable


@dataclass(frozen=True)
class GroupPropensity:
    group: str
    pool_count: int
    selected_count: int
    pool_share: float
    propensity: float
    predicted_selected_share: float
    actual_selected_share: float
    prediction_error: float


@dataclass(frozen=True)
class PropensityIdentityReport:
    pool_size: int
    selected_size: int
    groups: dict[str, GroupPropensity]
    total_absolute_prediction_error: float


@dataclass(frozen=True)
class MIASSelectionAudit:
    pool_size: int
    selected_size: int
    acquisition_tv: float
    maximum_propensity_ratio: float
    total_absolute_prediction_error: float
    groups: dict[str, GroupPropensity]

    def as_dict(self) -> dict[str, Any]:
        return {
            "pool_size": self.pool_size,
            "selected_size": self.selected_size,
            "acquisition_tv": self.acquisition_tv,
            "maximum_propensity_ratio": self.maximum_propensity_ratio,
            "total_absolute_prediction_error": self.total_absolute_prediction_error,
            "groups": {
                group: {
                    "group": report.group,
                    "pool_count": report.pool_count,
                    "selected_count": report.selected_count,
                    "pool_share": report.pool_share,
                    "propensity": report.propensity,
                    "predicted_selected_share": report.predicted_selected_share,
                    "actual_selected_share": report.actual_selected_share,
                    "prediction_error": report.prediction_error,
                }
                for group, report in self.groups.items()
            },
        }


def propensity_identity_report(
    rows: Iterable[dict[str, Any]],
    *,
    group_field: str,
    selected_field: str,
) -> PropensityIdentityReport:
    pool = [dict(row) for row in rows]
    if not pool:
        raise ValueError("rows must not be empty")
    pool_counts: Counter[str] = Counter()
    selected_counts: Counter[str] = Counter()
    for row in pool:
        group = str(row[group_field])
        pool_counts[group] += 1
        if bool(row[selected_field]):
            selected_counts[group] += 1

    selected_size = sum(selected_counts.values())
    if selected_size == 0:
        raise ValueError("at least one row must be selected")

    denominator = sum(
        (pool_counts[group] / len(pool)) * (selected_counts[group] / pool_counts[group])
        for group in pool_counts
    )
    group_reports: dict[str, GroupPropensity] = {}
    total_error = 0.0
    for group in sorted(pool_counts):
        pool_count = pool_counts[group]
        selected_count = selected_counts[group]
        pool_share = pool_count / len(pool)
        propensity = selected_count / pool_count if pool_count else 0.0
        predicted = (pool_share * propensity / denominator) if denominator > 0.0 else 0.0
        actual = selected_count / selected_size
        error = abs(predicted - actual)
        total_error += error
        group_reports[group] = GroupPropensity(
            group=group,
            pool_count=pool_count,
            selected_count=selected_count,
            pool_share=pool_share,
            propensity=propensity,
            predicted_selected_share=predicted,
            actual_selected_share=actual,
            prediction_error=error,
        )
    return PropensityIdentityReport(
        pool_size=len(pool),
        selected_size=selected_size,
        groups=group_reports,
        total_absolute_prediction_error=total_error,
    )


def acquisition_tv(
    rows: Iterable[dict[str, Any]],
    *,
    group_field: str,
    selected_field: str,
) -> float:
    report = propensity_identity_report(rows, group_field=group_field, selected_field=selected_field)
    return 0.5 * sum(abs(group.pool_share - group.actual_selected_share) for group in report.groups.values())


def maximum_propensity_ratio(
    rows: Iterable[dict[str, Any]],
    *,
    group_field: str,
    selected_field: str,
) -> float:
    report = propensity_identity_report(rows, group_field=group_field, selected_field=selected_field)
    positive_propensities = [group.propensity for group in report.groups.values() if group.propensity > 0.0]
    if not positive_propensities:
        return 0.0
    return max(positive_propensities) / min(positive_propensities)


def mias_selection_audit(
    rows: Iterable[dict[str, Any]],
    *,
    group_field: str,
    selected_field: str,
) -> MIASSelectionAudit:
    pool = [dict(row) for row in rows]
    report = propensity_identity_report(pool, group_field=group_field, selected_field=selected_field)
    return MIASSelectionAudit(
        pool_size=report.pool_size,
        selected_size=report.selected_size,
        acquisition_tv=0.5
        * sum(abs(group.pool_share - group.actual_selected_share) for group in report.groups.values()),
        maximum_propensity_ratio=maximum_propensity_ratio(
            pool,
            group_field=group_field,
            selected_field=selected_field,
        ),
        total_absolute_prediction_error=report.total_absolute_prediction_error,
        groups=report.groups,
    )
