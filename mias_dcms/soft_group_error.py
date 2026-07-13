from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from mias_dcms.selection.dcms import DCMSResult, solve_dcms


@dataclass(frozen=True)
class SoftGroupErrorSelectionAudit:
    selected_ids: list[str]
    estimated_moments: dict[str, float]
    observed_moments: dict[str, float]
    observed_max_constraint_violation: float
    utility_retained: float
    solver_status: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "selected_ids": list(self.selected_ids),
            "estimated_moments": dict(self.estimated_moments),
            "observed_moments": dict(self.observed_moments),
            "observed_max_constraint_violation": self.observed_max_constraint_violation,
            "utility_retained": self.utility_retained,
            "solver_status": self.solver_status,
        }


@dataclass(frozen=True)
class SoftGroupErrorAudit:
    nominal: SoftGroupErrorSelectionAudit
    robust: SoftGroupErrorSelectionAudit
    observed_violation_delta: float
    robust_improves_observed_coverage: bool

    def as_dict(self) -> dict[str, Any]:
        return {
            "nominal": self.nominal.as_dict(),
            "robust": self.robust.as_dict(),
            "observed_violation_delta": self.observed_violation_delta,
            "robust_improves_observed_coverage": self.robust_improves_observed_coverage,
        }


def soft_group_error_audit(
    *,
    sample_ids: Sequence[str],
    utilities: Sequence[float],
    group_membership: Sequence[Mapping[str, float]],
    membership_lower: Sequence[Mapping[str, float]],
    membership_upper: Sequence[Mapping[str, float]],
    observed_membership: Sequence[Mapping[str, float]],
    budget: int,
    target_moments: Mapping[str, float],
    tolerance: float,
    rounding_seed: int | None = None,
) -> SoftGroupErrorAudit:
    _validate_equal_lengths(
        sample_ids,
        utilities,
        group_membership,
        membership_lower,
        membership_upper,
        observed_membership,
    )
    nominal_result = solve_dcms(
        sample_ids=sample_ids,
        utilities=utilities,
        group_membership=group_membership,
        budget=budget,
        target_moments=target_moments,
        tolerance=tolerance,
        rounding_seed=rounding_seed,
    )
    robust_result = solve_dcms(
        sample_ids=sample_ids,
        utilities=utilities,
        group_membership=group_membership,
        membership_lower=membership_lower,
        membership_upper=membership_upper,
        budget=budget,
        target_moments=target_moments,
        tolerance=tolerance,
        rounding_seed=rounding_seed,
    )
    ids = [str(sample_id) for sample_id in sample_ids]
    nominal = _selection_audit(
        result=nominal_result,
        sample_ids=ids,
        observed_membership=observed_membership,
        target_moments=target_moments,
    )
    robust = _selection_audit(
        result=robust_result,
        sample_ids=ids,
        observed_membership=observed_membership,
        target_moments=target_moments,
    )
    delta = nominal.observed_max_constraint_violation - robust.observed_max_constraint_violation
    return SoftGroupErrorAudit(
        nominal=nominal,
        robust=robust,
        observed_violation_delta=delta,
        robust_improves_observed_coverage=delta > 1e-12,
    )


def _selection_audit(
    *,
    result: DCMSResult,
    sample_ids: Sequence[str],
    observed_membership: Sequence[Mapping[str, float]],
    target_moments: Mapping[str, float],
) -> SoftGroupErrorSelectionAudit:
    selected_indexes = [
        index
        for index, sample_id in enumerate(sample_ids)
        if int(result.selection_indicator.get(str(sample_id), 0)) == 1
    ]
    observed_moments = _moments(selected_indexes, observed_membership, _groups(target_moments, observed_membership))
    targets = {
        group: float(target_moments.get(group, 0.0))
        for group in sorted(set(observed_moments) | {str(group) for group in target_moments})
    }
    return SoftGroupErrorSelectionAudit(
        selected_ids=list(result.selected_ids),
        estimated_moments=dict(result.rounded_moments),
        observed_moments=observed_moments,
        observed_max_constraint_violation=_max_violation(observed_moments, targets),
        utility_retained=float(result.utility_retained),
        solver_status=str(result.solver_status),
    )


def _validate_equal_lengths(*sequences: Sequence[object]) -> None:
    lengths = {len(sequence) for sequence in sequences}
    if len(lengths) != 1:
        raise ValueError("all input sequences must have equal length")


def _groups(
    target_moments: Mapping[str, float],
    membership: Sequence[Mapping[str, float]],
) -> set[str]:
    groups = {str(group) for group in target_moments}
    for row in membership:
        groups.update(str(group) for group in row)
    return groups


def _moments(
    selected_indexes: Sequence[int],
    membership: Sequence[Mapping[str, float]],
    groups: set[str],
) -> dict[str, float]:
    if not selected_indexes:
        return {group: 0.0 for group in sorted(groups)}
    return {
        group: sum(float(membership[index].get(group, 0.0)) for index in selected_indexes) / len(selected_indexes)
        for group in sorted(groups)
    }


def _max_violation(moments: Mapping[str, float], targets: Mapping[str, float]) -> float:
    return max((abs(float(moments.get(group, 0.0)) - float(targets[group])) for group in targets), default=0.0)
