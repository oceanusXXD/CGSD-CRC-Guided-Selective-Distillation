from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations
import random
from typing import Mapping, Sequence


EXACT_ENUMERATION_MAX_SAMPLES = 18


@dataclass(frozen=True)
class DCMSSlackTrace:
    slack: float
    feasible: bool
    utility_retained: float
    max_constraint_violation: float | None
    expected_moments: dict[str, float] = field(default_factory=dict)
    meets_utility_threshold: bool = False
    solver_status: str = "not_run"


@dataclass(frozen=True)
class DCMSFrontierPoint:
    slack: float
    feasible: bool
    utility_retained: float
    coverage_deviation: float | None
    max_constraint_violation: float | None
    expected_moments: dict[str, float] = field(default_factory=dict)
    selected_ids: list[str] = field(default_factory=list)
    meets_utility_threshold: bool = False
    solver_status: str = "not_run"

    def as_dict(self) -> dict[str, object]:
        return {
            "slack": self.slack,
            "feasible": self.feasible,
            "utility_retained": self.utility_retained,
            "coverage_deviation": self.coverage_deviation,
            "max_constraint_violation": self.max_constraint_violation,
            "expected_moments": dict(self.expected_moments),
            "selected_ids": list(self.selected_ids),
            "meets_utility_threshold": self.meets_utility_threshold,
            "solver_status": self.solver_status,
        }


@dataclass(frozen=True)
class DCMSUtilityCoverageFrontier:
    selected_slack: float | None
    kappa: float
    utility_threshold: float
    target_moments: dict[str, float]
    points: list[DCMSFrontierPoint]

    def as_dict(self) -> dict[str, object]:
        return {
            "selected_slack": self.selected_slack,
            "kappa": self.kappa,
            "utility_threshold": self.utility_threshold,
            "target_moments": dict(self.target_moments),
            "points": [point.as_dict() for point in self.points],
        }


@dataclass(frozen=True)
class DCMSResult:
    selected_ids: list[str]
    q_propensity: dict[str, float]
    selection_indicator: dict[str, int]
    continuous_moments: dict[str, float]
    rounded_moments: dict[str, float]
    robust_lower_moments: dict[str, float]
    robust_upper_moments: dict[str, float]
    utility_retained: float
    max_constraint_violation: float
    solver_status: str
    rounding_seed: int | None = None
    selected_slack: float | None = None
    slack_trace: list[DCMSSlackTrace] = field(default_factory=list)


def solve_dcms(
    *,
    sample_ids: Sequence[str],
    utilities: Sequence[float],
    group_membership: Sequence[Mapping[str, float]],
    membership_lower: Sequence[Mapping[str, float]] | None = None,
    membership_upper: Sequence[Mapping[str, float]] | None = None,
    budget: int,
    target_moments: Mapping[str, float],
    tolerance: float,
    rounding_seed: int | None = None,
) -> DCMSResult:
    if len(sample_ids) != len(utilities) or len(sample_ids) != len(group_membership):
        raise ValueError("sample_ids, utilities, and group_membership must have equal length")
    if membership_lower is not None and len(membership_lower) != len(sample_ids):
        raise ValueError("membership_lower must match sample count")
    if membership_upper is not None and len(membership_upper) != len(sample_ids):
        raise ValueError("membership_upper must match sample count")
    if budget < 0 or budget > len(sample_ids):
        raise ValueError("budget must be between 0 and sample count")
    if tolerance < 0.0:
        raise ValueError("tolerance must be non-negative")
    if len(set(sample_ids)) != len(sample_ids):
        raise ValueError("sample_ids must be unique")

    ids = [str(sample_id) for sample_id in sample_ids]
    utility_values = [float(value) for value in utilities]
    groups = {str(group) for group in target_moments}
    for membership in group_membership:
        groups.update(str(group) for group in membership)
    for bounds in (membership_lower or []):
        groups.update(str(group) for group in bounds)
    for bounds in (membership_upper or []):
        groups.update(str(group) for group in bounds)
    targets = {group: float(target_moments.get(group, 0.0)) for group in groups}
    lower_membership = list(membership_lower) if membership_lower is not None else group_membership
    upper_membership = list(membership_upper) if membership_upper is not None else group_membership

    if len(ids) > EXACT_ENUMERATION_MAX_SAMPLES:
        return _solve_dcms_scalable(
            ids=ids,
            utility_values=utility_values,
            group_membership=group_membership,
            lower_membership=lower_membership,
            upper_membership=upper_membership,
            groups=groups,
            targets=targets,
            budget=budget,
            tolerance=tolerance,
            rounding_seed=rounding_seed,
        )

    top_utility = _top_utility(utility_values, budget)
    best: tuple[
        float,
        float,
        tuple[int, ...],
        dict[str, float],
        dict[str, float],
        dict[str, float],
    ] | None = None
    candidate_indexes = list(combinations(range(len(ids)), budget))
    if rounding_seed is not None:
        random.Random(rounding_seed).shuffle(candidate_indexes)
    for candidate in candidate_indexes:
        moments = _moments(candidate, group_membership, groups, budget)
        lower_moments = _moments(candidate, lower_membership, groups, budget)
        upper_moments = _moments(candidate, upper_membership, groups, budget)
        violation = _max_robust_violation(lower_moments, upper_moments, targets)
        if violation > tolerance + 1e-12:
            continue
        utility = sum(utility_values[index] for index in candidate)
        # Maximize utility, then prefer stricter moments, then stable or seeded tie order.
        tie_break = tuple(-index for index in candidate) if rounding_seed is None else ()
        score = (utility, -violation, tie_break, moments)
        if best is None or score[:3] > best[:3]:
            best = (utility, violation, candidate, moments, lower_moments, upper_moments)
    if best is None:
        raise ValueError("DCMS constraints are infeasible for the requested budget")

    selected_utility, violation, selected_indexes, moments, lower_moments, upper_moments = best
    selected_ids = [ids[index] for index in sorted(selected_indexes, key=lambda index: (-utility_values[index], ids[index]))]
    retained = selected_utility / top_utility if top_utility > 0.0 else 1.0
    propensities = {sample_id: (1.0 if index in selected_indexes else 0.0) for index, sample_id in enumerate(ids)}
    indicators = {sample_id: (1 if index in selected_indexes else 0) for index, sample_id in enumerate(ids)}
    ordered_moments = {group: moments[group] for group in sorted(moments)}
    ordered_lower = {group: lower_moments[group] for group in sorted(lower_moments)}
    ordered_upper = {group: upper_moments[group] for group in sorted(upper_moments)}
    return DCMSResult(
        selected_ids=selected_ids,
        q_propensity=propensities,
        selection_indicator=indicators,
        continuous_moments=ordered_moments,
        rounded_moments=ordered_moments,
        robust_lower_moments=ordered_lower,
        robust_upper_moments=ordered_upper,
        utility_retained=retained,
        max_constraint_violation=violation,
        solver_status="feasible",
        rounding_seed=rounding_seed,
    )


def solve_dcms_with_slack(
    *,
    sample_ids: Sequence[str],
    utilities: Sequence[float],
    group_membership: Sequence[Mapping[str, float]],
    budget: int,
    target_moments: Mapping[str, float],
    slack_grid: Sequence[float],
    kappa: float,
    membership_lower: Sequence[Mapping[str, float]] | None = None,
    membership_upper: Sequence[Mapping[str, float]] | None = None,
    rounding_seed: int | None = None,
) -> DCMSResult:
    if not slack_grid:
        raise ValueError("slack_grid must not be empty")
    if not 0.0 <= kappa <= 1.0:
        raise ValueError("kappa must be between 0 and 1")

    threshold = 1.0 - kappa
    traces: list[DCMSSlackTrace] = []
    chosen: tuple[float, DCMSResult] | None = None
    for slack in sorted(float(value) for value in slack_grid):
        try:
            result = solve_dcms(
                sample_ids=sample_ids,
                utilities=utilities,
                group_membership=group_membership,
                membership_lower=membership_lower,
                membership_upper=membership_upper,
                budget=budget,
                target_moments=target_moments,
                tolerance=slack,
                rounding_seed=rounding_seed,
            )
        except ValueError:
            traces.append(
                DCMSSlackTrace(
                    slack=slack,
                    feasible=False,
                    utility_retained=0.0,
                    max_constraint_violation=None,
                    solver_status="infeasible",
                )
            )
            continue

        meets_threshold = result.utility_retained >= threshold - 1e-12
        traces.append(
            DCMSSlackTrace(
                slack=slack,
                feasible=True,
                utility_retained=result.utility_retained,
                max_constraint_violation=result.max_constraint_violation,
                expected_moments=dict(result.continuous_moments),
                meets_utility_threshold=meets_threshold,
                solver_status=result.solver_status,
            )
        )
        if chosen is None and meets_threshold:
            chosen = (slack, result)

    if chosen is None:
        raise ValueError("no feasible slack retains the requested utility")

    selected_slack, result = chosen
    return DCMSResult(
        selected_ids=result.selected_ids,
        q_propensity=result.q_propensity,
        selection_indicator=result.selection_indicator,
        continuous_moments=result.continuous_moments,
        rounded_moments=result.rounded_moments,
        robust_lower_moments=result.robust_lower_moments,
        robust_upper_moments=result.robust_upper_moments,
        utility_retained=result.utility_retained,
        max_constraint_violation=result.max_constraint_violation,
        solver_status=result.solver_status,
        rounding_seed=result.rounding_seed,
        selected_slack=selected_slack,
        slack_trace=traces,
    )


def dcms_utility_coverage_frontier(
    *,
    sample_ids: Sequence[str],
    utilities: Sequence[float],
    group_membership: Sequence[Mapping[str, float]],
    budget: int,
    target_moments: Mapping[str, float],
    slack_grid: Sequence[float],
    kappa: float,
    membership_lower: Sequence[Mapping[str, float]] | None = None,
    membership_upper: Sequence[Mapping[str, float]] | None = None,
    rounding_seed: int | None = None,
) -> DCMSUtilityCoverageFrontier:
    if not slack_grid:
        raise ValueError("slack_grid must not be empty")
    if not 0.0 <= kappa <= 1.0:
        raise ValueError("kappa must be between 0 and 1")

    threshold = 1.0 - kappa
    groups = _groups_from_inputs(
        target_moments=target_moments,
        group_membership=group_membership,
        membership_lower=membership_lower,
        membership_upper=membership_upper,
    )
    targets = {group: float(target_moments.get(group, 0.0)) for group in sorted(groups)}
    points: list[DCMSFrontierPoint] = []
    selected_slack: float | None = None
    for slack in sorted(float(value) for value in slack_grid):
        try:
            result = solve_dcms(
                sample_ids=sample_ids,
                utilities=utilities,
                group_membership=group_membership,
                membership_lower=membership_lower,
                membership_upper=membership_upper,
                budget=budget,
                target_moments=target_moments,
                tolerance=slack,
                rounding_seed=rounding_seed,
            )
        except ValueError:
            points.append(
                DCMSFrontierPoint(
                    slack=slack,
                    feasible=False,
                    utility_retained=0.0,
                    coverage_deviation=None,
                    max_constraint_violation=None,
                    solver_status="infeasible",
                )
            )
            continue

        meets_threshold = result.utility_retained >= threshold - 1e-12
        if selected_slack is None and meets_threshold:
            selected_slack = slack
        points.append(
            DCMSFrontierPoint(
                slack=slack,
                feasible=True,
                utility_retained=result.utility_retained,
                coverage_deviation=_max_violation(result.rounded_moments, targets),
                max_constraint_violation=result.max_constraint_violation,
                expected_moments=dict(result.continuous_moments),
                selected_ids=list(result.selected_ids),
                meets_utility_threshold=meets_threshold,
                solver_status=result.solver_status,
            )
        )
    return DCMSUtilityCoverageFrontier(
        selected_slack=selected_slack,
        kappa=float(kappa),
        utility_threshold=threshold,
        target_moments=targets,
        points=points,
    )


def rank_normalize_utilities(utilities: Sequence[float]) -> list[float]:
    if not utilities:
        return []
    if len(utilities) == 1:
        return [1.0]
    ordered = sorted(enumerate(float(value) for value in utilities), key=lambda item: (item[1], item[0]))
    normalized = [0.0] * len(ordered)
    denominator = len(ordered) - 1
    for rank, (index, _) in enumerate(ordered):
        normalized[index] = rank / denominator
    return normalized


def _top_utility(utilities: Sequence[float], budget: int) -> float:
    if budget == 0:
        return 0.0
    return sum(sorted(utilities, reverse=True)[:budget])


def _solve_dcms_scalable(
    *,
    ids: Sequence[str],
    utility_values: Sequence[float],
    group_membership: Sequence[Mapping[str, float]],
    lower_membership: Sequence[Mapping[str, float]],
    upper_membership: Sequence[Mapping[str, float]],
    groups: set[str],
    targets: Mapping[str, float],
    budget: int,
    tolerance: float,
    rounding_seed: int | None,
) -> DCMSResult:
    """Solve the large-pool relaxation with LP + entropy refinement.

    Small pools retain the exact reference solver above for transparent unit
    tests. Real pools use a continuous robust relaxation and systematic
    rounding, avoiding combinatorial enumeration.
    """
    if budget == 0:
        zero = {group: 0.0 for group in sorted(groups)}
        return DCMSResult(
            selected_ids=[],
            q_propensity={sample_id: 0.0 for sample_id in ids},
            selection_indicator={sample_id: 0 for sample_id in ids},
            continuous_moments=zero,
            rounded_moments=zero,
            robust_lower_moments=zero,
            robust_upper_moments=zero,
            utility_retained=1.0,
            max_constraint_violation=0.0,
            solver_status="scalable_zero_budget",
            rounding_seed=rounding_seed,
        )

    try:
        import numpy as np
        from scipy.optimize import linprog, minimize
    except ImportError as exc:
        raise ImportError(
            "large-pool DCMS requires numpy and scipy; install requirements.txt"
        ) from exc

    sample_count = len(ids)
    utilities = np.asarray(utility_values, dtype=float)
    lower = np.asarray(
        [[float(row.get(group, 0.0)) for group in sorted(groups)] for row in lower_membership],
        dtype=float,
    ).T
    upper = np.asarray(
        [[float(row.get(group, 0.0)) for group in sorted(groups)] for row in upper_membership],
        dtype=float,
    ).T
    sorted_groups = sorted(groups)
    target_values = np.asarray([float(targets[group]) for group in sorted_groups], dtype=float)
    if budget <= 0:
        raise ValueError("budget must be positive in scalable DCMS")

    a_ub = []
    b_ub = []
    for index in range(len(sorted_groups)):
        a_ub.append(upper[index] / budget)
        b_ub.append(target_values[index] + tolerance)
        a_ub.append(-lower[index] / budget)
        b_ub.append(-(target_values[index] - tolerance))
    equality = np.ones((1, sample_count), dtype=float)
    lp = linprog(
        -utilities,
        A_ub=np.asarray(a_ub, dtype=float),
        b_ub=np.asarray(b_ub, dtype=float),
        A_eq=equality,
        b_eq=np.asarray([float(budget)]),
        bounds=[(0.0, 1.0)] * sample_count,
        method="highs",
    )
    if not lp.success:
        raise ValueError(f"DCMS continuous relaxation is infeasible: {lp.message}")

    q = np.asarray(lp.x, dtype=float)
    entropy_weight = 1e-3

    def objective(values: np.ndarray) -> float:
        clipped = np.clip(values, 1e-9, 1.0 - 1e-9)
        entropy = -clipped * np.log(clipped) - (1.0 - clipped) * np.log(1.0 - clipped)
        return float(-(utilities @ clipped + entropy_weight * entropy.sum()))

    def gradient(values: np.ndarray) -> np.ndarray:
        clipped = np.clip(values, 1e-9, 1.0 - 1e-9)
        entropy_gradient = np.log((1.0 - clipped) / clipped)
        return -utilities - entropy_weight * entropy_gradient

    refined = minimize(
        objective,
        q,
        jac=gradient,
        method="SLSQP",
        bounds=[(0.0, 1.0)] * sample_count,
        constraints=[
            {"type": "eq", "fun": lambda values: float(np.sum(values) - budget), "jac": lambda _values: equality[0]},
            {
                "type": "ineq",
                "fun": lambda values: np.asarray(b_ub) - np.asarray(a_ub) @ values,
                "jac": lambda _values: -np.asarray(a_ub),
            },
        ],
        options={"maxiter": 200, "ftol": 1e-9},
    )
    if refined.success and np.all(np.isfinite(refined.x)):
        q = np.clip(np.asarray(refined.x, dtype=float), 0.0, 1.0)

    selected_indexes = _systematic_round(q, budget=budget, seed=rounding_seed)
    selected_indexes = _repair_rounding(
        selected_indexes,
        q=q,
        utilities=utilities,
        lower=lower,
        upper=upper,
        targets=target_values,
        budget=budget,
        tolerance=tolerance,
    )
    selected_set = set(selected_indexes)
    continuous_moments = {
        group: float(np.dot(q, np.asarray([row.get(group, 0.0) for row in group_membership])) / budget)
        for group in sorted_groups
    }
    rounded_moments = {
        group: float(np.mean([group_membership[index].get(group, 0.0) for index in selected_indexes]))
        for group in sorted_groups
    }
    lower_moments = {
        group: float(np.mean([lower_membership[index].get(group, 0.0) for index in selected_indexes]))
        for group in sorted_groups
    }
    upper_moments = {
        group: float(np.mean([upper_membership[index].get(group, 0.0) for index in selected_indexes]))
        for group in sorted_groups
    }
    selected_utility = sum(float(utility_values[index]) for index in selected_indexes)
    top_utility = _top_utility(utility_values, budget)
    selected_ids = [
        str(ids[index])
        for index in sorted(selected_indexes, key=lambda index: (-utility_values[index], ids[index]))
    ]
    return DCMSResult(
        selected_ids=selected_ids,
        q_propensity={str(sample_id): float(value) for sample_id, value in zip(ids, q, strict=True)},
        selection_indicator={str(sample_id): int(index in selected_set) for index, sample_id in enumerate(ids)},
        continuous_moments=continuous_moments,
        rounded_moments=rounded_moments,
        robust_lower_moments=lower_moments,
        robust_upper_moments=upper_moments,
        utility_retained=selected_utility / top_utility if top_utility > 0.0 else 1.0,
        max_constraint_violation=_max_robust_violation(lower_moments, upper_moments, targets),
        solver_status="scalable_slsqp" if refined.success else "scalable_lp",
        rounding_seed=rounding_seed,
    )


def _systematic_round(values: Any, *, budget: int, seed: int | None) -> list[int]:
    import numpy as np

    probabilities = np.asarray(values, dtype=float)
    total = float(probabilities.sum())
    if total <= 0.0:
        raise ValueError("continuous DCMS solution has zero inclusion mass")
    probabilities = probabilities * (float(budget) / total)
    probabilities = np.clip(probabilities, 0.0, 1.0)
    cumulative = np.cumsum(probabilities)
    rng = random.Random(seed)
    offset = rng.random() if seed is not None else 0.5
    thresholds = offset + np.arange(int(budget), dtype=float)
    selected = [int(np.searchsorted(cumulative, threshold, side="right")) for threshold in thresholds]
    selected = sorted({index for index in selected if 0 <= index < len(probabilities)})
    if len(selected) < budget:
        ranked = sorted(range(len(probabilities)), key=lambda index: (-probabilities[index], index))
        selected.extend(index for index in ranked if index not in set(selected))
        selected = selected[:budget]
    return selected


def _repair_rounding(
    selected_indexes: Sequence[int],
    *,
    q: Any,
    utilities: Any,
    lower: Any,
    upper: Any,
    targets: Any,
    budget: int,
    tolerance: float,
) -> list[int]:
    import numpy as np

    selected = set(int(index) for index in selected_indexes)
    if len(selected) != budget:
        ranked = sorted(range(len(q)), key=lambda index: (-float(q[index]), -float(utilities[index]), index))
        selected = set(ranked[:budget])

    def violation(indexes: set[int]) -> float:
        chosen = sorted(indexes)
        lower_moments = np.mean(lower[:, chosen], axis=1)
        upper_moments = np.mean(upper[:, chosen], axis=1)
        return float(
            np.max(
                np.maximum(
                    0.0,
                    np.maximum(upper_moments - targets, targets - lower_moments),
                )
            )
        )

    current_violation = violation(selected)
    if current_violation <= tolerance + 1e-9:
        return sorted(selected)
    for _ in range(max(1, len(q) * min(budget, 64))):
        best = None
        for outgoing in sorted(selected):
            for incoming in range(len(q)):
                if incoming in selected:
                    continue
                candidate = set(selected)
                candidate.remove(outgoing)
                candidate.add(incoming)
                candidate_violation = violation(candidate)
                candidate_key = (candidate_violation, -float(utilities[list(candidate)].sum()), -float(q[incoming]), incoming)
                if best is None or candidate_key < best[0]:
                    best = (candidate_key, candidate)
        if best is None or best[0][0] >= current_violation - 1e-12:
            break
        current_violation, selected = best[0][0], best[1]
        if current_violation <= tolerance + 1e-9:
            return sorted(selected)
    raise ValueError(
        f"dependent rounding could not satisfy DCMS tolerance={tolerance}; "
        f"rounded_violation={current_violation}"
    )


def _moments(
    selected_indexes: Sequence[int],
    group_membership: Sequence[Mapping[str, float]],
    groups: set[str],
    budget: int,
) -> dict[str, float]:
    if budget == 0:
        return {group: 0.0 for group in groups}
    return {
        group: sum(float(group_membership[index].get(group, 0.0)) for index in selected_indexes) / budget
        for group in groups
    }


def _groups_from_inputs(
    *,
    target_moments: Mapping[str, float],
    group_membership: Sequence[Mapping[str, float]],
    membership_lower: Sequence[Mapping[str, float]] | None,
    membership_upper: Sequence[Mapping[str, float]] | None,
) -> set[str]:
    groups = {str(group) for group in target_moments}
    for membership in group_membership:
        groups.update(str(group) for group in membership)
    for bounds in membership_lower or []:
        groups.update(str(group) for group in bounds)
    for bounds in membership_upper or []:
        groups.update(str(group) for group in bounds)
    return groups


def _max_violation(moments: Mapping[str, float], targets: Mapping[str, float]) -> float:
    return max((abs(float(moments[group]) - float(targets[group])) for group in targets), default=0.0)


def _max_robust_violation(
    lower_moments: Mapping[str, float],
    upper_moments: Mapping[str, float],
    targets: Mapping[str, float],
) -> float:
    """Return the worst endpoint deviation from a robust target interval.

    For every feasible true membership ``a`` satisfying ``lower <= a <= upper``,
    the selected batch must satisfy ``target - tolerance <= a_bar <= target +
    tolerance``.  Consequently the upper endpoint must not exceed the target
    and the lower endpoint must not fall below it (before applying tolerance).
    """
    violations = []
    for group, target in targets.items():
        lower = float(lower_moments[group])
        upper = float(upper_moments[group])
        target_value = float(target)
        violations.append(max(0.0, upper - target_value, target_value - lower))
    return max(violations, default=0.0)
