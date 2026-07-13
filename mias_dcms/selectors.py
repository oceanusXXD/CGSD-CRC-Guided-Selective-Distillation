from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
import math
import random
from typing import Any, Iterable, Mapping, Sequence


FORBIDDEN_SELECTOR_INPUT_FIELDS = {
    "chosen",
    "rejected",
    "preference_label",
    "preference_strength",
    "preference_magnitude",
    "preferred_response",
    "preference_statement",
    "preference_elaboration",
    "justification",
    "oracle_label",
    "true_class",
    "true_label",
    "true_label_name",
    "label",
    "label_name",
    "groundtruth",
    "ground_truth",
    "prediction_correct",
    "is_correct",
    "test_metric",
}


@dataclass(frozen=True)
class MomentMatchedRandomResult:
    selected_ids: list[str]
    selection_indicator: dict[str, int]
    rounded_moments: dict[str, float]
    target_moments: dict[str, float]
    max_constraint_violation: float
    budget: int
    seed: int
    solver_status: str = "feasible"


def assert_selector_rows_are_label_safe(rows: Iterable[dict[str, Any]]) -> None:
    for index, row in enumerate(rows):
        leaked = sorted(FORBIDDEN_SELECTOR_INPUT_FIELDS.intersection(row))
        if leaked:
            sample_id = row.get("sample_id", row.get("id", index))
            raise ValueError(f"selector row {sample_id!r} contains hidden fields: {leaked}")


def random_without_replacement(
    sample_ids: Sequence[str],
    *,
    budget: int,
    seed: int,
) -> list[str]:
    ids = _validate_ids_and_budget(sample_ids, budget)
    selected = random.Random(seed).sample(ids, budget)
    return sorted(selected)


def random_group_without_replacement(
    sample_ids: Sequence[str],
    group_ids: Sequence[str],
    *,
    budget: int,
    seed: int,
) -> list[str]:
    """Sample at most one candidate from each observable selection group."""
    ids = _validate_ids_and_budget(sample_ids, budget=0)
    groups = [str(group_id) for group_id in group_ids]
    if len(ids) != len(groups):
        raise ValueError("sample_ids and group_ids must have equal length")
    grouped: dict[str, list[str]] = {}
    for sample_id, group_id in zip(ids, groups, strict=True):
        if not group_id:
            raise ValueError("group_ids must not contain empty values")
        grouped.setdefault(group_id, []).append(sample_id)
    if budget < 0 or budget > len(grouped):
        raise ValueError("budget must be between 0 and unique selection-group count")
    rng = random.Random(seed)
    selected_groups = rng.sample(sorted(grouped), budget)
    return sorted(rng.choice(sorted(grouped[group_id])) for group_id in selected_groups)


def select_top_budget(
    *,
    sample_ids: Sequence[str],
    scores: Sequence[float],
    budget: int,
) -> list[str]:
    ids = _validate_ids_and_budget(sample_ids, budget)
    if len(ids) != len(scores):
        raise ValueError("sample_ids and scores must have equal length")
    ranked = sorted(
        zip(ids, (float(score) for score in scores)),
        key=lambda item: (-item[1], item[0]),
    )
    return [sample_id for sample_id, _ in ranked[:budget]]


def select_top_budget_by_group(
    *,
    sample_ids: Sequence[str],
    scores: Sequence[float],
    group_ids: Sequence[str],
    budget: int,
) -> list[str]:
    """Select the highest-score representative of each group, then top budget groups."""
    ids = _validate_ids_and_budget(sample_ids, budget=0)
    normalized_scores = [float(score) for score in scores]
    groups = [str(group_id) for group_id in group_ids]
    if len(ids) != len(normalized_scores) or len(ids) != len(groups):
        raise ValueError("sample_ids, scores, and group_ids must have equal length")
    representatives: dict[str, tuple[str, float]] = {}
    for sample_id, score, group_id in zip(ids, normalized_scores, groups, strict=True):
        if not group_id:
            raise ValueError("group_ids must not contain empty values")
        candidate = (sample_id, score)
        current = representatives.get(group_id)
        if current is None or (-candidate[1], candidate[0]) < (-current[1], current[0]):
            representatives[group_id] = candidate
    if budget < 0 or budget > len(representatives):
        raise ValueError("budget must be between 0 and unique selection-group count")
    ranked = sorted(representatives.values(), key=lambda item: (-item[1], item[0]))
    return [sample_id for sample_id, _ in ranked[:budget]]


def moment_matched_random(
    *,
    sample_ids: Sequence[str],
    group_membership: Sequence[Mapping[str, float]],
    budget: int,
    target_moments: Mapping[str, float],
    tolerance: float,
    seed: int,
) -> MomentMatchedRandomResult:
    ids = _validate_ids_and_budget(sample_ids, budget)
    if len(ids) != len(group_membership):
        raise ValueError("sample_ids and group_membership must have equal length")
    if tolerance < 0.0:
        raise ValueError("tolerance must be non-negative")

    groups = {str(group) for group in target_moments}
    for membership in group_membership:
        groups.update(str(group) for group in membership)
    targets = {group: float(target_moments.get(group, 0.0)) for group in groups}

    candidates: list[tuple[int, ...]] = []
    candidate_moments: dict[tuple[int, ...], dict[str, float]] = {}
    candidate_violations: dict[tuple[int, ...], float] = {}
    for candidate in combinations(range(len(ids)), budget):
        moments = _membership_moments(candidate, group_membership, groups, budget)
        violation = _max_moment_violation(moments, targets)
        if violation <= tolerance + 1e-12:
            candidates.append(candidate)
            candidate_moments[candidate] = moments
            candidate_violations[candidate] = violation
    if not candidates:
        raise ValueError("no moment-matched random batch is feasible")

    selected_indexes = random.Random(seed).choice(candidates)
    selected_id_set = {ids[index] for index in selected_indexes}
    selected_ids = sorted(selected_id_set)
    moments = {group: candidate_moments[selected_indexes][group] for group in sorted(groups)}
    return MomentMatchedRandomResult(
        selected_ids=selected_ids,
        selection_indicator={sample_id: int(sample_id in selected_id_set) for sample_id in ids},
        rounded_moments=moments,
        target_moments={group: targets[group] for group in sorted(targets)},
        max_constraint_violation=candidate_violations[selected_indexes],
        budget=int(budget),
        seed=int(seed),
    )


def entropy_uncertainty_scores(probabilities: Sequence[Sequence[float]]) -> list[float]:
    return [_entropy(row) for row in probabilities]


def margin_uncertainty_scores(probabilities: Sequence[Sequence[float]]) -> list[float]:
    scores: list[float] = []
    for row in probabilities:
        values = _normalize_probabilities(row)
        if len(values) < 2:
            raise ValueError("margin uncertainty requires at least two classes")
        top_two = sorted(values, reverse=True)[:2]
        scores.append(1.0 - (top_two[0] - top_two[1]))
    return scores


def _membership_moments(
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


def _max_moment_violation(
    moments: Mapping[str, float],
    targets: Mapping[str, float],
) -> float:
    return max((abs(float(moments[group]) - float(targets[group])) for group in targets), default=0.0)


def _validate_ids_and_budget(sample_ids: Sequence[str], budget: int) -> list[str]:
    ids = [str(sample_id) for sample_id in sample_ids]
    if len(set(ids)) != len(ids):
        raise ValueError("sample_ids must be unique")
    if budget < 0 or budget > len(ids):
        raise ValueError("budget must be between 0 and sample count")
    return ids


def _entropy(probabilities: Sequence[float]) -> float:
    values = _normalize_probabilities(probabilities)
    return -sum(value * math.log(value) for value in values if value > 0.0)


def _normalize_probabilities(probabilities: Sequence[float]) -> list[float]:
    values = [float(value) for value in probabilities]
    if not values:
        raise ValueError("probability rows must not be empty")
    if any(value < 0.0 for value in values):
        raise ValueError("probabilities must be non-negative")
    total = sum(values)
    if total <= 0.0:
        raise ValueError("probability row must have positive mass")
    return [value / total for value in values]
