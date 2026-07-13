from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
import math
import random
import re
from statistics import mean, median, pstdev
from typing import Any, Mapping

from mias_dcms.selection.dcms import rank_normalize_utilities, solve_dcms_with_slack
from mias_dcms.selectors import (
    FORBIDDEN_SELECTOR_INPUT_FIELDS,
    assert_selector_rows_are_label_safe,
)


DEFAULT_DCMS_SLACK_GRID = (0.0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5)
DEFAULT_DCMS_KAPPA = 0.05


def select_rows(
    rows: Iterable[dict[str, Any]],
    *,
    method: str,
    budget: int,
    seed: int,
) -> list[dict[str, Any]]:
    candidates = [dict(row) for row in rows]
    if budget < 0 or budget > len(candidates):
        raise ValueError(f"budget must be between 0 and {len(candidates)}")
    if method == "random":
        random.Random(seed).shuffle(candidates)
    elif method == "entropy":
        candidates.sort(key=lambda row: (-_entropy(_probabilities(row)), str(row["id"])))
    elif method == "margin":
        candidates.sort(key=lambda row: (_margin(_probabilities(row)), str(row["id"])))
    elif method == "badge":
        return _select_badge(candidates, budget=budget, seed=seed)
    elif method == "galaxy":
        return _select_galaxy(candidates, budget=budget, seed=seed)
    else:
        raise ValueError(f"unsupported selection method: {method}")
    return candidates[:budget]


def select_classification_rows(
    rows: Iterable[dict[str, Any]],
    *,
    method: str,
    budget: int,
    seed: int,
    dcms_target: str | Mapping[str, float] = "uniform",
    dcms_slack_grid: Iterable[float] = DEFAULT_DCMS_SLACK_GRID,
    dcms_kappa: float = DEFAULT_DCMS_KAPPA,
) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    """Select classification rows and optionally apply a soft-posterior DCMS wrapper.

    The input must already be selector-safe.  For ``*_dcms`` methods, the
    class memberships are the model probabilities, so true labels never enter
    the solver.  The returned metadata is suitable for an auditable selection
    artifact and contains the continuous inclusion propensities.
    """
    candidates = [dict(row) for row in rows]
    assert_selector_rows_are_label_safe(candidates)
    normalized_method = _normalize_classification_method(method)
    if not normalized_method.endswith("_dcms"):
        return (
            select_rows(candidates, method=normalized_method, budget=budget, seed=seed),
            None,
        )

    base_method = normalized_method.removesuffix("_dcms")
    sample_ids = [str(row["id"]) for row in candidates]
    posterior_values = [row.get("cross_fitted_class_posterior") for row in candidates]
    has_cross_fitted_posterior = any(value is not None for value in posterior_values)
    if has_cross_fitted_posterior and not all(value is not None for value in posterior_values):
        raise ValueError("cross_fitted_class_posterior must be present for every classification row")
    probabilities = [
        _classification_group_probabilities(row)
        for row in candidates
    ]
    memberships = [
        {f"class={index}": probability for index, probability in enumerate(values)}
        for values in probabilities
    ]
    utilities = [
        _classification_dcms_utility(row, base_method)
        for row in candidates
    ]
    target_moments = _classification_target_moments(
        probabilities,
        target=dcms_target,
    )
    result = solve_dcms_with_slack(
        sample_ids=sample_ids,
        utilities=rank_normalize_utilities(utilities),
        group_membership=memberships,
        budget=budget,
        target_moments=target_moments,
        slack_grid=tuple(float(value) for value in dcms_slack_grid),
        kappa=dcms_kappa,
        rounding_seed=seed,
    )
    rows_by_id = {str(row["id"]): row for row in candidates}
    selected = [rows_by_id[sample_id] for sample_id in result.selected_ids]
    metadata = {
        "method": normalized_method,
        "base_method": base_method,
        "budget": int(budget),
        "target": str(dcms_target) if isinstance(dcms_target, str) else "explicit",
        "target_moments": dict(target_moments),
        "group_membership": "soft_class_posterior",
        "posterior_source": (
            "cross_fitted_class_posterior" if has_cross_fitted_posterior else "probabilities_proxy"
        ),
        "utility_normalization": "rank",
        "selected_ids": list(result.selected_ids),
        "q_propensity": dict(result.q_propensity),
        "selection_indicator": dict(result.selection_indicator),
        "continuous_moments": dict(result.continuous_moments),
        "rounded_moments": dict(result.rounded_moments),
        "robust_lower_moments": dict(result.robust_lower_moments),
        "robust_upper_moments": dict(result.robust_upper_moments),
        "utility_retained": float(result.utility_retained),
        "max_constraint_violation": float(result.max_constraint_violation),
        "selected_slack": result.selected_slack,
        "solver_status": result.solver_status,
        "rounding_seed": result.rounding_seed,
        "slack_trace": [
            {
                "slack": trace.slack,
                "feasible": trace.feasible,
                "utility_retained": trace.utility_retained,
                "max_constraint_violation": trace.max_constraint_violation,
                "expected_moments": dict(trace.expected_moments),
                "meets_utility_threshold": trace.meets_utility_threshold,
                "solver_status": trace.solver_status,
            }
            for trace in result.slack_trace
        ],
    }
    return selected, metadata


def _normalize_classification_method(method: str) -> str:
    normalized = str(method).strip().lower().replace("+", "_").replace("-", "_")
    aliases = {
        "random": "random",
        "entropy": "entropy",
        "margin": "margin",
        "badge": "badge",
        "galaxy": "galaxy",
        "entropy_dcms": "entropy_dcms",
        "badge_dcms": "badge_dcms",
    }
    if normalized not in aliases:
        raise ValueError(f"unsupported classification selection method: {method!r}")
    return aliases[normalized]


def _classification_dcms_utility(row: Mapping[str, Any], method: str) -> float:
    probabilities = _probabilities(dict(row))
    if method == "entropy":
        return _entropy(probabilities)
    if method == "badge":
        return math.sqrt(sum(value * value for value in _badge_gradient_embedding(row)))
    raise ValueError(f"unsupported DCMS base method: {method!r}")


def _classification_group_probabilities(row: Mapping[str, Any]) -> list[float]:
    value = row.get("cross_fitted_class_posterior")
    if value is None:
        return _probabilities(dict(row))
    if not isinstance(value, (list, tuple)):
        raise ValueError("cross_fitted_class_posterior must be a list or tuple")
    values = [float(item) for item in value]
    if len(values) < 2:
        raise ValueError("cross_fitted_class_posterior must contain at least two classes")
    if any(not math.isfinite(item) or item < 0.0 for item in values):
        raise ValueError("cross_fitted_class_posterior must be finite and non-negative")
    total = sum(values)
    if total <= 0.0:
        raise ValueError("cross_fitted_class_posterior must have positive mass")
    return [item / total for item in values]


def _classification_target_moments(
    probabilities: list[list[float]],
    *,
    target: str | Mapping[str, float],
) -> dict[str, float]:
    if not probabilities:
        raise ValueError("probabilities must not be empty")
    class_count = len(probabilities[0])
    if any(len(values) != class_count for values in probabilities):
        raise ValueError("all probability rows must have the same class count")
    if isinstance(target, Mapping):
        resolved = {str(key): float(value) for key, value in target.items()}
        expected = {f"class={index}" for index in range(class_count)}
        if set(resolved) != expected:
            raise ValueError("explicit DCMS target must cover every class exactly once")
        return resolved
    normalized_target = str(target).strip().lower()
    if normalized_target == "uniform":
        value = 1.0 / class_count
        return {f"class={index}": value for index in range(class_count)}
    if normalized_target == "pool":
        return {
            f"class={index}": mean(values[index] for values in probabilities)
            for index in range(class_count)
        }
    raise ValueError(f"unsupported DCMS target: {target!r}")


def _select_badge(
    candidates: list[dict[str, Any]],
    *,
    budget: int,
    seed: int,
) -> list[dict[str, Any]]:
    """Select diverse uncertainty-weighted gradient embeddings.

    Rows must expose ``representation_embedding`` (or ``embedding``).  The
    embedding is combined with the predicted-class residual, matching the
    BADGE gradient proxy while keeping selection label-safe.
    """
    vectors = [_badge_gradient_embedding(row) for row in candidates]
    selected_indexes = _farthest_point_indexes(vectors, budget=budget, seed=seed)
    return [candidates[index] for index in selected_indexes]


def _select_galaxy(
    candidates: list[dict[str, Any]],
    *,
    budget: int,
    seed: int,
) -> list[dict[str, Any]]:
    """Select a graph-covering batch using frozen representation embeddings."""
    vectors = [_representation_embedding(row) for row in candidates]
    selected_indexes = _facility_location_indexes(vectors, candidates, budget=budget, seed=seed)
    return [candidates[index] for index in selected_indexes]


def _badge_gradient_embedding(row: Mapping[str, Any]) -> list[float]:
    probabilities = _probabilities(dict(row))
    representation = _representation_embedding(row)
    predicted = max(range(len(probabilities)), key=probabilities.__getitem__)
    residual = [
        (1.0 if class_index == predicted else 0.0) - probability
        for class_index, probability in enumerate(probabilities)
    ]
    return [
        float(residual_value * embedding_value)
        for residual_value in residual
        for embedding_value in representation
    ]


def _representation_embedding(row: Mapping[str, Any]) -> list[float]:
    value = row.get("representation_embedding", row.get("embedding"))
    if not isinstance(value, (list, tuple)) or not value:
        sample_id = row.get("id", row.get("sample_id", "<unknown>"))
        raise ValueError(
            f"{sample_id!r} is missing representation_embedding required by BADGE/GALAXY"
        )
    vector = [float(item) for item in value]
    if any(not math.isfinite(item) for item in vector):
        raise ValueError("representation embeddings must be finite")
    norm = math.sqrt(sum(item * item for item in vector))
    if norm <= 1e-12:
        raise ValueError("representation embeddings must have non-zero norm")
    return [item / norm for item in vector]


def _farthest_point_indexes(
    vectors: list[list[float]],
    *,
    budget: int,
    seed: int,
) -> list[int]:
    if not vectors:
        return []
    norms = [sum(value * value for value in vector) for vector in vectors]
    first_candidates = [index for index, norm in enumerate(norms) if norm == max(norms)]
    first = random.Random(seed).choice(sorted(first_candidates))
    selected = [first]
    remaining = set(range(len(vectors))) - {first}
    while len(selected) < budget and remaining:
        best_index = max(
            remaining,
            key=lambda index: (
                min(_squared_distance(vectors[index], vectors[chosen]) for chosen in selected),
                str(index),
            ),
        )
        selected.append(best_index)
        remaining.remove(best_index)
    return selected


def _facility_location_indexes(
    vectors: list[list[float]],
    candidates: list[dict[str, Any]],
    *,
    budget: int,
    seed: int,
) -> list[int]:
    if not vectors:
        return []
    similarities = [
        [max(-1.0, min(1.0, _dot(left, right))) for right in vectors]
        for left in vectors
    ]
    first = max(
        range(len(vectors)),
        key=lambda index: (_entropy(_probabilities(candidates[index])), str(candidates[index]["id"])),
    )
    selected = [first]
    remaining = set(range(len(vectors))) - {first}
    current_coverage = [similarities[index][first] for index in range(len(vectors))]
    while len(selected) < budget and remaining:
        best_index = max(
            remaining,
            key=lambda candidate_index: (
                sum(
                    max(0.0, similarities[index][candidate_index] - current_coverage[index])
                    for index in range(len(vectors))
                ),
                str(candidates[candidate_index]["id"]),
            ),
        )
        selected.append(best_index)
        remaining.remove(best_index)
        current_coverage = [
            max(current_coverage[index], similarities[index][best_index])
            for index in range(len(vectors))
        ]
    return selected


def _dot(left: list[float], right: list[float]) -> float:
    if len(left) != len(right):
        raise ValueError("representation embeddings must have equal dimensions")
    return sum(a * b for a, b in zip(left, right, strict=True))


def _squared_distance(left: list[float], right: list[float]) -> float:
    if len(left) != len(right):
        raise ValueError("representation embeddings must have equal dimensions")
    return sum((a - b) ** 2 for a, b in zip(left, right, strict=True))


def selector_safe_view(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return rows with oracle/ground-truth fields removed before selection."""
    return [
        {
            key: value
            for key, value in dict(row).items()
            if key not in FORBIDDEN_SELECTOR_INPUT_FIELDS
        }
        for row in rows
    ]


def classification_shift_report(
    rows: Iterable[dict[str, Any]],
    *,
    selected_ids: set[str],
) -> dict[str, Any]:
    pool = list(rows)
    selected = [row for row in pool if str(row["id"]) in selected_ids]
    if not selected and selected_ids:
        raise ValueError("selected_ids did not match any rows")
    pool_counts = Counter(str(int(row["label"])) for row in pool)
    selected_counts = Counter(str(int(row["label"])) for row in selected)
    labels = sorted(set(pool_counts) | set(selected_counts), key=int)
    pool_size = len(pool)
    selected_size = len(selected)
    per_class: dict[str, dict[str, float | int | None]] = {}
    category_tv = 0.0
    coverage_values: list[float] = []
    for label in labels:
        pool_share = pool_counts[label] / pool_size if pool_size else 0.0
        selected_share = selected_counts[label] / selected_size if selected_size else 0.0
        coverage_ratio = selected_share / pool_share if pool_share else None
        category_tv += abs(selected_share - pool_share)
        if coverage_ratio is not None:
            coverage_values.append(coverage_ratio)
        per_class[label] = {
            "pool_count": pool_counts[label],
            "selected_count": selected_counts[label],
            "pool_share": pool_share,
            "selected_share": selected_share,
            "signed_shift": selected_share - pool_share,
            "enrichment": coverage_ratio,
            "coverage_ratio": coverage_ratio,
        }
    return {
        "pool_size": pool_size,
        "selected_size": selected_size,
        "category_tv": 0.5 * category_tv,
        "worst_group_coverage": min(coverage_values) if coverage_values else None,
        "per_class": per_class,
    }


def uncertainty_group_dependence_report(
    rows: Iterable[dict[str, Any]],
    *,
    method: str,
    quantile_bins: int = 10,
    permutations: int = 999,
    seed: int = 42,
) -> dict[str, Any]:
    pool = [dict(row) for row in rows]
    if not pool:
        raise ValueError("rows must not be empty")
    if quantile_bins <= 1:
        raise ValueError("quantile_bins must be greater than one")
    if permutations <= 0:
        raise ValueError("permutations must be positive")
    if method not in {"entropy", "margin"}:
        raise ValueError(f"unsupported uncertainty method: {method}")

    actual_bins = min(quantile_bins, len(pool))
    scored = [
        (
            _uncertainty_score(row, method),
            str(row["id"]),
            str(int(row["label"])),
        )
        for row in pool
    ]
    scored.sort(key=lambda item: (item[0], item[1]))
    bins = [min(actual_bins - 1, (index * actual_bins) // len(scored)) for index in range(len(scored))]
    groups = [item[2] for item in scored]
    observed_mi = _mutual_information(groups, bins)
    group_entropy = _categorical_entropy(groups)

    rng = random.Random(seed)
    permuted = list(groups)
    permutation_values: list[float] = []
    for _ in range(permutations):
        rng.shuffle(permuted)
        permutation_values.append(_mutual_information(permuted, bins))
    permutation_mean = mean(permutation_values)
    denominator = group_entropy - permutation_mean
    adjusted = (observed_mi - permutation_mean) / denominator if denominator > 0.0 else 0.0
    permutation_p_value = (
        1 + sum(value >= observed_mi - 1e-15 for value in permutation_values)
    ) / (permutations + 1)
    return {
        "method": method,
        "pool_size": len(pool),
        "quantile_bins": actual_bins,
        "observed_mutual_information": observed_mi,
        "group_entropy": group_entropy,
        "permutation_mean_mutual_information": permutation_mean,
        "adjusted_uncertainty_coefficient": adjusted,
        "permutation_p_value": permutation_p_value,
        "permutations": permutations,
        "seed": seed,
    }


def classification_random_baseline(
    rows: Iterable[dict[str, Any]],
    *,
    budgets: Iterable[int],
    repetitions: int = 1000,
    seed: int = 42,
    quantile: float = 0.95,
) -> dict[str, Any]:
    pool = [dict(row) for row in rows]
    if not pool:
        raise ValueError("rows must not be empty")
    normalized_budgets = sorted(set(int(budget) for budget in budgets))
    if not normalized_budgets:
        raise ValueError("budgets must not be empty")
    if normalized_budgets[0] <= 0 or normalized_budgets[-1] > len(pool):
        raise ValueError(f"budgets must be between 1 and {len(pool)}")
    if repetitions <= 0:
        raise ValueError("repetitions must be positive")
    if not 0.0 < quantile < 1.0:
        raise ValueError("quantile must be between zero and one")

    labels = sorted({str(int(row["label"])) for row in pool}, key=int)
    pool_counts = Counter(str(int(row["label"])) for row in pool)
    pool_shares = {label: pool_counts[label] / len(pool) for label in labels}
    tv_by_budget = {budget: [] for budget in normalized_budgets}
    enrichment_by_budget = {
        budget: {label: [] for label in labels} for budget in normalized_budgets
    }
    rng = random.Random(seed)
    max_budget = normalized_budgets[-1]
    budget_set = set(normalized_budgets)
    trajectory_tvs: list[dict[int, float]] = []
    for _ in range(repetitions):
        selected_counts: Counter[str] = Counter()
        trajectory: dict[int, float] = {}
        for selected_size, index in enumerate(rng.sample(range(len(pool)), max_budget), start=1):
            selected_counts[str(int(pool[index]["label"]))] += 1
            if selected_size not in budget_set:
                continue
            tv = 0.5 * sum(
                abs(selected_counts[label] / selected_size - pool_shares[label]) for label in labels
            )
            tv_by_budget[selected_size].append(tv)
            trajectory[selected_size] = tv
            for label in labels:
                selected_share = selected_counts[label] / selected_size
                enrichment_by_budget[selected_size][label].append(
                    selected_share / pool_shares[label] if pool_shares[label] else 0.0
                )
        trajectory_tvs.append(trajectory)

    budget_reports: dict[str, Any] = {}
    tv_means: dict[int, float] = {}
    tv_stds: dict[int, float] = {}
    for budget in normalized_budgets:
        tv_values = tv_by_budget[budget]
        tv_mean = mean(tv_values)
        tv_std = pstdev(tv_values)
        tv_means[budget] = tv_mean
        tv_stds[budget] = tv_std
        budget_reports[str(budget)] = {
            "tv_mean": tv_mean,
            "tv_std": tv_std,
            "tv_q95": _quantile(tv_values, quantile),
            "per_class": {
                label: {
                    "enrichment_mean": mean(enrichment_by_budget[budget][label]),
                    "enrichment_q05": _quantile(enrichment_by_budget[budget][label], 1.0 - quantile),
                    "enrichment_q95": _quantile(enrichment_by_budget[budget][label], quantile),
                }
                for label in labels
            },
        }

    max_z_values = []
    for trajectory in trajectory_tvs:
        z_values = [
            (trajectory[budget] - tv_means[budget]) / tv_stds[budget]
            if tv_stds[budget] > 0.0
            else 0.0
            for budget in normalized_budgets
        ]
        max_z_values.append(max(z_values))
    return {
        "repetitions": repetitions,
        "seed": seed,
        "quantile": quantile,
        "nested_trajectories": True,
        "budgets": budget_reports,
        "global_envelope": {
            "max_tv_z_q95": _quantile(max_z_values, quantile),
        },
    }


def aggregate_dual_order_probability(
    *,
    probability_first_order_12: float,
    probability_first_order_21: float,
) -> float:
    first = _validate_probability(probability_first_order_12)
    reversed_first = _validate_probability(probability_first_order_21)
    return 0.5 * (first + (1.0 - reversed_first))


def preference_shift_report(
    rows: Iterable[dict[str, Any]],
    *,
    selected_ids: set[str],
) -> dict[str, Any]:
    pool = list(rows)
    selected = [row for row in pool if str(row["id"]) in selected_ids]
    if not selected and selected_ids:
        raise ValueError("selected_ids did not match any rows")
    length_getters = {
        "response_1_word_count": lambda row: row.get("response_1_word_count"),
        "response_2_word_count": lambda row: row.get("response_2_word_count"),
        "response_1_char_count": lambda row: row.get("response_1_char_count"),
        "response_2_char_count": lambda row: row.get("response_2_char_count"),
        "preferred_word_count": lambda row: _preferred_response_value(row, "word_count"),
        "rejected_word_count": lambda row: _rejected_response_value(row, "word_count"),
        "preferred_minus_rejected_word_count": lambda row: _preferred_minus_rejected_value(
            row, "word_count"
        ),
        "preferred_char_count": lambda row: _preferred_response_value(row, "char_count"),
        "rejected_char_count": lambda row: _rejected_response_value(row, "char_count"),
        "preferred_minus_rejected_char_count": lambda row: _preferred_minus_rejected_value(
            row, "char_count"
        ),
    }
    lengths = {
        field: _numeric_shift(pool, selected, getter)
        for field, getter in length_getters.items()
        if any(getter(row) is not None for row in pool)
    }
    attributes: dict[str, dict[str, dict[str, float | int | None]]] = {}
    attribute_names = sorted(
        {
            attribute
            for row in pool
            for response_key in ("response_1", "response_2")
            for attribute in ((row.get(f"{response_key}_attributes") or {}).keys())
        }
    )
    attribute_getters = {
        "response_1": lambda row, name: (row.get("response_1_attributes") or {}).get(name),
        "response_2": lambda row, name: (row.get("response_2_attributes") or {}).get(name),
        "preferred": _preferred_attribute,
        "rejected": _rejected_attribute,
        "preferred_minus_rejected": _preferred_minus_rejected_attribute,
    }
    for group, getter in attribute_getters.items():
        attributes[group] = {
            attribute: _numeric_shift(
                pool,
                selected,
                lambda row, name=attribute, getter=getter: getter(row, name),
            )
            for attribute in attribute_names
        }
    preference_direction = _categorical_shift(
        [str(row.get("preferred_response", 0)) for row in pool],
        [str(row.get("preferred_response", 0)) for row in selected],
    )
    prompt_distribution = _token_distribution_shift(
        [str(row.get("prompt", "")) for row in pool],
        [str(row.get("prompt", "")) for row in selected],
    )
    scoring_fields = ("probability_response_1", "entropy", "margin", "order_disagreement")
    scoring = {
        field: _numeric_shift(pool, selected, lambda row, field=field: row.get(field))
        for field in scoring_fields
        if any(row.get(field) is not None for row in pool)
    }
    return {
        "pool_size": len(pool),
        "selected_size": len(selected),
        "length": lengths,
        "attributes": attributes,
        "scoring": scoring,
        "preference_direction": preference_direction,
        "prompt_distribution": prompt_distribution,
    }


def preference_domain_effects(report: dict[str, Any]) -> dict[str, float]:
    length_effect = 0.0
    for values in report.get("length", {}).values():
        delta = abs(float(values.get("delta") or 0.0))
        scale = max(abs(float(values.get("pool_mean") or 0.0)), 1.0)
        length_effect = max(length_effect, delta / scale)

    attribute_effect = 0.0
    for attributes in report.get("attributes", {}).values():
        for values in attributes.values():
            attribute_effect = max(attribute_effect, abs(float(values.get("delta") or 0.0)))

    return {
        "length": length_effect,
        "attributes": attribute_effect,
        "preference_direction": abs(
            float(report.get("preference_direction", {}).get("tv") or 0.0)
        ),
        "prompt_distribution": abs(
            float(report.get("prompt_distribution", {}).get("token_js_divergence") or 0.0)
        ),
        "order_disagreement": abs(
            float(
                report.get("scoring", {})
                .get("order_disagreement", {})
                .get("delta")
                or 0.0
            )
        ),
    }


def preference_random_baseline(
    rows: Iterable[dict[str, Any]],
    *,
    budget: int,
    repetitions: int = 1000,
    seed: int = 42,
    quantile: float = 0.95,
) -> dict[str, Any]:
    pool = [dict(row) for row in rows]
    if not pool:
        raise ValueError("rows must not be empty")
    if budget <= 0 or budget > len(pool):
        raise ValueError(f"budget must be between 1 and {len(pool)}")
    if repetitions <= 0:
        raise ValueError("repetitions must be positive")
    if not 0.0 < quantile < 1.0:
        raise ValueError("quantile must be between zero and one")

    rng = random.Random(seed)
    domain_values = {
        domain: []
        for domain in (
            "length",
            "attributes",
            "preference_direction",
            "prompt_distribution",
            "order_disagreement",
        )
    }
    for _ in range(repetitions):
        selected_indices = rng.sample(range(len(pool)), budget)
        selected_ids = {str(pool[index]["id"]) for index in selected_indices}
        effects = preference_domain_effects(
            preference_shift_report(pool, selected_ids=selected_ids)
        )
        for domain, value in effects.items():
            domain_values[domain].append(value)
    return {
        "budget": budget,
        "repetitions": repetitions,
        "seed": seed,
        "quantile": quantile,
        "domains": {
            domain: {
                "mean": mean(values),
                "q95": _quantile(values, quantile),
            }
            for domain, values in domain_values.items()
        },
    }


def _preferred_response_value(row: dict[str, Any], suffix: str) -> Any:
    preferred = int(row.get("preferred_response", 0))
    return row.get(f"response_{preferred}_{suffix}") if preferred in (1, 2) else None


def _rejected_response_value(row: dict[str, Any], suffix: str) -> Any:
    preferred = int(row.get("preferred_response", 0))
    if preferred not in (1, 2):
        return None
    rejected = 2 if preferred == 1 else 1
    return row.get(f"response_{rejected}_{suffix}")


def _preferred_minus_rejected_value(row: dict[str, Any], suffix: str) -> float | None:
    preferred = _preferred_response_value(row, suffix)
    rejected = _rejected_response_value(row, suffix)
    if preferred is None or rejected is None:
        return None
    return float(preferred) - float(rejected)


def _preferred_attribute(row: dict[str, Any], attribute: str) -> Any:
    preferred = int(row.get("preferred_response", 0))
    if preferred not in (1, 2):
        return None
    return (row.get(f"response_{preferred}_attributes") or {}).get(attribute)


def _rejected_attribute(row: dict[str, Any], attribute: str) -> Any:
    preferred = int(row.get("preferred_response", 0))
    if preferred not in (1, 2):
        return None
    rejected = 2 if preferred == 1 else 1
    return (row.get(f"response_{rejected}_attributes") or {}).get(attribute)


def _preferred_minus_rejected_attribute(row: dict[str, Any], attribute: str) -> float | None:
    preferred = _preferred_attribute(row, attribute)
    rejected = _rejected_attribute(row, attribute)
    if preferred is None or rejected is None:
        return None
    return float(preferred) - float(rejected)


def _probabilities(row: dict[str, Any]) -> list[float]:
    values = [float(value) for value in row["probabilities"]]
    if len(values) < 2:
        raise ValueError("probabilities must contain at least two classes")
    total = sum(values)
    if total <= 0.0 or any(value < 0.0 for value in values):
        raise ValueError("probabilities must be non-negative and sum to a positive value")
    return [value / total for value in values]


def _uncertainty_score(row: dict[str, Any], method: str) -> float:
    probabilities = _probabilities(row)
    if method == "entropy":
        return _entropy(probabilities)
    if method == "margin":
        return 1.0 - _margin(probabilities)
    raise ValueError(f"unsupported uncertainty method: {method}")


def _entropy(probabilities: list[float]) -> float:
    return -sum(value * math.log(value) for value in probabilities if value > 0.0)


def _margin(probabilities: list[float]) -> float:
    first, second = sorted(probabilities, reverse=True)[:2]
    return first - second


def _validate_probability(value: float) -> float:
    probability = float(value)
    if not 0.0 <= probability <= 1.0:
        raise ValueError("probability must be between zero and one")
    return probability


def _numeric_shift(
    pool: list[dict[str, Any]],
    selected: list[dict[str, Any]],
    getter: Any,
) -> dict[str, float | int | None]:
    pool_values = [float(value) for row in pool if (value := getter(row)) is not None]
    selected_values = [float(value) for row in selected if (value := getter(row)) is not None]
    pool_mean = mean(pool_values) if pool_values else None
    selected_mean = mean(selected_values) if selected_values else None
    return {
        "pool_count": len(pool_values),
        "selected_count": len(selected_values),
        "pool_mean": pool_mean,
        "selected_mean": selected_mean,
        "delta": selected_mean - pool_mean if pool_mean is not None and selected_mean is not None else None,
        "pool_median": median(pool_values) if pool_values else None,
        "selected_median": median(selected_values) if selected_values else None,
    }


def _categorical_shift(pool_values: list[str], selected_values: list[str]) -> dict[str, Any]:
    pool_counts = Counter(pool_values)
    selected_counts = Counter(selected_values)
    categories = sorted(set(pool_counts) | set(selected_counts))
    pool_size = len(pool_values)
    selected_size = len(selected_values)
    tv = 0.0
    per_value: dict[str, dict[str, float | int | None]] = {}
    for category in categories:
        pool_share = pool_counts[category] / pool_size if pool_size else 0.0
        selected_share = selected_counts[category] / selected_size if selected_size else 0.0
        tv += abs(selected_share - pool_share)
        per_value[category] = {
            "pool_count": pool_counts[category],
            "selected_count": selected_counts[category],
            "pool_share": pool_share,
            "selected_share": selected_share,
            "enrichment": selected_share / pool_share if pool_share else None,
        }
    return {"tv": 0.5 * tv, "per_value": per_value}


def _token_distribution_shift(pool_prompts: list[str], selected_prompts: list[str]) -> dict[str, Any]:
    pool_counts = _token_counts(pool_prompts)
    selected_counts = _token_counts(selected_prompts)
    vocabulary = sorted(set(pool_counts) | set(selected_counts))
    if not vocabulary:
        return {"token_js_divergence": 0.0, "top_enriched_tokens": [], "top_depleted_tokens": []}
    smoothing = 1e-12
    pool_total = sum(pool_counts.values())
    selected_total = sum(selected_counts.values())
    pool_distribution = {
        token: (pool_counts[token] / pool_total if pool_total else 0.0) for token in vocabulary
    }
    selected_distribution = {
        token: (selected_counts[token] / selected_total if selected_total else 0.0) for token in vocabulary
    }
    midpoint = {
        token: 0.5 * (pool_distribution[token] + selected_distribution[token]) for token in vocabulary
    }
    js_divergence = 0.5 * _kl_divergence(pool_distribution, midpoint) + 0.5 * _kl_divergence(
        selected_distribution, midpoint
    )
    token_shifts = [
        {
            "token": token,
            "pool_share": pool_distribution[token],
            "selected_share": selected_distribution[token],
            "log_enrichment": math.log(
                (selected_distribution[token] + smoothing) / (pool_distribution[token] + smoothing)
            ),
        }
        for token in vocabulary
    ]
    return {
        "token_js_divergence": js_divergence,
        "top_enriched_tokens": sorted(token_shifts, key=lambda item: (-item["log_enrichment"], item["token"]))[:20],
        "top_depleted_tokens": sorted(token_shifts, key=lambda item: (item["log_enrichment"], item["token"]))[:20],
    }


def _token_counts(prompts: list[str]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for prompt in prompts:
        counts.update(re.findall(r"[a-z0-9]+", prompt.lower()))
    return counts


def _kl_divergence(left: dict[str, float], right: dict[str, float]) -> float:
    return sum(
        probability * math.log(probability / right[token])
        for token, probability in left.items()
        if probability > 0.0 and right[token] > 0.0
    )


def _categorical_entropy(values: list[str]) -> float:
    if not values:
        return 0.0
    counts = Counter(values)
    total = len(values)
    return -sum(
        (count / total) * math.log(count / total)
        for count in counts.values()
        if count
    )


def _mutual_information(groups: list[str], bins: list[int]) -> float:
    if len(groups) != len(bins):
        raise ValueError("groups and bins must have the same length")
    if not groups:
        return 0.0
    joint = Counter(zip(groups, bins, strict=True))
    group_counts = Counter(groups)
    bin_counts = Counter(bins)
    total = len(groups)
    return sum(
        (count / total)
        * math.log((count * total) / (group_counts[group] * bin_counts[bin_index]))
        for (group, bin_index), count in joint.items()
        if count
    )


def _quantile(values: list[float], probability: float) -> float:
    if not values:
        raise ValueError("values must not be empty")
    if not 0.0 <= probability <= 1.0:
        raise ValueError("probability must be between zero and one")
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    position = probability * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight
