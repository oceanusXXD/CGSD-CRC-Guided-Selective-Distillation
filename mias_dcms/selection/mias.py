from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
import math
import random
from typing import Any

import numpy as np
from scipy.optimize import minimize, minimize_scalar

from mias_dcms.preference_pool import length_gap_bin
from mias_dcms.prompt_clusters import build_prompt_cluster_assignments
from mias_dcms.selection.dcms import DCMSResult, solve_dcms_with_slack
from mias_dcms.selectors import assert_selector_rows_are_label_safe, select_top_budget


DEFAULT_L2_GRID = (1e-3, 1e-2, 1e-1, 1.0, 10.0)
DEFAULT_SLACK_GRID = (0.0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5)
DEFAULT_KAPPA = 0.1


@dataclass(frozen=True)
class MIASScoringResult:
    class_values: list[Any]
    split_ids: dict[str, list[str]]
    weights: list[Any]
    bias: list[float]
    feature_mean: list[float]
    feature_scale: list[float]
    l2: float
    l2_selection_source: str
    temperature: float
    temperature_status: str
    calibration_nll_before: float | None
    calibration_nll_after: float | None
    validation_gradient: list[float]
    posterior: dict[str, list[float]]
    posterior_lower: dict[str, list[float]]
    posterior_upper: dict[str, list[float]]
    per_label_influence: dict[str, list[float]]
    utility: dict[str, float]
    costs: dict[str, float]
    bootstrap_status: str
    bootstrap_heads_fitted: int
    optimizer_status: str = "converged"
    auxiliary_models: dict[str, Any] = field(default_factory=dict)

    def model_dict(self) -> dict[str, Any]:
        return {
            "algorithm": "MIAS",
            "class_values": list(self.class_values),
            "split_ids": {key: list(value) for key, value in self.split_ids.items()},
            "weights": self.weights,
            "bias": list(self.bias),
            "feature_mean": list(self.feature_mean),
            "feature_scale": list(self.feature_scale),
            "l2": self.l2,
            "l2_selection_source": self.l2_selection_source,
            "temperature": self.temperature,
            "temperature_status": self.temperature_status,
            "calibration_nll_before": self.calibration_nll_before,
            "calibration_nll_after": self.calibration_nll_after,
            "validation_gradient": list(self.validation_gradient),
            "bootstrap_status": self.bootstrap_status,
            "bootstrap_heads_fitted": self.bootstrap_heads_fitted,
            "optimizer_status": self.optimizer_status,
            "auxiliary_models": dict(self.auxiliary_models),
        }

    def score_rows(self, selected_ids: Sequence[str] = ()) -> list[dict[str, Any]]:
        selected = {str(sample_id) for sample_id in selected_ids}
        return [
            {
                "sample_id": sample_id,
                "posterior": list(self.posterior[sample_id]),
                "posterior_lower": list(self.posterior_lower[sample_id]),
                "posterior_upper": list(self.posterior_upper[sample_id]),
                "per_label_influence": list(self.per_label_influence[sample_id]),
                "expected_influence_utility": float(self.utility[sample_id]),
                "selection_cost": float(self.costs[sample_id]),
                "selected": int(sample_id in selected),
            }
            for sample_id in self.posterior
        ]


@dataclass(frozen=True)
class MIASSelectionResult:
    selected_ids: list[str]
    scoring: MIASScoringResult
    group_membership: dict[str, dict[str, float]] = field(default_factory=dict)
    membership_lower: dict[str, dict[str, float]] = field(default_factory=dict)
    membership_upper: dict[str, dict[str, float]] = field(default_factory=dict)
    target_moments: dict[str, float] = field(default_factory=dict)
    dcms: DCMSResult | None = None

    def summary_dict(self, *, method: str, budget: int) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "algorithm": "MIAS-DCMS" if self.dcms is not None else "MIAS",
            "method": method,
            "budget": int(budget),
            "selected_count": len(self.selected_ids),
            "selected_ids": list(self.selected_ids),
            "pool_size": len(self.scoring.posterior),
            "utility": "calibrated_expected_positive_validation_influence_per_cost",
            "candidate_scope": "complete_unlabeled_pool",
            "selector_model": self.scoring.model_dict(),
            "target_moments": dict(self.target_moments),
        }
        if self.dcms is not None:
            dcms_metadata = _dcms_metadata(self.dcms)
            payload.update(dcms_metadata)
            payload["dcms"] = dict(dcms_metadata)
        else:
            selected = set(self.selected_ids)
            payload.update(
                {
                    "continuous_moments": {},
                    "rounded_moments": {},
                    "robust_lower_moments": {},
                    "robust_upper_moments": {},
                    "utility_retained": 1.0,
                    "max_constraint_violation": 0.0,
                    "solver_status": "top_utility",
                    "selected_slack": None,
                    "rounding_seed": None,
                    "q_propensity": {
                        sample_id: float(sample_id in selected)
                        for sample_id in self.scoring.posterior
                    },
                    "selection_indicator": {
                        sample_id: int(sample_id in selected)
                        for sample_id in self.scoring.posterior
                    },
                    "slack_trace": [],
                }
            )
        return payload


def deterministic_stratified_split(
    sample_ids: Sequence[str],
    labels: Sequence[int],
    *,
    seed: int,
) -> dict[str, list[int]]:
    """Split seed labels into disjoint 60/20/20 fit/calibration/meta sets."""
    ids = [str(sample_id) for sample_id in sample_ids]
    targets = _split_counts(len(ids))
    if len(ids) != len(labels):
        raise ValueError("sample_ids and labels must have equal length")
    if len(set(ids)) != len(ids):
        raise ValueError("seed sample_ids must be unique")

    names = ("fit", "calibration", "meta_validation")
    remaining = dict(zip(names, targets, strict=True))
    assigned: dict[str, list[int]] = {name: [] for name in names}
    per_class: dict[int, dict[str, int]] = {}
    grouped: dict[int, list[int]] = {}
    for index, label in enumerate(labels):
        grouped.setdefault(int(label), []).append(index)
    rng = random.Random(int(seed))
    for label in sorted(grouped):
        indexes = list(grouped[label])
        rng.shuffle(indexes)
        per_class[label] = {name: 0 for name in names}
        for index in indexes:
            available = [name for name in names if remaining[name] > 0]
            if not available:
                raise AssertionError("split capacity exhausted before all seed rows were assigned")
            chosen = max(
                available,
                key=lambda name: (
                    (targets[names.index(name)] * len(indexes) / len(ids))
                    - per_class[label][name],
                    remaining[name],
                    -names.index(name),
                ),
            )
            assigned[chosen].append(index)
            per_class[label][chosen] += 1
            remaining[chosen] -= 1
    if any(remaining.values()):
        raise AssertionError("seed split did not fill every partition")
    return {name: sorted(indexes, key=lambda index: ids[index]) for name, indexes in assigned.items()}


def score_expected_validation_influence(
    *,
    seed_ids: Sequence[str],
    seed_features: Sequence[Sequence[float]],
    seed_labels: Sequence[int],
    candidate_ids: Sequence[str],
    candidate_features: Sequence[Sequence[float]],
    costs: Sequence[float] | None = None,
    class_values: Sequence[Any] | None = None,
    add_intercept: bool = True,
    center_features: bool = True,
    seed: int = 42,
    l2_grid: Sequence[float] = DEFAULT_L2_GRID,
    bootstrap_heads: int = 20,
) -> MIASScoringResult:
    """Fit the seed-only surrogate and score every unlabeled candidate.

    Candidate labels are deliberately absent from this API. The utility is
    E_y[max(0, grad(L_meta) dot grad(loss_i(y)))] divided by annotation cost.
    """
    seed_id_values = [str(value) for value in seed_ids]
    candidate_id_values = [str(value) for value in candidate_ids]
    if set(seed_id_values).intersection(candidate_id_values):
        raise ValueError("seed and candidate ids must be disjoint")
    if len(set(candidate_id_values)) != len(candidate_id_values):
        raise ValueError("candidate ids must be unique")
    if len(seed_id_values) < 3:
        raise ValueError("MIAS requires at least three seed labels for the 60/20/20 split")

    x_seed = _feature_matrix(seed_features, name="seed_features")
    x_candidates = _feature_matrix(candidate_features, name="candidate_features")
    if x_seed.shape[0] != len(seed_id_values) or x_candidates.shape[0] != len(candidate_id_values):
        raise ValueError("feature row counts must match their ids")
    if x_seed.shape[1] != x_candidates.shape[1]:
        raise ValueError("seed and candidate features must have the same dimension")
    y_seed = np.asarray([int(label) for label in seed_labels], dtype=np.int64)
    if y_seed.shape[0] != len(seed_id_values):
        raise ValueError("seed_labels must match seed_ids")
    resolved_class_values = list(class_values) if class_values is not None else list(range(int(y_seed.max()) + 1))
    class_count = len(resolved_class_values)
    if class_count < 2 or np.any(y_seed < 0) or np.any(y_seed >= class_count):
        raise ValueError("seed labels must index at least two declared classes")
    if len(np.unique(y_seed)) < 2:
        raise ValueError("MIAS requires seed labels from at least two observed classes")
    cost_values = np.ones(len(candidate_id_values), dtype=np.float64)
    if costs is not None:
        cost_values = np.asarray([float(value) for value in costs], dtype=np.float64)
    if (
        cost_values.shape != (len(candidate_id_values),)
        or np.any(~np.isfinite(cost_values))
        or np.any(cost_values <= 0)
    ):
        raise ValueError("candidate costs must be finite and positive")
    l2_values = tuple(float(value) for value in l2_grid)
    if not l2_values or any(not math.isfinite(value) or value < 0 for value in l2_values):
        raise ValueError("l2_grid must contain finite non-negative values")

    split = deterministic_stratified_split(seed_id_values, y_seed.tolist(), seed=seed)
    fit_indexes = np.asarray(split["fit"], dtype=np.int64)
    calibration_indexes = np.asarray(split["calibration"], dtype=np.int64)
    meta_indexes = np.asarray(split["meta_validation"], dtype=np.int64)
    mean, scale = _feature_transform(x_seed[fit_indexes], center=center_features)
    xs = _apply_feature_transform(x_seed, mean=mean, scale=scale)
    xc = _apply_feature_transform(x_candidates, mean=mean, scale=scale)

    selected_l2, params, l2_source, optimizer_status = _select_l2_and_fit(
        xs,
        y_seed,
        fit_indexes=fit_indexes,
        calibration_indexes=calibration_indexes,
        class_count=class_count,
        add_intercept=add_intercept,
        l2_grid=l2_values,
    )
    calibration_logits = _logits(xs[calibration_indexes], params, class_count, add_intercept)
    temperature, temperature_status = _fit_temperature(
        calibration_logits,
        y_seed[calibration_indexes],
        class_count=class_count,
    )
    nll_before = _maybe_nll(calibration_logits, y_seed[calibration_indexes], temperature=1.0)
    nll_after = _maybe_nll(calibration_logits, y_seed[calibration_indexes], temperature=temperature)

    meta_logits = _logits(xs[meta_indexes], params, class_count, add_intercept)
    meta_probabilities = _probabilities_from_logits(meta_logits, temperature=temperature)
    validation_gradient = _mean_parameter_gradient(
        xs[meta_indexes],
        y_seed[meta_indexes],
        meta_probabilities,
        class_count=class_count,
        add_intercept=add_intercept,
        temperature=temperature,
    )
    candidate_logits = _logits(xc, params, class_count, add_intercept)
    candidate_probabilities = _probabilities_from_logits(candidate_logits, temperature=temperature)
    influences, utilities = _candidate_influences(
        xc,
        candidate_probabilities,
        validation_gradient,
        cost_values,
        class_count=class_count,
        add_intercept=add_intercept,
        temperature=temperature,
    )

    bootstrap_probabilities, bootstrap_status = _bootstrap_posteriors(
        xs=xs,
        y=y_seed,
        xc=xc,
        fit_indexes=fit_indexes,
        calibration_indexes=calibration_indexes,
        class_count=class_count,
        add_intercept=add_intercept,
        l2=selected_l2,
        head_count=int(bootstrap_heads),
        seed=int(seed),
    )
    if bootstrap_probabilities:
        stack = np.stack(bootstrap_probabilities, axis=0)
        lower = np.quantile(stack, 0.05, axis=0)
        upper = np.quantile(stack, 0.95, axis=0)
    else:
        lower = candidate_probabilities.copy()
        upper = candidate_probabilities.copy()

    weights, bias = _unpack_serializable(params, class_count=class_count, add_intercept=add_intercept)
    split_ids = {
        name: [seed_id_values[index] for index in indexes]
        for name, indexes in split.items()
    }
    return MIASScoringResult(
        class_values=resolved_class_values,
        split_ids=split_ids,
        weights=weights,
        bias=bias,
        feature_mean=mean.tolist(),
        feature_scale=scale.tolist(),
        l2=float(selected_l2),
        l2_selection_source=l2_source,
        temperature=float(temperature),
        temperature_status=temperature_status,
        calibration_nll_before=nll_before,
        calibration_nll_after=nll_after,
        validation_gradient=validation_gradient.tolist(),
        posterior={
            sample_id: candidate_probabilities[index].tolist()
            for index, sample_id in enumerate(candidate_id_values)
        },
        posterior_lower={sample_id: lower[index].tolist() for index, sample_id in enumerate(candidate_id_values)},
        posterior_upper={sample_id: upper[index].tolist() for index, sample_id in enumerate(candidate_id_values)},
        per_label_influence={
            sample_id: influences[index].tolist()
            for index, sample_id in enumerate(candidate_id_values)
        },
        utility={sample_id: float(utilities[index]) for index, sample_id in enumerate(candidate_id_values)},
        costs={sample_id: float(cost_values[index]) for index, sample_id in enumerate(candidate_id_values)},
        bootstrap_status=bootstrap_status,
        bootstrap_heads_fitted=len(bootstrap_probabilities),
        optimizer_status=optimizer_status,
    )


def select_mias_classification(
    *,
    seed_rows: Sequence[Mapping[str, Any]],
    candidate_rows: Sequence[Mapping[str, Any]],
    budget: int,
    seed: int,
    use_dcms: bool,
    label_field: str = "label",
    bootstrap_heads: int = 20,
    semantic_cluster_count: int | None = None,
    slack_grid: Sequence[float] = DEFAULT_SLACK_GRID,
    kappa: float = DEFAULT_KAPPA,
) -> MIASSelectionResult:
    candidates = [dict(row) for row in candidate_rows]
    assert_selector_rows_are_label_safe(candidates)
    seed_values = [dict(row) for row in seed_rows]
    candidate_ids = [_row_id(row) for row in candidates]
    seed_ids = [_row_id(row) for row in seed_values]
    raw_labels = [_required_seed_label(row, label_field=label_field) for row in seed_values]
    class_values, labels = _classification_label_indexes(raw_labels, candidates)
    scoring = score_expected_validation_influence(
        seed_ids=seed_ids,
        seed_features=[_classification_feature(row) for row in seed_values],
        seed_labels=labels,
        candidate_ids=candidate_ids,
        candidate_features=[_classification_feature(row) for row in candidates],
        class_values=class_values,
        add_intercept=True,
        center_features=True,
        seed=seed,
        bootstrap_heads=bootstrap_heads,
    )
    if not use_dcms:
        selected_ids = select_top_budget(
            sample_ids=candidate_ids,
            scores=[scoring.utility[sample_id] for sample_id in candidate_ids],
            budget=budget,
        )
        return MIASSelectionResult(selected_ids=selected_ids, scoring=scoring)

    semantic = _classification_semantic_memberships(
        candidates,
        budget=budget,
        cluster_count=semantic_cluster_count,
    )
    length_groups = _classification_length_memberships(candidates)
    memberships: list[dict[str, float]] = []
    lowers: list[dict[str, float]] = []
    uppers: list[dict[str, float]] = []
    for index, sample_id in enumerate(candidate_ids):
        exact = {**semantic[index], **length_groups[index]}
        nominal = {
            **{f"class={class_values[c]}": value for c, value in enumerate(scoring.posterior[sample_id])},
            **exact,
        }
        lower = {
            **{f"class={class_values[c]}": value for c, value in enumerate(scoring.posterior_lower[sample_id])},
            **exact,
        }
        upper = {
            **{f"class={class_values[c]}": value for c, value in enumerate(scoring.posterior_upper[sample_id])},
            **exact,
        }
        memberships.append(nominal)
        lowers.append(lower)
        uppers.append(upper)
    return _solve_mias_dcms(
        scoring=scoring,
        sample_ids=candidate_ids,
        memberships=memberships,
        lowers=lowers,
        uppers=uppers,
        budget=budget,
        seed=seed,
        slack_grid=slack_grid,
        kappa=kappa,
    )


def select_mias_preference(
    *,
    seed_rows: Sequence[Mapping[str, Any]],
    candidate_rows: Sequence[Mapping[str, Any]],
    budget: int,
    seed: int,
    use_dcms: bool,
    bootstrap_heads: int = 20,
    slack_grid: Sequence[float] = DEFAULT_SLACK_GRID,
    kappa: float = DEFAULT_KAPPA,
) -> MIASSelectionResult:
    candidates = [dict(row) for row in candidate_rows]
    assert_selector_rows_are_label_safe(candidates)
    seeds = [dict(row) for row in seed_rows]
    candidate_ids = [_row_id(row) for row in candidates]
    seed_ids = [_row_id(row) for row in seeds]
    direction_seed_indexes = [
        index
        for index, row in enumerate(seeds)
        if _preference_label_or_none(row) is not None
    ]
    if len(direction_seed_indexes) < 3:
        raise ValueError("MIAS preference direction scoring requires at least three non-tie seed pairs")
    direction_seed_ids = [seed_ids[index] for index in direction_seed_indexes]
    direction_seed_rows = [seeds[index] for index in direction_seed_indexes]
    direction_scoring = score_expected_validation_influence(
        seed_ids=direction_seed_ids,
        seed_features=[preference_difference_feature(row) for row in direction_seed_rows],
        seed_labels=[_preference_label(row) for row in direction_seed_rows],
        candidate_ids=candidate_ids,
        candidate_features=[preference_difference_feature(row) for row in candidates],
        costs=[_preference_cost(row) for row in candidates],
        class_values=["B", "A"],
        add_intercept=False,
        center_features=False,
        seed=seed,
        bootstrap_heads=bootstrap_heads,
    )
    scoring = direction_scoring
    if len(direction_seed_indexes) != len(seeds):
        trainability_scoring = score_expected_validation_influence(
            seed_ids=seed_ids,
            seed_features=[preference_trainability_feature(row) for row in seeds],
            seed_labels=[int(_preference_label_or_none(row) is not None) for row in seeds],
            candidate_ids=candidate_ids,
            candidate_features=[preference_trainability_feature(row) for row in candidates],
            class_values=["tie", "non_tie"],
            add_intercept=True,
            center_features=True,
            seed=seed,
            bootstrap_heads=bootstrap_heads,
        )
        scoring = _combine_preference_direction_and_trainability(
            direction_scoring,
            trainability_scoring,
        )
    if not use_dcms:
        selected_ids = select_top_budget(
            sample_ids=candidate_ids,
            scores=[scoring.utility[sample_id] for sample_id in candidate_ids],
            budget=budget,
        )
        return MIASSelectionResult(selected_ids=selected_ids, scoring=scoring)

    memberships: list[dict[str, float]] = []
    lowers: list[dict[str, float]] = []
    uppers: list[dict[str, float]] = []
    for row, sample_id in zip(candidates, candidate_ids, strict=True):
        exact = _preference_exact_memberships(row)
        posterior = _posterior_by_class(scoring.posterior[sample_id], scoring.class_values)
        lower_posterior = _posterior_by_class(scoring.posterior_lower[sample_id], scoring.class_values)
        upper_posterior = _posterior_by_class(scoring.posterior_upper[sample_id], scoring.class_values)
        nominal = {
            "preference=B": posterior["B"],
            "preference=A": posterior["A"],
            **exact,
        }
        lower = {
            "preference=B": lower_posterior["B"],
            "preference=A": lower_posterior["A"],
            **exact,
        }
        upper = {
            "preference=B": upper_posterior["B"],
            "preference=A": upper_posterior["A"],
            **exact,
        }
        memberships.append(nominal)
        lowers.append(lower)
        uppers.append(upper)
    return _solve_mias_dcms(
        scoring=scoring,
        sample_ids=candidate_ids,
        memberships=memberships,
        lowers=lowers,
        uppers=uppers,
        budget=budget,
        seed=seed,
        slack_grid=slack_grid,
        kappa=kappa,
    )


def preference_difference_feature(row: Mapping[str, Any]) -> list[float]:
    """Return [h(prompt,A)-h(prompt,B), signed normalized length gap]."""
    a = _embedding(row, "response_a_embedding", "response_1_embedding")
    b = _embedding(row, "response_b_embedding", "response_2_embedding")
    if a.shape != b.shape:
        raise ValueError("response A/B embeddings must have the same dimension")
    length_a, length_b = _response_lengths(row)
    signed_gap = (length_a - length_b) / max(1.0, length_a + length_b)
    return np.concatenate((a - b, np.asarray([signed_gap], dtype=np.float64))).tolist()


def preference_trainability_feature(row: Mapping[str, Any]) -> list[float]:
    """Return a response-order-invariant feature for predicting DPO-usable pairs."""
    return np.abs(np.asarray(preference_difference_feature(row), dtype=np.float64)).tolist()


def _combine_preference_direction_and_trainability(
    direction: MIASScoringResult,
    trainability: MIASScoringResult,
) -> MIASScoringResult:
    if set(direction.posterior) != set(trainability.posterior):
        raise ValueError("direction and trainability heads must score the same candidate ids")
    posterior: dict[str, list[float]] = {}
    lower: dict[str, list[float]] = {}
    upper: dict[str, list[float]] = {}
    influences: dict[str, list[float]] = {}
    utility: dict[str, float] = {}
    for sample_id in direction.posterior:
        direction_values = _posterior_by_class(direction.posterior[sample_id], direction.class_values)
        direction_lower = _posterior_by_class(
            direction.posterior_lower[sample_id], direction.class_values
        )
        direction_upper = _posterior_by_class(
            direction.posterior_upper[sample_id], direction.class_values
        )
        trainability_values = _values_by_class(
            trainability.posterior[sample_id], trainability.class_values
        )
        trainability_lower = _values_by_class(
            trainability.posterior_lower[sample_id], trainability.class_values
        )
        trainability_upper = _values_by_class(
            trainability.posterior_upper[sample_id], trainability.class_values
        )
        non_tie = trainability_values["non_tie"]
        non_tie_lower = trainability_lower["non_tie"]
        non_tie_upper = trainability_upper["non_tie"]
        posterior[sample_id] = [
            1.0 - non_tie,
            non_tie * direction_values["B"],
            non_tie * direction_values["A"],
        ]
        lower[sample_id] = [
            max(0.0, 1.0 - non_tie_upper),
            non_tie_lower * direction_lower["B"],
            non_tie_lower * direction_lower["A"],
        ]
        upper[sample_id] = [
            min(1.0, 1.0 - non_tie_lower),
            non_tie_upper * direction_upper["B"],
            non_tie_upper * direction_upper["A"],
        ]
        direction_influences = _posterior_by_class(
            direction.per_label_influence[sample_id], direction.class_values
        )
        influences[sample_id] = [
            0.0,
            non_tie * direction_influences["B"],
            non_tie * direction_influences["A"],
        ]
        utility[sample_id] = non_tie * direction.utility[sample_id]

    bootstrap_heads_fitted = min(
        direction.bootstrap_heads_fitted,
        trainability.bootstrap_heads_fitted,
    )
    return MIASScoringResult(
        class_values=["tie", "B", "A"],
        split_ids=dict(direction.split_ids),
        weights=direction.weights,
        bias=direction.bias,
        feature_mean=direction.feature_mean,
        feature_scale=direction.feature_scale,
        l2=direction.l2,
        l2_selection_source=direction.l2_selection_source,
        temperature=direction.temperature,
        temperature_status=direction.temperature_status,
        calibration_nll_before=direction.calibration_nll_before,
        calibration_nll_after=direction.calibration_nll_after,
        validation_gradient=direction.validation_gradient,
        posterior=posterior,
        posterior_lower=lower,
        posterior_upper=upper,
        per_label_influence=influences,
        utility=utility,
        costs=dict(direction.costs),
        bootstrap_status=(
            f"direction={direction.bootstrap_status};trainability={trainability.bootstrap_status}"
        ),
        bootstrap_heads_fitted=bootstrap_heads_fitted,
        optimizer_status=(
            f"direction={direction.optimizer_status};trainability={trainability.optimizer_status}"
        ),
        auxiliary_models={
            "preference_trainability": {
                "utility": "non_tie_probability",
                "model": trainability.model_dict(),
            }
        },
    )


def _posterior_by_class(values: Sequence[float], classes: Sequence[Any]) -> dict[str, float]:
    output = _values_by_class(values, classes)
    required = {"A", "B"}
    if not required.issubset(output):
        raise ValueError("preference posterior must include A and B classes")
    return output


def _values_by_class(values: Sequence[float], classes: Sequence[Any]) -> dict[str, float]:
    if len(values) != len(classes):
        raise ValueError("posterior values must match declared class values")
    return {str(class_value): float(value) for class_value, value in zip(classes, values, strict=True)}


def _split_counts(sample_count: int) -> tuple[int, int, int]:
    if sample_count < 3:
        raise ValueError("at least three seed labels are required for a 60/20/20 split")
    raw = np.asarray([0.6, 0.2, 0.2], dtype=np.float64) * sample_count
    counts = np.floor(raw).astype(int)
    counts = np.maximum(counts, 1)
    while int(counts.sum()) > sample_count:
        removable = [index for index in range(3) if counts[index] > 1]
        index = min(removable, key=lambda value: (raw[value] - counts[value], value))
        counts[index] -= 1
    while int(counts.sum()) < sample_count:
        index = max(range(3), key=lambda value: (raw[value] - counts[value], -value))
        counts[index] += 1
    return tuple(int(value) for value in counts)


def _feature_matrix(values: Sequence[Sequence[float]], *, name: str) -> np.ndarray:
    matrix = np.asarray(values, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[1] == 0:
        raise ValueError(f"{name} must be a non-empty rank-2 matrix")
    if np.any(~np.isfinite(matrix)):
        raise ValueError(f"{name} must contain only finite values")
    return matrix


def _feature_transform(x_fit: np.ndarray, *, center: bool) -> tuple[np.ndarray, np.ndarray]:
    mean = x_fit.mean(axis=0) if center else np.zeros(x_fit.shape[1], dtype=np.float64)
    centered = x_fit - mean
    scale = np.sqrt(np.mean(centered * centered, axis=0))
    positive = scale[scale > 1e-8]
    relative_floor = max(1e-6, float(np.median(positive)) * 0.1) if len(positive) else 1.0
    scale = np.where(scale >= relative_floor, scale, relative_floor)
    return mean, scale


def _apply_feature_transform(x: np.ndarray, *, mean: np.ndarray, scale: np.ndarray) -> np.ndarray:
    return (x - mean) / scale


def _parameter_size(feature_count: int, class_count: int, add_intercept: bool) -> int:
    if class_count == 2:
        return feature_count + int(add_intercept)
    return class_count * feature_count + class_count * int(add_intercept)


def _select_l2_and_fit(
    x: np.ndarray,
    y: np.ndarray,
    *,
    fit_indexes: np.ndarray,
    calibration_indexes: np.ndarray,
    class_count: int,
    add_intercept: bool,
    l2_grid: Sequence[float],
) -> tuple[float, np.ndarray, str, str]:
    if len(calibration_indexes) < 2:
        selected_l2 = min(l2_grid, key=lambda value: (abs(math.log10(max(value, 1e-12))), value))
        params, status = _fit_head(
            x[fit_indexes],
            y[fit_indexes],
            class_count=class_count,
            add_intercept=add_intercept,
            l2=selected_l2,
        )
        return float(selected_l2), params, "default_fallback", status
    evaluation_indexes = calibration_indexes
    source = "calibration"
    best: tuple[float, float, np.ndarray, str] | None = None
    initial = np.zeros(_parameter_size(x.shape[1], class_count, add_intercept), dtype=np.float64)
    for l2 in l2_grid:
        params, status = _fit_head(
            x[fit_indexes],
            y[fit_indexes],
            class_count=class_count,
            add_intercept=add_intercept,
            l2=l2,
            initial=initial,
        )
        score = _nll(
            _logits(x[evaluation_indexes], params, class_count, add_intercept),
            y[evaluation_indexes],
            temperature=1.0,
        )
        candidate = (float(score), float(l2), params, status)
        if best is None or candidate[:2] < best[:2]:
            best = candidate
        initial = params
    assert best is not None
    return best[1], best[2], source, best[3]


def _fit_head(
    x: np.ndarray,
    y: np.ndarray,
    *,
    class_count: int,
    add_intercept: bool,
    l2: float,
    initial: np.ndarray | None = None,
) -> tuple[np.ndarray, str]:
    start = initial
    if start is None:
        start = np.zeros(_parameter_size(x.shape[1], class_count, add_intercept), dtype=np.float64)

    def objective(params: np.ndarray) -> tuple[float, np.ndarray]:
        logits = _logits(x, params, class_count, add_intercept)
        probabilities = _probabilities_from_logits(logits, temperature=1.0)
        loss = _nll(logits, y, temperature=1.0)
        gradient = _mean_parameter_gradient(
            x,
            y,
            probabilities,
            class_count=class_count,
            add_intercept=add_intercept,
            temperature=1.0,
        )
        weight_size = x.shape[1] if class_count == 2 else class_count * x.shape[1]
        loss += 0.5 * float(l2) * float(np.dot(params[:weight_size], params[:weight_size]))
        gradient[:weight_size] += float(l2) * params[:weight_size]
        return float(loss), gradient

    result = minimize(objective, start, method="L-BFGS-B", jac=True, options={"maxiter": 300, "ftol": 1e-10})
    if not np.all(np.isfinite(result.x)):
        raise RuntimeError("surrogate head optimization produced non-finite parameters")
    status = "converged" if result.success else f"stopped:{result.message}"
    return np.asarray(result.x, dtype=np.float64), status


def _logits(x: np.ndarray, params: np.ndarray, class_count: int, add_intercept: bool) -> np.ndarray:
    feature_count = x.shape[1]
    if class_count == 2:
        values = x @ params[:feature_count]
        if add_intercept:
            values = values + params[feature_count]
        return values[:, None]
    weight_size = class_count * feature_count
    weights = params[:weight_size].reshape(class_count, feature_count)
    values = x @ weights.T
    if add_intercept:
        values = values + params[weight_size:]
    return values


def _probabilities_from_logits(logits: np.ndarray, *, temperature: float) -> np.ndarray:
    scaled = logits / float(temperature)
    if scaled.shape[1] == 1:
        positive = _sigmoid(scaled[:, 0])
        return np.column_stack((1.0 - positive, positive))
    shifted = scaled - scaled.max(axis=1, keepdims=True)
    exponentials = np.exp(shifted)
    return exponentials / exponentials.sum(axis=1, keepdims=True)


def _sigmoid(values: np.ndarray) -> np.ndarray:
    output = np.empty_like(values, dtype=np.float64)
    positive = values >= 0
    output[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
    exponential = np.exp(values[~positive])
    output[~positive] = exponential / (1.0 + exponential)
    return output


def _nll(logits: np.ndarray, labels: np.ndarray, *, temperature: float) -> float:
    probabilities = _probabilities_from_logits(logits, temperature=temperature)
    selected = probabilities[np.arange(len(labels)), labels]
    return float(-np.log(np.clip(selected, 1e-12, 1.0)).mean())


def _maybe_nll(logits: np.ndarray, labels: np.ndarray, *, temperature: float) -> float | None:
    return _nll(logits, labels, temperature=temperature) if len(labels) else None


def _fit_temperature(
    logits: np.ndarray,
    labels: np.ndarray,
    *,
    class_count: int,
) -> tuple[float, str]:
    if len(labels) < 4 or len(set(int(value) for value in labels)) < 2:
        return 1.0, "insufficient_data"
    result = minimize_scalar(
        lambda log_temperature: _nll(logits, labels, temperature=math.exp(float(log_temperature))),
        bounds=(math.log(0.05), math.log(20.0)),
        method="bounded",
        options={"xatol": 1e-6},
    )
    if not result.success or not math.isfinite(float(result.fun)):
        return 1.0, "optimization_failed"
    return float(math.exp(float(result.x))), "calibrated"


def _mean_parameter_gradient(
    x: np.ndarray,
    labels: np.ndarray,
    probabilities: np.ndarray,
    *,
    class_count: int,
    add_intercept: bool,
    temperature: float,
) -> np.ndarray:
    residual = probabilities.copy()
    residual[np.arange(len(labels)), labels] -= 1.0
    residual /= float(temperature)
    if class_count == 2:
        positive_residual = residual[:, 1]
        parts = [np.mean(positive_residual[:, None] * x, axis=0)]
        if add_intercept:
            parts.append(np.asarray([positive_residual.mean()]))
        return np.concatenate(parts)
    weight_gradient = (residual.T @ x) / len(x)
    parts = [weight_gradient.reshape(-1)]
    if add_intercept:
        parts.append(residual.mean(axis=0))
    return np.concatenate(parts)


def _single_label_gradient(
    x: np.ndarray,
    probability: np.ndarray,
    label: int,
    *,
    class_count: int,
    add_intercept: bool,
    temperature: float,
) -> np.ndarray:
    residual = probability.copy()
    residual[label] -= 1.0
    residual /= float(temperature)
    if class_count == 2:
        parts = [residual[1] * x]
        if add_intercept:
            parts.append(np.asarray([residual[1]]))
        return np.concatenate(parts)
    parts = [(residual[:, None] * x[None, :]).reshape(-1)]
    if add_intercept:
        parts.append(residual)
    return np.concatenate(parts)


def _candidate_influences(
    x: np.ndarray,
    probabilities: np.ndarray,
    validation_gradient: np.ndarray,
    costs: np.ndarray,
    *,
    class_count: int,
    add_intercept: bool,
    temperature: float,
) -> tuple[np.ndarray, np.ndarray]:
    influences = np.zeros_like(probabilities)
    for index in range(len(x)):
        for label in range(class_count):
            gradient = _single_label_gradient(
                x[index],
                probabilities[index],
                label,
                class_count=class_count,
                add_intercept=add_intercept,
                temperature=temperature,
            )
            influences[index, label] = max(0.0, float(np.dot(validation_gradient, gradient)))
    utilities = np.sum(probabilities * influences, axis=1) / costs
    return influences, utilities


def _bootstrap_posteriors(
    *,
    xs: np.ndarray,
    y: np.ndarray,
    xc: np.ndarray,
    fit_indexes: np.ndarray,
    calibration_indexes: np.ndarray,
    class_count: int,
    add_intercept: bool,
    l2: float,
    head_count: int,
    seed: int,
) -> tuple[list[np.ndarray], str]:
    counts = Counter(int(y[index]) for index in fit_indexes)
    sufficient = len(fit_indexes) >= max(10, 2 * class_count) and all(
        counts[label] >= 2 for label in range(class_count)
    )
    if head_count <= 0:
        return [], "disabled"
    if not sufficient:
        return [], "insufficient_data"
    rng = np.random.default_rng(seed + 7919)
    by_class = {label: fit_indexes[y[fit_indexes] == label] for label in range(class_count)}
    output: list[np.ndarray] = []
    for _ in range(head_count):
        sampled = np.concatenate(
            [rng.choice(indexes, size=len(indexes), replace=True) for indexes in by_class.values()]
        )
        params, _status = _fit_head(
            xs[sampled],
            y[sampled],
            class_count=class_count,
            add_intercept=add_intercept,
            l2=l2,
        )
        calibration_logits = _logits(xs[calibration_indexes], params, class_count, add_intercept)
        temperature, _ = _fit_temperature(calibration_logits, y[calibration_indexes], class_count=class_count)
        output.append(
            _probabilities_from_logits(
                _logits(xc, params, class_count, add_intercept),
                temperature=temperature,
            )
        )
    return output, "fitted"


def _unpack_serializable(
    params: np.ndarray,
    *,
    class_count: int,
    add_intercept: bool,
) -> tuple[list[Any], list[float]]:
    if class_count == 2:
        feature_count = len(params) - int(add_intercept)
        bias = [float(params[feature_count])] if add_intercept else []
        return params[:feature_count].tolist(), bias
    feature_count = (len(params) - class_count * int(add_intercept)) // class_count
    weight_size = class_count * feature_count
    bias = params[weight_size:].tolist() if add_intercept else []
    return params[:weight_size].reshape(class_count, feature_count).tolist(), bias


def _solve_mias_dcms(
    *,
    scoring: MIASScoringResult,
    sample_ids: list[str],
    memberships: list[dict[str, float]],
    lowers: list[dict[str, float]],
    uppers: list[dict[str, float]],
    budget: int,
    seed: int,
    slack_grid: Sequence[float],
    kappa: float,
) -> MIASSelectionResult:
    targets = _pool_target_moments(memberships)
    use_robust_bounds = scoring.bootstrap_heads_fitted > 0
    result = solve_dcms_with_slack(
        sample_ids=sample_ids,
        utilities=[scoring.utility[sample_id] for sample_id in sample_ids],
        group_membership=memberships,
        membership_lower=lowers if use_robust_bounds else None,
        membership_upper=uppers if use_robust_bounds else None,
        budget=budget,
        target_moments=targets,
        slack_grid=tuple(float(value) for value in slack_grid),
        kappa=float(kappa),
        rounding_seed=int(seed),
    )
    return MIASSelectionResult(
        selected_ids=list(result.selected_ids),
        scoring=scoring,
        group_membership=dict(zip(sample_ids, memberships, strict=True)),
        membership_lower=dict(zip(sample_ids, lowers, strict=True)),
        membership_upper=dict(zip(sample_ids, uppers, strict=True)),
        target_moments=targets,
        dcms=result,
    )


def _pool_target_moments(memberships: Sequence[Mapping[str, float]]) -> dict[str, float]:
    if not memberships:
        raise ValueError("cannot derive DCMS targets from an empty pool")
    groups = sorted({str(group) for membership in memberships for group in membership})
    return {
        group: sum(float(membership.get(group, 0.0)) for membership in memberships) / len(memberships)
        for group in groups
    }


def _classification_feature(row: Mapping[str, Any]) -> list[float]:
    value = row.get("representation_embedding", row.get("embedding"))
    if not isinstance(value, (list, tuple)) or not value:
        raise ValueError(f"classification row {_row_id(row)!r} is missing representation_embedding")
    vector = [float(item) for item in value]
    if any(not math.isfinite(item) for item in vector):
        raise ValueError("classification embeddings must be finite")
    return vector


def _classification_label_indexes(
    raw_labels: Sequence[Any],
    candidates: Sequence[Mapping[str, Any]],
) -> tuple[list[Any], list[int]]:
    probability_counts = {
        len(value)
        for row in candidates
        for value in [row.get("probabilities")]
        if isinstance(value, (list, tuple))
    }
    if len(probability_counts) > 1:
        raise ValueError("candidate probability rows disagree on class count")
    if raw_labels and all(isinstance(value, (int, np.integer, bool)) for value in raw_labels):
        inferred = max(int(value) for value in raw_labels) + 1
        class_count = next(iter(probability_counts), inferred)
        if class_count < inferred:
            raise ValueError("seed label is outside the candidate class range")
        class_values = list(range(class_count))
        return class_values, [int(value) for value in raw_labels]
    class_values = sorted(set(raw_labels), key=str)
    if probability_counts and next(iter(probability_counts)) != len(class_values):
        raise ValueError("string seed labels must cover all candidate classes")
    by_value = {value: index for index, value in enumerate(class_values)}
    return class_values, [by_value[value] for value in raw_labels]


def _required_seed_label(row: Mapping[str, Any], *, label_field: str) -> Any:
    if label_field not in row:
        raise ValueError(f"seed row {_row_id(row)!r} is missing label field {label_field!r}")
    return row[label_field]


def _classification_semantic_memberships(
    rows: list[dict[str, Any]],
    *,
    budget: int,
    cluster_count: int | None,
) -> list[dict[str, float]]:
    explicit: list[dict[str, float]] = []
    has_explicit = True
    for row in rows:
        value = row.get("semantic_cluster_membership", row.get("prompt_cluster_membership"))
        if not isinstance(value, Mapping):
            has_explicit = False
            break
        explicit.append({f"semantic_cluster={key}": float(item) for key, item in value.items()})
    if has_explicit:
        return explicit
    resolved = min(
        len(rows),
        int(cluster_count if cluster_count is not None else min(32, max(2, math.ceil(max(1, budget) / 4)))),
    )
    cluster_rows = [{**row, "id": _row_id(row)} for row in rows]
    assignments = build_prompt_cluster_assignments(
        rows=cluster_rows,
        embeddings_by_id={_row_id(row): _classification_feature(row) for row in rows},
        cluster_count=resolved,
        id_field="id",
    )
    by_id = {_row_id(row): row for row in assignments.rows}
    return [
        {
            f"semantic_cluster={cluster}": float(value)
            for cluster, value in by_id[_row_id(row)]["prompt_cluster_membership"].items()
        }
        for row in rows
    ]


def _classification_length_memberships(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, float]]:
    values = np.asarray([_classification_length(row) for row in rows], dtype=np.float64)
    lower, upper = np.quantile(values, [1.0 / 3.0, 2.0 / 3.0])
    output = []
    for value in values:
        name = "short" if value <= lower else "medium" if value <= upper else "long"
        output.append({f"length_bin={name}": 1.0})
    return output


def _classification_length(row: Mapping[str, Any]) -> float:
    for field in ("token_count", "input_token_count", "length"):
        if row.get(field) is not None:
            return float(row[field])
    return float(len(str(row.get("text", row.get("document", row.get("query", "")))).split()))


def _preference_label(row: Mapping[str, Any]) -> int:
    label = _preference_label_or_none(row)
    if label is not None:
        return label
    raise ValueError(f"seed preference row {_row_id(row)!r} is missing an A/B label")


def _preference_label_or_none(row: Mapping[str, Any]) -> int | None:
    if row.get("preferred_response") is not None:
        value = row["preferred_response"]
        if str(value).strip().upper() in {"1", "A"}:
            return 1
        if str(value).strip().upper() in {"2", "B"}:
            return 0
    if row.get("oracle_label") is not None:
        value = str(row["oracle_label"]).strip().upper()
        if value == "A":
            return 1
        if value == "B":
            return 0
    return None


def _preference_cost(row: Mapping[str, Any]) -> float:
    if row.get("completion_token_cost") is not None:
        return float(row["completion_token_cost"])
    a, b = _response_lengths(row, require_tokens=True)
    return a + b


def _response_lengths(
    row: Mapping[str, Any],
    *,
    require_tokens: bool = False,
) -> tuple[float, float]:
    pairs = (
        ("response_a_token_count", "response_b_token_count"),
        ("response_1_token_count", "response_2_token_count"),
    )
    for first, second in pairs:
        if row.get(first) is not None and row.get(second) is not None:
            return float(row[first]), float(row[second])
    if require_tokens:
        raise ValueError(f"preference candidate {_row_id(row)!r} is missing completion token counts")
    word_pairs = (
        ("response_a_word_count", "response_b_word_count"),
        ("response_1_word_count", "response_2_word_count"),
    )
    for first, second in word_pairs:
        if row.get(first) is not None and row.get(second) is not None:
            return float(row[first]), float(row[second])
    response_a = row.get("response_a", row.get("response_1", ""))
    response_b = row.get("response_b", row.get("response_2", ""))
    return float(len(str(response_a).split())), float(len(str(response_b).split()))


def _embedding(row: Mapping[str, Any], primary: str, fallback: str) -> np.ndarray:
    value = row.get(primary, row.get(fallback))
    if not isinstance(value, (list, tuple)) or not value:
        raise ValueError(f"preference row {_row_id(row)!r} is missing {primary}")
    vector = np.asarray(value, dtype=np.float64)
    if vector.ndim != 1 or np.any(~np.isfinite(vector)):
        raise ValueError("preference embeddings must be finite vectors")
    return vector


def _preference_exact_memberships(row: Mapping[str, Any]) -> dict[str, float]:
    memberships: dict[str, float] = {}
    cluster_membership = row.get("prompt_cluster_membership")
    if isinstance(cluster_membership, Mapping):
        memberships.update({f"prompt_cluster={key}": float(value) for key, value in cluster_membership.items()})
    elif row.get("prompt_cluster") is not None:
        memberships[f"prompt_cluster={row['prompt_cluster']}"] = 1.0
    elif row.get("prompt_cluster_id") is not None:
        memberships[f"prompt_cluster={row['prompt_cluster_id']}"] = 1.0
    else:
        raise ValueError(f"preference candidate {_row_id(row)!r} is missing prompt-cluster metadata")
    length_a, length_b = _response_lengths(row)
    gap = (length_a - length_b) / max(1.0, length_a + length_b)
    memberships[f"length_gap_bin={length_gap_bin(gap)}"] = 1.0
    memberships[f"ab_position={row.get('ab_position', 'unknown')}"] = 1.0
    return memberships


def _row_id(row: Mapping[str, Any]) -> str:
    value = row.get("sample_id", row.get("id"))
    if value is None or not str(value):
        raise ValueError("row is missing a non-empty sample_id/id")
    return str(value)


def _dcms_metadata(result: DCMSResult) -> dict[str, Any]:
    return {
        "selected_slack": result.selected_slack,
        "utility_retained": result.utility_retained,
        "max_constraint_violation": result.max_constraint_violation,
        "continuous_moments": dict(result.continuous_moments),
        "rounded_moments": dict(result.rounded_moments),
        "robust_lower_moments": dict(result.robust_lower_moments),
        "robust_upper_moments": dict(result.robust_upper_moments),
        "solver_status": result.solver_status,
        "rounding_seed": result.rounding_seed,
        "q_propensity": dict(result.q_propensity),
        "selection_indicator": dict(result.selection_indicator),
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
