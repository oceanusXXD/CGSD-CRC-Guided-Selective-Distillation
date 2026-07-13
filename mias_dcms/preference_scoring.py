from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
import math
from typing import Any

from mias_dcms.selectors import assert_selector_rows_are_label_safe


SUPPORTED_PREFERENCE_SCORE_METHODS = ("reward_margin", "apl", "active_dpo")


def reward_margin_scores(
    rows: Iterable[Mapping[str, Any]],
    *,
    probability_field: str = "probability_response_1",
    id_field: str = "sample_id",
) -> dict[str, float]:
    return {
        _row_id(row, id_field=id_field): 1.0 - abs(2.0 * _probability(row, probability_field) - 1.0)
        for row in rows
    }


def apl_scores(
    rows: Iterable[Mapping[str, Any]],
    *,
    probability_field: str = "probability_response_1",
    prompt_cluster_probabilities_field: str = "prompt_cluster_probabilities",
    prompt_entropy_field: str = "prompt_entropy",
    prompt_entropy_weight: float = 1.0,
    id_field: str = "sample_id",
) -> dict[str, float]:
    if prompt_entropy_weight < 0.0:
        raise ValueError("prompt_entropy_weight must be non-negative")
    margin = reward_margin_scores(rows, probability_field=probability_field, id_field=id_field)
    scores: dict[str, float] = {}
    for row in rows:
        sample_id = _row_id(row, id_field=id_field)
        prompt_entropy = _prompt_entropy(
            row,
            probabilities_field=prompt_cluster_probabilities_field,
            entropy_field=prompt_entropy_field,
        )
        scores[sample_id] = margin[sample_id] + float(prompt_entropy_weight) * prompt_entropy
    return scores


def active_dpo_scores(
    rows: Iterable[Mapping[str, Any]],
    *,
    policy_response_1_field: str = "policy_logprob_response_1",
    policy_response_2_field: str = "policy_logprob_response_2",
    reference_response_1_field: str = "reference_logprob_response_1",
    reference_response_2_field: str = "reference_logprob_response_2",
    policy_gap_field: str = "policy_logprob_gap",
    reference_gap_field: str = "reference_logprob_gap",
    token_count_response_1_field: str = "token_count_response_1",
    token_count_response_2_field: str = "token_count_response_2",
    length_normalize: bool = False,
    prompt_cluster_probabilities_field: str = "prompt_cluster_probabilities",
    prompt_entropy_field: str = "prompt_entropy",
    novelty_weight: float = 0.0,
    id_field: str = "sample_id",
) -> dict[str, float]:
    components = active_dpo_score_components(
        rows,
        policy_response_1_field=policy_response_1_field,
        policy_response_2_field=policy_response_2_field,
        reference_response_1_field=reference_response_1_field,
        reference_response_2_field=reference_response_2_field,
        policy_gap_field=policy_gap_field,
        reference_gap_field=reference_gap_field,
        token_count_response_1_field=token_count_response_1_field,
        token_count_response_2_field=token_count_response_2_field,
        length_normalize=length_normalize,
        prompt_cluster_probabilities_field=prompt_cluster_probabilities_field,
        prompt_entropy_field=prompt_entropy_field,
        novelty_weight=novelty_weight,
        id_field=id_field,
    )
    return {
        sample_id: float(values["active_dpo_score"])
        for sample_id, values in components.items()
    }


def active_dpo_score_components(
    rows: Iterable[Mapping[str, Any]],
    *,
    policy_response_1_field: str = "policy_logprob_response_1",
    policy_response_2_field: str = "policy_logprob_response_2",
    reference_response_1_field: str = "reference_logprob_response_1",
    reference_response_2_field: str = "reference_logprob_response_2",
    policy_gap_field: str = "policy_logprob_gap",
    reference_gap_field: str = "reference_logprob_gap",
    token_count_response_1_field: str = "token_count_response_1",
    token_count_response_2_field: str = "token_count_response_2",
    length_normalize: bool = False,
    prompt_cluster_probabilities_field: str = "prompt_cluster_probabilities",
    prompt_entropy_field: str = "prompt_entropy",
    novelty_weight: float = 0.0,
    id_field: str = "sample_id",
) -> dict[str, dict[str, float]]:
    if novelty_weight < 0.0:
        raise ValueError("novelty_weight must be non-negative")
    components: dict[str, dict[str, float]] = {}
    for row in rows:
        policy_gap = _logprob_gap(
            row,
            gap_field=policy_gap_field,
            response_1_field=policy_response_1_field,
            response_2_field=policy_response_2_field,
        )
        reference_gap = _logprob_gap(
            row,
            gap_field=reference_gap_field,
            response_1_field=reference_response_1_field,
            response_2_field=reference_response_2_field,
        )
        sample_id = _row_id(row, id_field=id_field)
        gradient_proxy = abs(policy_gap - reference_gap)
        length_normalized_proxy = gradient_proxy
        if length_normalize:
            length_normalized_proxy = gradient_proxy / _pair_token_count(
                row,
                response_1_field=token_count_response_1_field,
                response_2_field=token_count_response_2_field,
            )
        novelty_score = _prompt_entropy(
            row,
            probabilities_field=prompt_cluster_probabilities_field,
            entropy_field=prompt_entropy_field,
        )
        score = length_normalized_proxy + float(novelty_weight) * novelty_score
        components[sample_id] = {
            "active_dpo_gradient_proxy": gradient_proxy,
            "active_dpo_length_normalized_proxy": length_normalized_proxy,
            "active_dpo_novelty_score": novelty_score,
            "active_dpo_score": score,
        }
    return components


def build_preference_baseline_score_rows(
    rows: Iterable[dict[str, Any]],
    *,
    methods: Sequence[str],
    prompt_entropy_weight: float = 1.0,
    active_dpo_length_normalize: bool = False,
    active_dpo_novelty_weight: float = 0.0,
    id_field: str = "sample_id",
) -> list[dict[str, Any]]:
    source_rows = [dict(row) for row in rows]
    assert_selector_rows_are_label_safe(source_rows)
    normalized_methods = _normalize_methods(methods)
    score_maps: dict[str, dict[str, float]] = {}
    active_dpo_components: dict[str, dict[str, float]] = {}
    if "reward_margin" in normalized_methods:
        score_maps["reward_margin"] = reward_margin_scores(source_rows, id_field=id_field)
    if "apl" in normalized_methods:
        score_maps["apl"] = apl_scores(
            source_rows,
            prompt_entropy_weight=prompt_entropy_weight,
            id_field=id_field,
        )
    if "active_dpo" in normalized_methods:
        active_dpo_components = active_dpo_score_components(
            source_rows,
            length_normalize=active_dpo_length_normalize,
            novelty_weight=active_dpo_novelty_weight,
            id_field=id_field,
        )
        score_maps["active_dpo"] = {
            sample_id: float(values["active_dpo_score"])
            for sample_id, values in active_dpo_components.items()
        }

    scored_rows: list[dict[str, Any]] = []
    for row in source_rows:
        sample_id = _row_id(row, id_field=id_field)
        selector_scores = {
            method: score_maps[method][sample_id] for method in normalized_methods
        }
        scored_row = dict(row)
        scored_row["selector_scores"] = selector_scores
        if "active_dpo" in normalized_methods:
            scored_row.update(active_dpo_components[sample_id])
        for method, score in selector_scores.items():
            scored_row[f"{method}_score"] = score
        scored_rows.append(scored_row)
    return scored_rows


def _normalize_methods(methods: Sequence[str]) -> tuple[str, ...]:
    normalized = tuple(_normalize_method(method) for method in methods)
    if not normalized:
        raise ValueError("methods must not be empty")
    return normalized


def _normalize_method(method: str) -> str:
    key = str(method).strip().lower().replace("-", "_")
    aliases = {
        "reward_margin": "reward_margin",
        "margin": "reward_margin",
        "apl": "apl",
        "active_dpo": "active_dpo",
        "activedpo": "active_dpo",
    }
    if key not in aliases:
        raise ValueError(f"unsupported preference score method: {method!r}")
    return aliases[key]


def _row_id(row: Mapping[str, Any], *, id_field: str) -> str:
    value = row.get(id_field, row.get("id"))
    if value is None:
        raise ValueError(f"row is missing id field {id_field!r} and fallback 'id'")
    return str(value)


def _probability(row: Mapping[str, Any], field: str) -> float:
    if field in row:
        probability = float(row[field])
    elif "probabilities" in row:
        probabilities = list(row["probabilities"])
        if not probabilities:
            raise ValueError("probabilities must not be empty")
        probability = float(probabilities[0])
    else:
        raise ValueError(f"row {_row_id(row, id_field='sample_id')!r} is missing {field!r}")
    if not 0.0 <= probability <= 1.0:
        raise ValueError(f"probability must be between zero and one, got {probability}")
    return probability


def _prompt_entropy(
    row: Mapping[str, Any],
    *,
    probabilities_field: str,
    entropy_field: str,
) -> float:
    if entropy_field in row and row[entropy_field] is not None:
        entropy = float(row[entropy_field])
        if entropy < 0.0:
            raise ValueError("prompt entropy must be non-negative")
        return entropy
    if probabilities_field not in row or row[probabilities_field] is None:
        return 0.0
    values = _normalize_probability_vector(row[probabilities_field])
    return -sum(value * math.log(value) for value in values if value > 0.0)


def _normalize_probability_vector(values: Any) -> list[float]:
    probabilities = [float(value) for value in values]
    if not probabilities:
        raise ValueError("probability vector must not be empty")
    if any(value < 0.0 for value in probabilities):
        raise ValueError("probability vector must be non-negative")
    total = sum(probabilities)
    if total <= 0.0:
        raise ValueError("probability vector must have positive mass")
    return [value / total for value in probabilities]


def _logprob_gap(
    row: Mapping[str, Any],
    *,
    gap_field: str,
    response_1_field: str,
    response_2_field: str,
) -> float:
    if gap_field in row and row[gap_field] is not None:
        return float(row[gap_field])
    missing = [
        field for field in (response_1_field, response_2_field) if field not in row or row[field] is None
    ]
    if missing:
        raise ValueError(
            f"row {_row_id(row, id_field='sample_id')!r} is missing logprob fields: {missing}"
        )
    return float(row[response_1_field]) - float(row[response_2_field])


def _pair_token_count(
    row: Mapping[str, Any],
    *,
    response_1_field: str,
    response_2_field: str,
) -> float:
    missing = [
        field for field in (response_1_field, response_2_field) if field not in row or row[field] is None
    ]
    if missing:
        raise ValueError(
            f"row {_row_id(row, id_field='sample_id')!r} is missing token count fields: {missing}"
        )
    token_count = float(row[response_1_field]) + float(row[response_2_field])
    if token_count <= 0.0:
        raise ValueError("pair token count must be positive")
    return token_count
