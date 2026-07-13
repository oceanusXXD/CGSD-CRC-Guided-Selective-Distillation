from __future__ import annotations

from collections.abc import Iterable, Mapping
import math
from typing import Any


DEFAULT_POLICY_RESPONSE_1_FIELD = "policy_logprob_response_1"
DEFAULT_POLICY_RESPONSE_2_FIELD = "policy_logprob_response_2"
DEFAULT_REFERENCE_RESPONSE_1_FIELD = "reference_logprob_response_1"
DEFAULT_REFERENCE_RESPONSE_2_FIELD = "reference_logprob_response_2"


def audit_preference_logprobs(
    rows: Iterable[Mapping[str, Any]],
    *,
    id_field: str = "sample_id",
    policy_response_1_field: str = DEFAULT_POLICY_RESPONSE_1_FIELD,
    policy_response_2_field: str = DEFAULT_POLICY_RESPONSE_2_FIELD,
    reference_response_1_field: str = DEFAULT_REFERENCE_RESPONSE_1_FIELD,
    reference_response_2_field: str = DEFAULT_REFERENCE_RESPONSE_2_FIELD,
    require_nonzero_implicit_margin: bool = True,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    source_rows = [dict(row) for row in rows]
    audited_rows: list[dict[str, Any]] = []
    policy_gaps: list[float] = []
    reference_gaps: list[float] = []
    implicit_gaps: list[float] = []

    for row in source_rows:
        sample_id = _row_id(row, id_field=id_field)
        policy_1 = _required_finite_logprob(row, policy_response_1_field, sample_id=sample_id)
        policy_2 = _required_finite_logprob(row, policy_response_2_field, sample_id=sample_id)
        reference_1 = _required_finite_logprob(row, reference_response_1_field, sample_id=sample_id)
        reference_2 = _required_finite_logprob(row, reference_response_2_field, sample_id=sample_id)

        policy_gap = policy_1 - policy_2
        reference_gap = reference_1 - reference_2
        implicit_reward_gap = policy_gap - reference_gap
        absolute_implicit_margin = abs(implicit_reward_gap)

        audited_row = dict(row)
        audited_row["policy_logprob_gap"] = policy_gap
        audited_row["reference_logprob_gap"] = reference_gap
        audited_row["implicit_reward_gap"] = implicit_reward_gap
        audited_row["absolute_implicit_margin"] = absolute_implicit_margin
        audited_rows.append(audited_row)

        policy_gaps.append(policy_gap)
        reference_gaps.append(reference_gap)
        implicit_gaps.append(implicit_reward_gap)

    implicit_margin_not_all_zero = any(abs(value) > 1e-12 for value in implicit_gaps)
    if require_nonzero_implicit_margin and not implicit_margin_not_all_zero:
        raise ValueError("implicit margin is zero for every row")

    summary = {
        "row_count": len(source_rows),
        "finite_row_count": len(audited_rows),
        "policy_gap_mean": _mean(policy_gaps),
        "reference_gap_mean": _mean(reference_gaps),
        "implicit_reward_gap_mean": _mean(implicit_gaps),
        "absolute_implicit_margin_mean": _mean([abs(value) for value in implicit_gaps]),
        "policy_gap_variance": _population_variance(policy_gaps),
        "reference_gap_variance": _population_variance(reference_gaps),
        "implicit_reward_gap_variance": _population_variance(implicit_gaps),
        "implicit_margin_not_all_zero": implicit_margin_not_all_zero,
        "implicit_reward_gap_sign_counts": _sign_counts(implicit_gaps),
        "logprob_fields": {
            "policy_response_1": policy_response_1_field,
            "policy_response_2": policy_response_2_field,
            "reference_response_1": reference_response_1_field,
            "reference_response_2": reference_response_2_field,
        },
    }
    return audited_rows, summary


def _row_id(row: Mapping[str, Any], *, id_field: str) -> str:
    value = row.get(id_field, row.get("id"))
    if value is None:
        raise ValueError(f"row is missing id field {id_field!r} and fallback 'id'")
    return str(value)


def _required_finite_logprob(row: Mapping[str, Any], field: str, *, sample_id: str) -> float:
    if field not in row or row[field] is None:
        raise ValueError(f"row {sample_id!r} is missing logprob fields: [{field!r}]")
    value = float(row[field])
    if not math.isfinite(value):
        raise ValueError(f"row {sample_id!r} has non-finite logprob in {field!r}: {value!r}")
    return value


def _mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def _population_variance(values: list[float]) -> float:
    if not values:
        return 0.0
    mean = _mean(values)
    return sum((value - mean) ** 2 for value in values) / len(values)


def _sign_counts(values: list[float]) -> dict[str, int]:
    counts = {"positive": 0, "negative": 0, "zero": 0}
    for value in values:
        if value > 1e-12:
            counts["positive"] += 1
        elif value < -1e-12:
            counts["negative"] += 1
        else:
            counts["zero"] += 1
    return counts
