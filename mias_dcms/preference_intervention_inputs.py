from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
import math
from typing import Any

from mias_dcms.preference_pool import length_gap_bin, normalized_response_length_gap
from mias_dcms.selectors import assert_selector_rows_are_label_safe


DEFAULT_LENGTH_BIN_EDGES = (-0.2, 0.2)


def build_preference_intervention_rows(
    *,
    active_pool_rows: Iterable[Mapping[str, Any]],
    logprob_rows: Iterable[Mapping[str, Any]] = (),
    score_rows: Iterable[Mapping[str, Any]] = (),
    id_field: str = "sample_id",
    length_bin_edges: Sequence[float] = DEFAULT_LENGTH_BIN_EDGES,
    base_margin_field: str = "implicit_reward_gap",
    selector_score_fields: Sequence[str] = ("reward_margin_score", "apl_score", "active_dpo_score"),
) -> list[dict[str, Any]]:
    active_rows = [dict(row) for row in active_pool_rows]
    assert_selector_rows_are_label_safe(active_rows)
    logprobs_by_id = _rows_by_id(logprob_rows, id_field=id_field)
    scores_by_id = _rows_by_id(score_rows, id_field=id_field)
    edges = _validate_length_bin_edges(length_bin_edges)

    output_rows: list[dict[str, Any]] = []
    for row in active_rows:
        sample_id = _row_id(row, id_field=id_field)
        merged = {**row, **logprobs_by_id.get(sample_id, {}), **scores_by_id.get(sample_id, {})}
        assert_selector_rows_are_label_safe([merged])

        length_gap = _length_gap(merged)
        intervention_row = {
            "sample_id": sample_id,
            "id": str(merged.get("id", sample_id)),
            "base_margin": _base_margin(merged, base_margin_field=base_margin_field),
            "length_gap": length_gap,
            "length_gap_bin": length_gap_bin(length_gap, edges=edges),
            "source_pair": str(merged.get("source_pair", "unknown|unknown")),
            "ab_position": str(merged.get("ab_position", "unknown")),
            "swap_pair_id": str(merged.get("swap_pair_id", sample_id)),
        }
        for field in selector_score_fields:
            if field in merged and merged[field] is not None:
                intervention_row[str(field)] = float(merged[field])
        selector_scores = merged.get("selector_scores")
        if isinstance(selector_scores, Mapping):
            for method, score in selector_scores.items():
                intervention_row[f"{method}_score"] = float(score)
        if "prompt_cluster" in merged:
            intervention_row["prompt_cluster"] = str(merged["prompt_cluster"])
        if "prompt_cluster_id" in merged:
            intervention_row["prompt_cluster"] = str(merged["prompt_cluster_id"])
        output_rows.append(intervention_row)
    return output_rows


def _rows_by_id(rows: Iterable[Mapping[str, Any]], *, id_field: str) -> dict[str, dict[str, Any]]:
    by_id: dict[str, dict[str, Any]] = {}
    for row in rows:
        payload = dict(row)
        sample_id = _row_id(payload, id_field=id_field)
        if sample_id in by_id:
            raise ValueError(f"duplicate row for sample id {sample_id!r}")
        by_id[sample_id] = payload
    assert_selector_rows_are_label_safe(by_id.values())
    return by_id


def _row_id(row: Mapping[str, Any], *, id_field: str) -> str:
    value = row.get(id_field, row.get("id"))
    if value is None:
        raise ValueError(f"row is missing id field {id_field!r} and fallback 'id'")
    return str(value)


def _length_gap(row: Mapping[str, Any]) -> float:
    if "length_gap" in row and row["length_gap"] is not None:
        return float(row["length_gap"])
    if "response_a" in row and "response_b" in row:
        return normalized_response_length_gap(str(row["response_a"]), str(row["response_b"]))
    if "response_a_word_count" in row and "response_b_word_count" in row:
        left = int(row["response_a_word_count"])
        right = int(row["response_b_word_count"])
        total = left + right
        return 0.0 if total <= 0 else (left - right) / total
    raise ValueError(f"row {_row_id(row, id_field='sample_id')!r} is missing length gap inputs")


def _base_margin(row: Mapping[str, Any], *, base_margin_field: str) -> float:
    if base_margin_field in row and row[base_margin_field] is not None:
        return _finite_float(row[base_margin_field], field=base_margin_field)
    if "policy_logprob_gap" in row and "reference_logprob_gap" in row:
        return _finite_float(row["policy_logprob_gap"], field="policy_logprob_gap") - _finite_float(
            row["reference_logprob_gap"],
            field="reference_logprob_gap",
        )
    required_fields = (
        "policy_logprob_response_1",
        "policy_logprob_response_2",
        "reference_logprob_response_1",
        "reference_logprob_response_2",
    )
    if all(field in row and row[field] is not None for field in required_fields):
        policy_gap = _finite_float(row["policy_logprob_response_1"], field="policy_logprob_response_1") - _finite_float(
            row["policy_logprob_response_2"],
            field="policy_logprob_response_2",
        )
        reference_gap = _finite_float(row["reference_logprob_response_1"], field="reference_logprob_response_1") - _finite_float(
            row["reference_logprob_response_2"],
            field="reference_logprob_response_2",
        )
        return policy_gap - reference_gap
    raise ValueError(f"row {_row_id(row, id_field='sample_id')!r} is missing base margin inputs")


def _finite_float(value: Any, *, field: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError(f"{field} must be finite")
    return parsed


def _validate_length_bin_edges(edges: Sequence[float]) -> tuple[float, float]:
    values = [float(value) for value in edges]
    if len(values) != 2:
        raise ValueError("length_bin_edges must contain exactly two values")
    if values[0] >= values[1]:
        raise ValueError("length_bin_edges must be strictly increasing")
    return values[0], values[1]
