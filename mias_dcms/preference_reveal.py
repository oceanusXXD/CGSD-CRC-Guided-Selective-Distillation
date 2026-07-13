from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class PreferenceRevealResult:
    revealed_rows: list[dict[str, Any]]
    training_rows: list[dict[str, Any]]
    revealed_ids: list[str]
    unrevealed_ids: list[str]


def reveal_selected_preference_labels(
    active_pool: Iterable[dict[str, Any]],
    *,
    oracle_store: Mapping[str, Mapping[str, Any]],
    selected_ids: Sequence[str],
    round_index: int,
    method: str,
    id_field: str = "sample_id",
) -> PreferenceRevealResult:
    active_rows = [dict(row) for row in active_pool]
    rows_by_id = {_row_id(row, id_field=id_field): row for row in active_rows}
    selected = [str(sample_id) for sample_id in selected_ids]
    if len(set(selected)) != len(selected):
        raise ValueError("selected_ids must be unique")

    missing_rows = [sample_id for sample_id in selected if sample_id not in rows_by_id]
    if missing_rows:
        raise ValueError(f"selected id not present in active pool: {missing_rows[0]!r}")
    missing_oracle = [sample_id for sample_id in selected if sample_id not in oracle_store]
    if missing_oracle:
        raise ValueError(f"selected id missing from oracle store: {missing_oracle[0]!r}")

    selected_set = set(selected)
    revealed_rows: list[dict[str, Any]] = []
    training_rows: list[dict[str, Any]] = []
    for sample_id in selected:
        active_row = rows_by_id[sample_id]
        oracle_row = dict(oracle_store[sample_id])
        label = _normalize_revealed_label(oracle_row.get("preference_label"))
        preferred_response = _preferred_response_index(label)
        revealed_row = {
            **active_row,
            "round": int(round_index),
            "method": str(method),
            "oracle_label": label,
            "preferred_response": preferred_response,
        }
        if "preference_strength" in oracle_row:
            revealed_row["preference_strength"] = oracle_row.get("preference_strength")
        revealed_rows.append(revealed_row)
        if preferred_response in (1, 2):
            training_rows.append(_dpo_training_row(revealed_row, id_field=id_field))

    unrevealed_ids = sorted(sample_id for sample_id in rows_by_id if sample_id not in selected_set)
    return PreferenceRevealResult(
        revealed_rows=revealed_rows,
        training_rows=training_rows,
        revealed_ids=selected,
        unrevealed_ids=unrevealed_ids,
    )


def _dpo_training_row(row: Mapping[str, Any], *, id_field: str) -> dict[str, Any]:
    sample_id = _row_id(row, id_field=id_field)
    training_row = {
        "id": sample_id,
        "sample_id": sample_id,
        "round": int(row["round"]),
        "method": str(row["method"]),
        "prompt": str(row["prompt"]),
        "response_1": str(row["response_a"]),
        "response_2": str(row["response_b"]),
        "preferred_response": int(row["preferred_response"]),
        "oracle_label": str(row["oracle_label"]),
    }
    if "preference_strength" in row:
        training_row["preference_strength"] = row.get("preference_strength")
    for field in ("length_gap", "source_pair", "source_a", "source_b", "ab_position"):
        if field in row:
            training_row[field] = row[field]
    return training_row


def _row_id(row: Mapping[str, Any], *, id_field: str) -> str:
    value = row.get(id_field, row.get("id"))
    if value is None:
        raise ValueError(f"row is missing id field {id_field!r} and fallback 'id'")
    return str(value)


def _normalize_revealed_label(value: Any) -> str:
    if value in ("A", "a", 1, "1", "response_a", "response_1"):
        return "A"
    if value in ("B", "b", 2, "2", "response_b", "response_2"):
        return "B"
    if value in ("tie", "Tie", "TIE", 0, "0", None):
        return "tie"
    raise ValueError(f"unsupported preference label: {value!r}")


def _preferred_response_index(label: str) -> int:
    if label == "A":
        return 1
    if label == "B":
        return 2
    return 0
