from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Any, Iterable


FORBIDDEN_SELECTOR_FIELDS = {
    "chosen",
    "rejected",
    "preference_label",
    "preference_strength",
    "preference_magnitude",
    "justification",
    "preference_statement",
    "preference_elaboration",
    "oracle_label",
    "preferred_response",
    "true_class",
    "true_label",
    "true_label_name",
    "label",
    "label_name",
    "prediction_correct",
    "is_correct",
}


DEFAULT_LENGTH_BIN_EDGES = (-0.2, 0.2)


@dataclass(frozen=True)
class PreferenceFixedPool:
    active_pool: list[dict[str, Any]]
    oracle_store: dict[str, dict[str, Any]]
    swap_manifest: list[dict[str, Any]]


def build_preference_fixed_pool(
    rows: Iterable[dict[str, Any]],
    *,
    seed: int,
    force_swap: bool | None = None,
    include_both_positions: bool = False,
) -> PreferenceFixedPool:
    rng = random.Random(seed)
    active_pool: list[dict[str, Any]] = []
    oracle_store: dict[str, dict[str, Any]] = {}
    swap_manifest: list[dict[str, Any]] = []

    for index, source_row in enumerate(rows):
        row = dict(source_row)
        sample_id = str(row.get("id") or row.get("sample_id") or f"pair:{index}")
        prompt = str(row["prompt"])
        response_a = str(_first_present(row, ("response_a", "response_A", "response_1")))
        response_b = str(_first_present(row, ("response_b", "response_B", "response_2")))
        original_label = _normalize_preference_label(row)
        swap_values = (
            (False, True)
            if include_both_positions
            else (bool(force_swap) if force_swap is not None else rng.random() < 0.5,)
        )

        for swapped in swap_values:
            active_sample_id = (
                f"{sample_id}:{'swapped' if swapped else 'original'}"
                if include_both_positions
                else sample_id
            )

            if swapped:
                selector_response_a = response_b
                selector_response_b = response_a
                preference_label = _flip_label(original_label)
                source_a = row.get("source_b") or row.get("response_b_source") or row.get("response_2_source")
                source_b = row.get("source_a") or row.get("response_a_source") or row.get("response_1_source")
            else:
                selector_response_a = response_a
                selector_response_b = response_b
                preference_label = original_label
                source_a = row.get("source_a") or row.get("response_a_source") or row.get("response_1_source")
                source_b = row.get("source_b") or row.get("response_b_source") or row.get("response_2_source")

            active_row = {
                "sample_id": active_sample_id,
                "id": active_sample_id,
                "swap_pair_id": sample_id,
                "prompt": prompt,
                "response_a": selector_response_a,
                "response_b": selector_response_b,
                "response_a_word_count": _word_count(selector_response_a),
                "response_b_word_count": _word_count(selector_response_b),
                "response_a_char_count": len(selector_response_a),
                "response_b_char_count": len(selector_response_b),
                "length_gap": normalized_response_length_gap(selector_response_a, selector_response_b),
                "source_a": str(source_a) if source_a is not None else None,
                "source_b": str(source_b) if source_b is not None else None,
                "source_pair": _source_pair(source_a, source_b),
                "ab_position": "swapped" if swapped else "original",
            }
            active_row["length_gap_bin"] = length_gap_bin(active_row["length_gap"])
            active_pool.append(_strip_forbidden_selector_fields(active_row))
            oracle_store[active_sample_id] = {
                "sample_id": active_sample_id,
                "swap_pair_id": sample_id,
                "preference_label": preference_label,
                "original_preference_label": original_label,
                "swapped": swapped,
                "preference_strength": row.get("preference_strength"),
            }
            swap_manifest.append(
                {
                    "sample_id": active_sample_id,
                    "swap_pair_id": sample_id,
                    "swapped": swapped,
                    "original_response_a": response_a,
                    "original_response_b": response_b,
                    "active_response_a_from": "original_b" if swapped else "original_a",
                    "active_response_b_from": "original_a" if swapped else "original_b",
                }
            )

    return PreferenceFixedPool(
        active_pool=active_pool,
        oracle_store=oracle_store,
        swap_manifest=swap_manifest,
    )


def normalized_response_length_gap(response_a: str, response_b: str) -> float:
    a_len = _word_count(response_a)
    b_len = _word_count(response_b)
    total = a_len + b_len
    if total <= 0:
        return 0.0
    return (a_len - b_len) / total


def length_gap_bin(
    value: float,
    *,
    edges: tuple[float, float] = DEFAULT_LENGTH_BIN_EDGES,
) -> str:
    """Map the signed response-length gap to the frozen observable bins."""
    lower, upper = (float(edges[0]), float(edges[1]))
    if lower >= upper:
        raise ValueError("length-bin edges must be strictly increasing")
    gap = float(value)
    if gap < lower:
        return "b_longer"
    if gap > upper:
        return "a_longer"
    return "balanced"


def _strip_forbidden_selector_fields(row: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in row.items() if key not in FORBIDDEN_SELECTOR_FIELDS}


def _normalize_preference_label(row: dict[str, Any]) -> str:
    value = row.get("preference_label", row.get("chosen", row.get("preferred_response")))
    if value in ("A", "a", 1, "1", "response_a", "response_1"):
        return "A"
    if value in ("B", "b", 2, "2", "response_b", "response_2"):
        return "B"
    if value in (0, "0", None):
        return "tie"
    raise ValueError(f"unsupported preference label: {value!r}")


def _flip_label(label: str) -> str:
    if label == "A":
        return "B"
    if label == "B":
        return "A"
    return label


def _first_present(row: dict[str, Any], names: tuple[str, ...]) -> Any:
    for name in names:
        if name in row:
            return row[name]
    raise KeyError(f"missing required field; tried {names}")


def _word_count(text: str) -> int:
    return len(str(text).split())


def _source_pair(source_a: Any, source_b: Any) -> str:
    left = "unknown" if source_a is None else str(source_a)
    right = "unknown" if source_b is None else str(source_b)
    return f"{left}|{right}"
