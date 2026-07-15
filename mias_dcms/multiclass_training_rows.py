from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping
from typing import Any


def build_multiclass_training_rows(
    seed_rows: Iterable[Mapping[str, Any]],
    selected_rows: Iterable[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Combine fixed seed labels with post-selection labels for one method."""
    seed = _validated_rows(seed_rows, source="seed")
    selected = _validated_rows(selected_rows, source="selected")
    seed_ids = {str(row["id"]) for row in seed}
    selected_ids = {str(row["id"]) for row in selected}
    overlap = sorted(seed_ids & selected_ids)
    if overlap:
        raise ValueError(f"seed and selected rows overlap: {overlap[:5]}")

    rows = [*seed, *selected]
    label_counts = Counter(str(int(row["label"])) for row in rows)
    return rows, {
        "seed_row_count": len(seed),
        "selected_row_count": len(selected),
        "training_row_count": len(rows),
        "label_counts": dict(sorted(label_counts.items(), key=lambda item: int(item[0]))),
    }


def _validated_rows(rows: Iterable[Mapping[str, Any]], *, source: str) -> list[dict[str, Any]]:
    validated: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for index, source_row in enumerate(rows):
        row = dict(source_row)
        sample_id = str(row.get("id", ""))
        if not sample_id:
            raise ValueError(f"{source} row {index} has an empty id")
        if sample_id in seen_ids:
            raise ValueError(f"{source} rows contain duplicate id {sample_id!r}")
        if not str(row.get("text", "")).strip():
            raise ValueError(f"{source} row {sample_id!r} has empty text")
        if "label" not in row:
            raise ValueError(f"{source} row {sample_id!r} has no label")
        try:
            row["label"] = int(row["label"])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{source} row {sample_id!r} has non-integer label") from exc
        row["id"] = sample_id
        row["training_row_source"] = source
        seen_ids.add(sample_id)
        validated.append(row)
    if not validated:
        raise ValueError(f"{source} rows must not be empty")
    return validated
