from __future__ import annotations

from collections.abc import Iterable, Mapping
import random
from typing import Any


def build_preference_split_manifest(
    rows: Iterable[Mapping[str, Any]],
    *,
    seed: int,
    seed_size: int,
    active_size: int,
    heldout_size: int,
    test_size: int,
    id_field: str = "sample_id",
) -> dict[str, Any]:
    row_ids = [_row_id(row, id_field=id_field, row_index=index) for index, row in enumerate(rows)]
    if len(set(row_ids)) != len(row_ids):
        raise ValueError("preference rows must have unique ids")

    requested = {
        "seed_size": int(seed_size),
        "active_size": int(active_size),
        "heldout_size": int(heldout_size),
        "test_size": int(test_size),
    }
    if any(value < 0 for value in requested.values()):
        raise ValueError("split sizes must be non-negative")
    requested_count = sum(requested.values())
    if requested_count > len(row_ids):
        raise ValueError("requested split sizes exceed row count")

    shuffled = list(row_ids)
    random.Random(int(seed)).shuffle(shuffled)
    seed_end = requested["seed_size"]
    active_end = seed_end + requested["active_size"]
    heldout_end = active_end + requested["heldout_size"]
    test_end = heldout_end + requested["test_size"]

    return {
        "seed": int(seed),
        "row_count": len(row_ids),
        "id_field": str(id_field),
        "seed_ids": sorted(shuffled[:seed_end]),
        "active_pool_ids": sorted(shuffled[seed_end:active_end]),
        "heldout_ids": sorted(shuffled[active_end:heldout_end]),
        "test_ids": sorted(shuffled[heldout_end:test_end]),
        "unused_ids": sorted(shuffled[test_end:]),
    }


def _row_id(row: Mapping[str, Any], *, id_field: str, row_index: int) -> str:
    value = row.get(id_field, row.get("id"))
    if value is None:
        raise ValueError(f"row {row_index} is missing id field {id_field!r} and fallback 'id'")
    return str(value)
