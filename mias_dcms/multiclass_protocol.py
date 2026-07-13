from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import random
from typing import Any, Iterable, Mapping


@dataclass(frozen=True)
class ClassPrior:
    total_count: int
    class_counts: dict[str, int]
    class_shares: dict[str, float]

    def as_dict(self) -> dict[str, Any]:
        return {
            "total_count": self.total_count,
            "class_counts": dict(self.class_counts),
            "class_shares": dict(self.class_shares),
        }


def pool_class_prior(
    rows: Iterable[Mapping[str, Any]],
    *,
    label_field: str,
) -> ClassPrior:
    counts: Counter[str] = Counter(str(row[label_field]) for row in rows)
    total = sum(counts.values())
    if total == 0:
        raise ValueError("rows must not be empty")
    ordered_counts = {label: int(counts[label]) for label in sorted(counts)}
    shares = {label: ordered_counts[label] / total for label in ordered_counts}
    return ClassPrior(total_count=total, class_counts=ordered_counts, class_shares=shares)


def build_fixed_multiclass_splits(
    rows: Iterable[Mapping[str, Any]],
    *,
    seed: int,
    seed_size: int,
    active_size: int,
    test_size: int,
    id_field: str = "id",
) -> dict[str, list[str]]:
    materialized = [dict(row) for row in rows]
    total_requested = int(seed_size) + int(active_size) + int(test_size)
    if min(seed_size, active_size, test_size) < 0:
        raise ValueError("split sizes must be non-negative")
    if total_requested > len(materialized):
        raise ValueError("requested split sizes exceed row count")

    ids = [str(row[id_field]) for row in materialized]
    if len(set(ids)) != len(ids):
        raise ValueError("row ids must be unique")

    shuffled = list(ids)
    random.Random(seed).shuffle(shuffled)
    seed_ids = sorted(shuffled[:seed_size])
    active_start = seed_size
    active_end = seed_size + active_size
    active_pool_ids = sorted(shuffled[active_start:active_end])
    test_ids = sorted(shuffled[active_end:active_end + test_size])
    splits = {
        "seed_ids": seed_ids,
        "active_pool_ids": active_pool_ids,
        "test_ids": test_ids,
    }
    validate_disjoint_splits(splits)
    return splits


def validate_disjoint_splits(splits: Mapping[str, Iterable[str]]) -> None:
    seen: dict[str, str] = {}
    for split_name, ids in splits.items():
        for sample_id in ids:
            key = str(sample_id)
            if key in seen:
                raise ValueError(
                    f"sample_id {key!r} appears in both {seen[key]!r} and {split_name!r}"
                )
            seen[key] = str(split_name)
