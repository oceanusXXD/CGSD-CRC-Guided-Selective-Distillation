from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import math
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
    label_field: str = "label",
) -> dict[str, list[str]]:
    materialized = [dict(row) for row in rows]
    total_requested = int(seed_size) + int(active_size) + int(test_size)
    if min(seed_size, active_size, test_size) < 0:
        raise ValueError("split sizes must be non-negative")
    if total_requested > len(materialized):
        raise ValueError("requested split sizes exceed row count")

    try:
        ids = [str(row[id_field]) for row in materialized]
        labels = [str(row[label_field]) for row in materialized]
    except KeyError as exc:
        raise ValueError(f"row is missing required split field {exc.args[0]!r}") from exc
    if len(set(ids)) != len(ids):
        raise ValueError("row ids must be unique")

    remaining: dict[str, list[str]] = {}
    for sample_id, label in zip(ids, labels, strict=True):
        remaining.setdefault(label, []).append(sample_id)
    for label, label_ids in remaining.items():
        random.Random(f"{seed}:{label}").shuffle(label_ids)

    selected_splits: list[list[str]] = []
    for split_index, requested_size in enumerate((seed_size, active_size, test_size)):
        allocation = _stratified_allocation(
            requested_size,
            {label: len(label_ids) for label, label_ids in remaining.items()},
            require_class_coverage=split_index == 0,
        )
        selected: list[str] = []
        for label in sorted(allocation):
            count = allocation[label]
            selected.extend(remaining[label][:count])
            remaining[label] = remaining[label][count:]
        selected_splits.append(sorted(selected))
    seed_ids, active_pool_ids, test_ids = selected_splits
    splits = {
        "seed_ids": seed_ids,
        "active_pool_ids": active_pool_ids,
        "test_ids": test_ids,
    }
    validate_disjoint_splits(splits)
    return splits


def _stratified_allocation(
    requested: int,
    capacities: Mapping[str, int],
    *,
    require_class_coverage: bool,
) -> dict[str, int]:
    available = sum(int(value) for value in capacities.values())
    if requested < 0 or requested > available:
        raise ValueError("requested split size exceeds remaining rows")
    allocation = {str(label): 0 for label in capacities}
    nonempty = [label for label in sorted(allocation) if int(capacities[label]) > 0]
    if require_class_coverage and requested >= len(nonempty):
        for label in nonempty:
            allocation[label] = 1

    remaining_requested = requested - sum(allocation.values())
    remaining_capacity = {
        label: int(capacities[label]) - allocation[label]
        for label in allocation
    }
    total_capacity = sum(remaining_capacity.values())
    if remaining_requested == 0:
        return allocation
    exact = {
        label: remaining_requested * capacity / total_capacity
        for label, capacity in remaining_capacity.items()
    }
    for label, value in exact.items():
        allocation[label] += min(remaining_capacity[label], math.floor(value))
    unallocated = requested - sum(allocation.values())
    order = sorted(
        allocation,
        key=lambda label: (
            -(exact[label] - math.floor(exact[label])),
            label,
        ),
    )
    while unallocated:
        progressed = False
        for label in order:
            if allocation[label] < int(capacities[label]):
                allocation[label] += 1
                unallocated -= 1
                progressed = True
                if unallocated == 0:
                    break
        if not progressed:
            raise AssertionError("could not allocate requested multiclass split")
    return allocation


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
