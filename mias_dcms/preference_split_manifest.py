from __future__ import annotations

from collections.abc import Iterable, Mapping
import random
from typing import Any


PREFERENCE_SPLIT_ID_FIELDS = {
    "seed": "seed_ids",
    "selection": "active_pool_ids",
    "active": "active_pool_ids",
    "heldout": "heldout_ids",
    "test": "test_ids",
    "unused": "unused_ids",
}


def build_preference_split_manifest(
    rows: Iterable[Mapping[str, Any]],
    *,
    seed: int,
    seed_size: int,
    active_size: int,
    heldout_size: int,
    test_size: int,
    id_field: str = "sample_id",
    prompt_field: str = "prompt",
    enforce_prompt_disjoint: bool = True,
    group_field: str = "swap_pair_id",
) -> dict[str, Any]:
    materialized = [dict(row) for row in rows]
    row_ids = [_row_id(row, id_field=id_field, row_index=index) for index, row in enumerate(materialized)]
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

    prompt_available = bool(enforce_prompt_disjoint and all(prompt_field in row for row in materialized))
    split_ids, group_disjoint_enforced = _split_ids_with_optional_groups(
        materialized,
        row_ids=row_ids,
        requested=requested,
        seed=int(seed),
        group_field=str(group_field),
        prompt_field=str(prompt_field) if prompt_available else "",
    )
    if prompt_available:
        split_by_id = {
            sample_id: split_name
            for split_name, ids in split_ids.items()
            for sample_id in ids
        }
        prompt_to_splits: dict[str, set[str]] = {}
        for row in materialized:
            sample_id = str(row.get(id_field, row.get("id")))
            split_name = split_by_id.get(sample_id)
            if split_name is None:
                continue
            prompt_to_splits.setdefault(str(row[prompt_field]), set()).add(split_name)
        leaked_prompts = sorted(
            prompt for prompt, split_names in prompt_to_splits.items() if len(split_names) > 1
        )
        if leaked_prompts:
            raise ValueError(
                f"prompt leakage across preference splits: {len(leaked_prompts)} prompts, "
                f"examples={leaked_prompts[:3]}"
            )

    return {
        "seed": int(seed),
        "row_count": len(row_ids),
        "id_field": str(id_field),
        "prompt_field": str(prompt_field),
        "prompt_disjoint_enforced": prompt_available,
        "group_field": str(group_field),
        "group_disjoint_enforced": bool(group_disjoint_enforced),
        **split_ids,
        "unused_ids": sorted(set(row_ids) - set().union(*split_ids.values())),
    }


def materialize_preference_split_rows(
    rows: Iterable[Mapping[str, Any]],
    split_manifest: Mapping[str, Any],
    *,
    split: str,
    id_field: str = "sample_id",
) -> list[dict[str, Any]]:
    """Return one fixed preference split in manifest order.

    Selection and training commands receive a physical split-specific file
    rather than the full pool, preventing held-out or test IDs from silently
    entering a selector input.
    """
    source_rows = [dict(row) for row in rows]
    rows_by_id: dict[str, dict[str, Any]] = {}
    for index, row in enumerate(source_rows):
        row_id = _row_id(row, id_field=id_field, row_index=index)
        if row_id in rows_by_id:
            raise ValueError(f"preference rows contain duplicate id {row_id!r}")
        rows_by_id[row_id] = row
    split_ids = preference_split_ids(split_manifest, split=split)
    _validate_split_rows_exist(rows_by_id, split_ids, split=split)
    return [dict(rows_by_id[row_id]) for row_id in split_ids]


def materialize_preference_split_oracle_store(
    oracle_store: Mapping[str, Mapping[str, Any]],
    split_manifest: Mapping[str, Any],
    *,
    split: str,
) -> dict[str, dict[str, Any]]:
    """Return only oracle entries corresponding to a fixed split."""
    normalized = {str(sample_id): dict(row) for sample_id, row in oracle_store.items()}
    split_ids = preference_split_ids(split_manifest, split=split)
    missing = [sample_id for sample_id in split_ids if sample_id not in normalized]
    if missing:
        raise ValueError(f"oracle store is missing {len(missing)} {split} split ids; example={missing[0]!r}")
    return {sample_id: normalized[sample_id] for sample_id in split_ids}


def preference_split_ids(split_manifest: Mapping[str, Any], *, split: str) -> list[str]:
    normalized_split = str(split).strip().lower()
    field_name = PREFERENCE_SPLIT_ID_FIELDS.get(normalized_split)
    if field_name is None:
        raise ValueError(f"unsupported preference split: {split!r}")
    values = split_manifest.get(field_name)
    if not isinstance(values, Iterable) or isinstance(values, (str, bytes, Mapping)):
        raise ValueError(f"split manifest field {field_name!r} must be a list of ids")
    ids = [str(value) for value in values]
    if len(set(ids)) != len(ids):
        raise ValueError(f"split manifest field {field_name!r} contains duplicate ids")
    return ids


def _validate_split_rows_exist(
    rows_by_id: Mapping[str, Mapping[str, Any]],
    split_ids: list[str],
    *,
    split: str,
) -> bool:
    missing = [row_id for row_id in split_ids if row_id not in rows_by_id]
    if missing:
        raise ValueError(f"input rows are missing {len(missing)} {split} split ids; example={missing[0]!r}")
    return True


def _split_ids_with_optional_groups(
    rows: list[dict[str, Any]],
    *,
    row_ids: list[str],
    requested: Mapping[str, int],
    seed: int,
    group_field: str,
    prompt_field: str,
) -> tuple[dict[str, list[str]], bool]:
    groups, grouped = _build_disjoint_components(
        rows,
        row_ids=row_ids,
        group_field=group_field,
        prompt_field=prompt_field,
    )
    if grouped:
        shuffled_groups = list(groups)
        random.Random(int(seed)).shuffle(shuffled_groups)
        remaining_groups = shuffled_groups
        split_ids: dict[str, list[str]] = {}
        for split_name, requested_size in requested.items():
            selected, selected_indexes = _take_components_exact(
                remaining_groups,
                target=int(requested_size),
                context="prompt leakage / group-disjoint split",
            )
            selected_index_set = set(selected_indexes)
            remaining_groups = [
                group for index, group in enumerate(remaining_groups) if index not in selected_index_set
            ]
            split_ids[split_name] = sorted(row_id for group in selected for row_id in group)
        return _rename_split_keys(split_ids), True

    requested_total = sum(int(value) for value in requested.values())
    shuffled = list(row_ids)
    random.Random(int(seed)).shuffle(shuffled)
    cursor = 0
    split_ids = {}
    for split_name, requested_size in requested.items():
        split_ids[split_name] = sorted(shuffled[cursor : cursor + int(requested_size)])
        cursor += int(requested_size)
    if cursor != requested_total:
        raise AssertionError("split cursor does not match requested row count")
    return _rename_split_keys(split_ids), False


def _build_disjoint_components(
    rows: list[dict[str, Any]],
    *,
    row_ids: list[str],
    group_field: str,
    prompt_field: str,
) -> tuple[list[list[str]], bool]:
    has_group = bool(group_field) and all(group_field in row for row in rows)
    has_prompt = bool(prompt_field) and all(prompt_field in row for row in rows)
    if not has_group and not has_prompt:
        return [], False

    parent = list(range(len(rows)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    key_to_index: dict[tuple[str, str], int] = {}
    for index, row in enumerate(rows):
        keys: list[tuple[str, str]] = []
        if has_group:
            keys.append(("group", str(row[group_field])))
        if has_prompt:
            keys.append(("prompt", str(row[prompt_field])))
        for key in keys:
            previous = key_to_index.get(key)
            if previous is None:
                key_to_index[key] = index
            else:
                union(index, previous)

    components: dict[int, list[str]] = {}
    for index, row_id in enumerate(row_ids):
        components.setdefault(find(index), []).append(row_id)
    return list(components.values()), True


def _take_components_exact(
    components: list[list[str]],
    *,
    target: int,
    context: str,
) -> tuple[list[list[str]], list[int]]:
    if target < 0:
        raise ValueError("split sizes must be non-negative")
    if target == 0:
        return [], []
    if not components:
        raise ValueError(f"cannot satisfy {context}: no components remain for target size {target}")

    sizes = [len(component) for component in components]
    unique_sizes = sorted(set(sizes))
    if len(unique_sizes) == 1:
        size = unique_sizes[0]
        if target % size:
            raise ValueError(
                f"cannot satisfy {context}: target size {target} is not divisible by component size {size}"
            )
        count = target // size
        if count > len(components):
            raise ValueError(f"cannot satisfy {context}: target size {target} exceeds remaining components")
        return components[:count], list(range(count))

    if len(unique_sizes) == 2:
        small, large = unique_sizes
        by_size = {
            small: [index for index, size in enumerate(sizes) if size == small],
            large: [index for index, size in enumerate(sizes) if size == large],
        }
        for large_count in range(min(len(by_size[large]), target // large), -1, -1):
            remainder = target - large_count * large
            if remainder < 0 or remainder % small:
                continue
            small_count = remainder // small
            if small_count <= len(by_size[small]):
                indexes = by_size[large][:large_count] + by_size[small][:small_count]
                return [components[index] for index in indexes], indexes

    reachable: dict[int, list[int]] = {0: []}
    for index, size in enumerate(sizes):
        if size > target:
            continue
        for total, selected_indexes in list(reachable.items()):
            next_total = total + size
            if next_total <= target and next_total not in reachable:
                reachable[next_total] = [*selected_indexes, index]
        if target in reachable:
            indexes = reachable[target]
            return [components[index] for index in indexes], indexes
    raise ValueError(
        f"cannot satisfy {context}: target size {target} is not reachable from component sizes {unique_sizes[:12]}"
    )


def _rename_split_keys(split_ids: Mapping[str, list[str]]) -> dict[str, list[str]]:
    names = {
        "seed_size": "seed_ids",
        "active_size": "active_pool_ids",
        "heldout_size": "heldout_ids",
        "test_size": "test_ids",
    }
    return {names.get(str(name), str(name)): list(values) for name, values in split_ids.items()}


def _row_id(row: Mapping[str, Any], *, id_field: str, row_index: int) -> str:
    value = row.get(id_field, row.get("id"))
    if value is None:
        raise ValueError(f"row {row_index} is missing id field {id_field!r} and fallback 'id'")
    return str(value)
