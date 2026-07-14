from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping
from typing import Any

from mias_dcms.binary_protocol import normalize_binary_label
from mias_dcms.binary_reaudit import _stratified_partition, _stratified_subset
from mias_dcms.selectors import assert_selector_rows_are_label_safe


def prepare_binary_benchmark_protocol(
    train_rows: Iterable[Mapping[str, Any]],
    *,
    validation_rows: Iterable[Mapping[str, Any]] | None,
    test_rows: Iterable[Mapping[str, Any]],
    dataset: str,
    seed_label_count: int,
    active_pool_size: int,
    seed: int,
    development_size: int = 0,
    train_row_limit: int | None = None,
    validation_row_limit: int | None = None,
    test_row_limit: int | None = None,
) -> dict[str, Any]:
    """Freeze a binary benchmark protocol without ever resplitting official test data.

    The active pool is stripped to selector-safe fields. Labels remain only in its
    separate oracle store until a method has selected sample ids.
    """
    if seed_label_count <= 0 or active_pool_size <= 0:
        raise ValueError("seed_label_count and active_pool_size must be positive")
    if development_size < 0:
        raise ValueError("development_size must be non-negative")

    train = _normalize_rows(train_rows, dataset=dataset, split="train")
    validation = _normalize_rows(validation_rows or [], dataset=dataset, split="validation")
    official_test = _normalize_rows(test_rows, dataset=dataset, split="test")
    _assert_global_unique_ids(train, validation, official_test)

    train = _limit_stratified(train, train_row_limit, seed=seed, name="train_row_limit")
    validation = _limit_stratified(
        validation,
        validation_row_limit,
        seed=seed + 1,
        name="validation_row_limit",
    )
    official_test = _limit_stratified(
        official_test,
        test_row_limit,
        seed=seed + 2,
        name="test_row_limit",
    )
    if not official_test:
        raise ValueError("official test rows must not be empty")

    if validation:
        if development_size:
            raise ValueError("development_size must be zero when official validation rows are supplied")
        seed_rows, active_rows, _ = _stratified_partition(
            train,
            split_sizes=(int(seed_label_count), int(active_pool_size), 0),
            seed=seed,
        )
        development_rows = validation
        development_source = "official_validation"
    else:
        if development_size <= 0:
            raise ValueError(
                "development_size must be positive when the dataset has no official validation split"
            )
        development_rows, seed_rows, active_rows = _stratified_partition(
            train,
            split_sizes=(int(development_size), int(seed_label_count), int(active_pool_size)),
            seed=seed,
        )
        development_source = "train_derived_fixed_holdout"

    selection_pool = [_selector_row(row) for row in active_rows]
    assert_selector_rows_are_label_safe(selection_pool)
    oracle_store = {str(row["id"]): {"label": int(row["label"])} for row in active_rows}
    split_ids = {
        "development_ids": [str(row["id"]) for row in development_rows],
        "seed_ids": [str(row["id"]) for row in seed_rows],
        "active_pool_ids": [str(row["id"]) for row in active_rows],
        "official_test_ids": [str(row["id"]) for row in official_test],
    }
    _assert_protocol_disjoint(split_ids)

    return {
        "seed_train_rows": [_training_row(row) for row in seed_rows],
        "selection_pool": selection_pool,
        "selection_oracle_store": oracle_store,
        "development_rows": [_training_row(row) for row in development_rows],
        "official_test_rows": [_training_row(row) for row in official_test],
        "protocol_manifest": {
            "schema_version": "binary-benchmark-protocol-v1",
            "dataset": str(dataset),
            "seed": int(seed),
            "label_policy": "native_groundtruth_only",
            "development_source": development_source,
            "official_test_policy": "fixed_source_split_never_used_for_selection_or_checkpoint_choice",
            "source_counts": {
                "train_available": len(train),
                "validation_available": len(validation),
                "official_test_available": len(official_test),
            },
            "source_label_counts": {
                "train": _label_counts(train),
                "validation": _label_counts(validation),
                "official_test": _label_counts(official_test),
            },
            "limits": {
                "train_row_limit": train_row_limit,
                "validation_row_limit": validation_row_limit,
                "test_row_limit": test_row_limit,
            },
            "split_sizes": {key.removesuffix("_ids"): len(value) for key, value in split_ids.items()},
            **split_ids,
        },
    }


def _normalize_rows(
    rows: Iterable[Mapping[str, Any]],
    *,
    dataset: str,
    split: str,
) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, source in enumerate(rows):
        row = dict(source)
        try:
            sample_id = str(row["id"])
            query = str(row["query"])
            document = str(row["document"])
            label = normalize_binary_label(row["groundtruth"], field_name="groundtruth")
        except KeyError as exc:
            raise ValueError(f"{split} row {index} is missing {exc.args[0]!r}") from exc
        if not sample_id:
            raise ValueError(f"{split} row {index} has an empty id")
        if not query.strip() or not document.strip():
            raise ValueError(f"{split} row {sample_id!r} has an empty query or document")
        if sample_id in seen:
            raise ValueError(f"duplicate {split} id {sample_id!r}")
        seen.add(sample_id)
        normalized.append(
            {
                "id": sample_id,
                "query": query,
                "document": document,
                "text": f"Query: {query}\n\nDocument: {document}",
                "label": label,
                "dataset": str(dataset),
            }
        )
    return normalized


def _limit_stratified(
    rows: list[dict[str, Any]],
    limit: int | None,
    *,
    seed: int,
    name: str,
) -> list[dict[str, Any]]:
    if limit is None:
        return rows
    if limit <= 0:
        raise ValueError(f"{name} must be positive when provided")
    if limit > len(rows):
        raise ValueError(f"{name} cannot exceed available rows")
    return _stratified_subset(rows, size=int(limit), seed=seed)


def _selector_row(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "id": str(row["id"]),
        "query": str(row["query"]),
        "document": str(row["document"]),
        "text": str(row["text"]),
        "dataset": str(row["dataset"]),
    }


def _training_row(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "id": str(row["id"]),
        "query": str(row["query"]),
        "document": str(row["document"]),
        "text": str(row["text"]),
        "label": int(row["label"]),
        "groundtruth": int(row["label"]),
        "dataset": str(row["dataset"]),
    }


def _assert_global_unique_ids(*splits: list[dict[str, Any]]) -> None:
    seen: dict[str, int] = {}
    for split_index, rows in enumerate(splits):
        for row in rows:
            sample_id = str(row["id"])
            previous = seen.get(sample_id)
            if previous is not None:
                raise ValueError(
                    f"sample id {sample_id!r} appears in both source splits {previous} and {split_index}"
                )
            seen[sample_id] = split_index


def _assert_protocol_disjoint(split_ids: Mapping[str, list[str]]) -> None:
    seen: dict[str, str] = {}
    for split_name, ids in split_ids.items():
        for sample_id in ids:
            previous = seen.get(sample_id)
            if previous is not None:
                raise AssertionError(f"protocol overlap: {sample_id!r} in {previous} and {split_name}")
            seen[sample_id] = split_name


def _label_counts(rows: Iterable[Mapping[str, Any]]) -> dict[str, int]:
    return {
        str(label): int(count)
        for label, count in sorted(Counter(int(row["label"]) for row in rows).items())
    }
