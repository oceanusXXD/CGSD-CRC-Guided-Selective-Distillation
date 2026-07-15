from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping
import hashlib
import math
import random
from typing import Any

from mias_dcms.binary_protocol import normalize_binary_label


def prepare_query_binary_source(
    rows: Iterable[Mapping[str, Any]],
    *,
    dataset: str,
    seed: int,
    test_size: int,
    expected_query_id: str | None = None,
) -> dict[str, Any]:
    """Sanitize a single-query binary dataset and freeze a document-disjoint holdout."""
    normalized, query_ids, query_texts = _normalize_rows(
        rows,
        dataset=dataset,
        expected_query_id=expected_query_id,
    )
    deduplicated, duplicate_ids = _deduplicate_exact_documents(normalized)
    if test_size <= 0 or test_size >= len(deduplicated):
        raise ValueError("test_size must be positive and smaller than the deduplicated source size")

    test_rows, train_rows = _stratified_partition(
        deduplicated,
        split_sizes=(int(test_size), len(deduplicated) - int(test_size)),
        seed=seed,
    )
    _assert_disjoint_documents(train_rows, test_rows)
    return {
        "source_train_rows": train_rows,
        "source_test_rows": test_rows,
        "source_summary": {
            "schema_version": "query-binary-source-v1",
            "dataset": str(dataset),
            "seed": int(seed),
            "query_ids": query_ids,
            "query_sha256": _sha256_text(query_texts[0]),
            "source_row_count": len(normalized),
            "deduplicated_row_count": len(deduplicated),
            "dropped_exact_duplicate_count": len(duplicate_ids),
            "dropped_exact_duplicate_ids": duplicate_ids,
            "label_counts_after_deduplication": _label_counts(deduplicated),
            "split_policy": "deterministic_stratified_document_disjoint_holdout",
            "split_sizes": {"train": len(train_rows), "test": len(test_rows)},
            "split_label_counts": {
                "train": _label_counts(train_rows),
                "test": _label_counts(test_rows),
            },
            "excluded_source_fields": ["parsed_answer", "parsed_confidence"],
        },
    }


def _normalize_rows(
    rows: Iterable[Mapping[str, Any]],
    *,
    dataset: str,
    expected_query_id: str | None,
) -> tuple[list[dict[str, Any]], list[str], list[str]]:
    normalized: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    query_ids: set[str] = set()
    query_texts: set[str] = set()
    for index, source in enumerate(rows):
        try:
            sample_id = str(source["id"])
            query = str(source["query"]).strip()
            document = str(source["document"]).strip()
            label = normalize_binary_label(source["groundtruth"], field_name="groundtruth")
        except KeyError as exc:
            raise ValueError(f"source row {index} is missing {exc.args[0]!r}") from exc
        query_id = str(source.get("query_id", "")).strip()
        document_id = str(source.get("document_id", "")).strip()
        if not sample_id or sample_id in seen_ids:
            raise ValueError(f"source row {index} has a missing or duplicate id")
        if not query_id or not query or not document:
            raise ValueError(f"source row {sample_id!r} has empty query metadata or document text")
        if expected_query_id is not None and query_id != str(expected_query_id):
            raise ValueError(
                f"source row {sample_id!r} has query_id={query_id!r}; expected {expected_query_id!r}"
            )
        seen_ids.add(sample_id)
        query_ids.add(query_id)
        query_texts.add(query)
        normalized.append(
            {
                "id": sample_id,
                "query": query,
                "document": document,
                "groundtruth": label,
                "dataset": str(dataset),
                "source_query_id": query_id,
                "source_document_id": document_id,
                "document_sha256": _sha256_text(_normalize_document(document)),
            }
        )
    if not normalized:
        raise ValueError("source rows must not be empty")
    if len(query_ids) != 1 or len(query_texts) != 1:
        raise ValueError("a query binary benchmark must contain exactly one query id and query text")
    return normalized, sorted(query_ids), sorted(query_texts)


def _deduplicate_exact_documents(
    rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    retained: dict[str, dict[str, Any]] = {}
    duplicate_ids: list[dict[str, str]] = []
    for row in sorted(rows, key=lambda item: str(item["id"])):
        key = str(row["document_sha256"])
        previous = retained.get(key)
        if previous is None:
            retained[key] = row
            continue
        if int(previous["groundtruth"]) != int(row["groundtruth"]):
            raise ValueError(
                "exactly duplicated documents have conflicting labels: "
                f"{previous['id']!r} and {row['id']!r}"
            )
        duplicate_ids.append({"dropped_id": str(row["id"]), "retained_id": str(previous["id"])})
    return list(retained.values()), duplicate_ids


def _stratified_partition(
    rows: list[dict[str, Any]],
    *,
    split_sizes: tuple[int, int],
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if sum(split_sizes) != len(rows) or min(split_sizes) < 0:
        raise ValueError("split sizes must partition the source rows")
    by_label: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        by_label.setdefault(int(row["groundtruth"]), []).append(dict(row))
    for label, items in by_label.items():
        random.Random(f"{seed}:{label}").shuffle(items)

    remaining = {label: list(items) for label, items in by_label.items()}
    partitions: list[list[dict[str, Any]]] = []
    for split_index, requested in enumerate(split_sizes):
        allocations = _proportional_allocation(
            requested,
            {label: len(items) for label, items in remaining.items()},
        )
        partition: list[dict[str, Any]] = []
        for label in sorted(allocations):
            count = allocations[label]
            partition.extend(remaining[label][:count])
            remaining[label] = remaining[label][count:]
        random.Random(f"{seed}:split:{split_index}").shuffle(partition)
        partitions.append(partition)
    return partitions[0], partitions[1]


def _proportional_allocation(total: int, capacities: Mapping[int, int]) -> dict[int, int]:
    available = sum(int(value) for value in capacities.values())
    if total < 0 or total > available:
        raise ValueError("requested split size exceeds remaining rows")
    if total == 0:
        return {int(label): 0 for label in capacities}
    exact = {int(label): total * int(capacity) / available for label, capacity in capacities.items()}
    allocation = {label: min(int(capacities[label]), math.floor(value)) for label, value in exact.items()}
    remaining = total - sum(allocation.values())
    order = sorted(allocation, key=lambda label: (-(exact[label] - math.floor(exact[label])), label))
    while remaining:
        for label in order:
            if allocation[label] < int(capacities[label]):
                allocation[label] += 1
                remaining -= 1
                if remaining == 0:
                    break
    return allocation


def _assert_disjoint_documents(*splits: list[dict[str, Any]]) -> None:
    seen: set[str] = set()
    for rows in splits:
        document_hashes = {str(row["document_sha256"]) for row in rows}
        if seen & document_hashes:
            raise AssertionError("document leakage across source splits")
        seen.update(document_hashes)


def _label_counts(rows: Iterable[Mapping[str, Any]]) -> dict[str, int]:
    return {
        str(label): int(count)
        for label, count in sorted(Counter(int(row["groundtruth"]) for row in rows).items())
    }


def _normalize_document(document: str) -> str:
    return " ".join(document.split())


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()
