from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
import hashlib
import math
from pathlib import Path
import random
from typing import Any

from mias_dcms.auditing import mias_selection_audit
from mias_dcms.budgeting import BudgetInputs, build_budget_report
from mias_dcms.sampling_diagnostics import select_classification_rows, selector_safe_view
from mias_dcms.selectors import assert_selector_rows_are_label_safe


def prepare_binary_reaudit_splits(
    rows: Iterable[Mapping[str, Any]],
    *,
    dataset: str,
    seed_label_count: int,
    active_pool_size: int,
    test_size: int,
    seed: int,
    id_field: str = "id",
    query_field: str = "query",
    document_field: str = "document",
    label_field: str = "groundtruth",
    row_limit: int | None = None,
) -> dict[str, Any]:
    """Create a selector-safe binary fixed pool with separate oracle labels.

    Labels are used only for a deterministic stratified split. The resulting
    active-pool rows contain no ground-truth field and are safe to score/select.
    """
    normalized = _normalize_source_rows(
        rows,
        dataset=dataset,
        id_field=id_field,
        query_field=query_field,
        document_field=document_field,
        label_field=label_field,
    )
    if row_limit is not None:
        if row_limit <= 0:
            raise ValueError("row_limit must be positive when provided")
        if row_limit < len(normalized):
            normalized = _stratified_subset(normalized, size=row_limit, seed=seed)

    requested = int(seed_label_count) + int(active_pool_size) + int(test_size)
    if min(seed_label_count, active_pool_size, test_size) < 0:
        raise ValueError("split sizes must be non-negative")
    if requested > len(normalized):
        raise ValueError("requested split sizes exceed available rows")

    seed_rows, active_rows, test_rows = _stratified_partition(
        normalized,
        split_sizes=(int(seed_label_count), int(active_pool_size), int(test_size)),
        seed=seed,
    )
    source_by_id = {str(row["id"]): row for row in normalized}
    seed_ids = [str(row["id"]) for row in seed_rows]
    active_ids = [str(row["id"]) for row in active_rows]
    test_ids = [str(row["id"]) for row in test_rows]
    if len(set(seed_ids) | set(active_ids) | set(test_ids)) != requested:
        raise AssertionError("split ids must be unique")

    selector_pool = [_selector_row(row) for row in active_rows]
    assert_selector_rows_are_label_safe(selector_pool)
    return {
        "dataset": str(dataset),
        "seed": int(seed),
        "source_size": len(normalized),
        "source_label_counts": _label_counts(normalized),
        "seed_train_rows": [_training_row(row) for row in seed_rows],
        "selection_pool": selector_pool,
        "selection_oracle_store": {
            sample_id: {"label": int(source_by_id[sample_id]["label"])}
            for sample_id in active_ids
        },
        "test_rows": [_training_row(row) for row in test_rows],
        "split_manifest": {
            "dataset": str(dataset),
            "seed": int(seed),
            "id_field": "id",
            "seed_ids": seed_ids,
            "active_pool_ids": active_ids,
            "test_ids": test_ids,
            "split_sizes": {
                "seed": len(seed_ids),
                "active_pool": len(active_ids),
                "test": len(test_ids),
            },
            "source_label_counts": _label_counts(normalized),
        },
    }


def materialize_binary_reaudit_selection(
    scored_rows: Iterable[Mapping[str, Any]],
    *,
    oracle_store: Mapping[str, Mapping[str, Any]],
    seed_train_rows: Iterable[Mapping[str, Any]],
    dataset: str,
    model: str,
    methods: Sequence[str],
    budget: int,
    seed: int,
    config_hash: str,
    evaluation_label_count: int,
) -> dict[str, dict[str, Any]]:
    """Select only from label-safe scores, then materialize post-selection audits."""
    scored = [dict(row) for row in scored_rows]
    if not scored:
        raise ValueError("scored_rows must not be empty")
    ids = [str(row.get("id", "")) for row in scored]
    if not all(ids) or len(set(ids)) != len(ids):
        raise ValueError("scored rows must contain unique non-empty ids")
    missing_oracles = sorted(set(ids) - set(str(key) for key in oracle_store))
    extra_oracles = sorted(set(str(key) for key in oracle_store) - set(ids))
    if missing_oracles or extra_oracles:
        raise ValueError(
            "oracle store must exactly cover scored rows: "
            f"missing={len(missing_oracles)}, extra={len(extra_oracles)}"
        )
    if budget <= 0 or budget > len(scored):
        raise ValueError("budget must be between 1 and scored row count")

    safe_rows = selector_safe_view(scored)
    assert_selector_rows_are_label_safe(safe_rows)
    seed_rows = [_validated_training_row(row) for row in seed_train_rows]
    output: dict[str, dict[str, Any]] = {}
    for method in methods:
        normalized_method = str(method).strip()
        if not normalized_method:
            continue
        selected_rows, selection_metadata = select_classification_rows(
            safe_rows,
            method=normalized_method,
            budget=int(budget),
            seed=int(seed),
        )
        selected_ids = {str(row["id"]) for row in selected_rows}
        if len(selected_ids) != budget:
            raise AssertionError("selection did not return exactly budget unique ids")

        membership = []
        revealed_rows = []
        for row in scored:
            sample_id = str(row["id"])
            label = _oracle_label(oracle_store[sample_id])
            selected = sample_id in selected_ids
            audit_row = {
                **row,
                "oracle_label": label,
                "selected": selected,
                "method": normalized_method,
            }
            membership.append(audit_row)
            if selected:
                revealed_rows.append(
                    {
                        "id": sample_id,
                        "text": str(row["text"]),
                        "label": label,
                        "dataset": str(dataset),
                    }
                )
        audit = mias_selection_audit(
            membership,
            group_field="oracle_label",
            selected_field="selected",
        )
        train_rows = [*seed_rows, *revealed_rows]
        if len({str(row["id"]) for row in train_rows}) != len(train_rows):
            raise ValueError("seed and active train rows overlap")
        budget_report = build_budget_report(
            BudgetInputs(
                method=normalized_method,
                seed_label_count=len(seed_rows),
                active_label_count=len(revealed_rows),
                evaluation_label_count=int(evaluation_label_count),
            )
        )
        output[normalized_method] = {
            "dataset": str(dataset),
            "model": str(model),
            "method": normalized_method,
            "budget": int(budget),
            "seed": int(seed),
            "config_hash": str(config_hash),
            "selected_ids": sorted(selected_ids),
            "membership": membership,
            "revealed_rows": revealed_rows,
            "train_rows": train_rows,
            "selection_metrics": audit.as_dict(),
            "cost_metrics": budget_report.as_dict(),
            "selection_metadata": selection_metadata,
        }
    if not output:
        raise ValueError("at least one selection method is required")
    return output


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _normalize_source_rows(
    rows: Iterable[Mapping[str, Any]],
    *,
    dataset: str,
    id_field: str,
    query_field: str,
    document_field: str,
    label_field: str,
) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, source in enumerate(rows):
        row = dict(source)
        try:
            sample_id = str(row[id_field])
            query = str(row[query_field])
            document = str(row[document_field])
            label = _binary_label(row[label_field])
        except KeyError as exc:
            raise ValueError(f"source row {index} is missing {exc.args[0]!r}") from exc
        if not sample_id:
            raise ValueError(f"source row {index} has an empty id")
        if sample_id in seen:
            raise ValueError(f"duplicate source id {sample_id!r}")
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
    if not normalized:
        raise ValueError("source rows must not be empty")
    return normalized


def _stratified_subset(rows: list[dict[str, Any]], *, size: int, seed: int) -> list[dict[str, Any]]:
    if size > len(rows):
        raise ValueError("row_limit cannot exceed source row count")
    selected, _, _ = _stratified_partition(rows, split_sizes=(size, 0, 0), seed=seed)
    return selected


def _stratified_partition(
    rows: list[dict[str, Any]],
    *,
    split_sizes: tuple[int, int, int],
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    by_label: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        by_label.setdefault(int(row["label"]), []).append(dict(row))
    for label, items in by_label.items():
        random.Random(f"{seed}:{label}").shuffle(items)

    remaining = {label: list(items) for label, items in by_label.items()}
    partitions: list[list[dict[str, Any]]] = []
    for split_index, requested in enumerate(split_sizes):
        if requested == 0:
            partitions.append([])
            continue
        allocations = _proportional_allocation(
            requested,
            {label: len(items) for label, items in remaining.items()},
        )
        part: list[dict[str, Any]] = []
        for label in sorted(allocations):
            count = allocations[label]
            part.extend(remaining[label][:count])
            remaining[label] = remaining[label][count:]
        random.Random(f"{seed}:split:{split_index}").shuffle(part)
        partitions.append(part)
    return partitions[0], partitions[1], partitions[2]


def _proportional_allocation(total: int, capacities: Mapping[int, int]) -> dict[int, int]:
    if total < 0:
        raise ValueError("total must be non-negative")
    available = sum(int(value) for value in capacities.values())
    if total > available:
        raise ValueError("requested split size exceeds remaining rows")
    if total == 0:
        return {int(label): 0 for label in capacities}
    exact = {int(label): total * int(capacity) / available for label, capacity in capacities.items()}
    allocation = {
        label: min(int(capacities[label]), math.floor(value))
        for label, value in exact.items()
    }
    remaining = total - sum(allocation.values())
    order = sorted(
        allocation,
        key=lambda label: (-(exact[label] - math.floor(exact[label])), label),
    )
    while remaining:
        progressed = False
        for label in order:
            if allocation[label] < int(capacities[label]):
                allocation[label] += 1
                remaining -= 1
                progressed = True
                if not remaining:
                    break
        if not progressed:
            raise AssertionError("could not allocate requested split size")
    return allocation


def _selector_row(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "id": str(row["id"]),
        "query": str(row["query"]),
        "document": str(row["document"]),
        "text": str(row["text"]),
        "dataset": str(row["dataset"]),
    }


def _training_row(row: Mapping[str, Any]) -> dict[str, Any]:
    training_row = {
        "id": str(row["id"]),
        "text": str(row["text"]),
        "label": int(row["label"]),
        "dataset": str(row["dataset"]),
    }
    if "query" in row:
        training_row["query"] = str(row["query"])
    if "document" in row:
        training_row["document"] = str(row["document"])
    return training_row


def _validated_training_row(row: Mapping[str, Any]) -> dict[str, Any]:
    if not row.get("id") or "text" not in row or "label" not in row:
        raise ValueError("seed_train_rows must contain id, text, and label")
    return _training_row(row)


def _binary_label(value: Any) -> int:
    if isinstance(value, bool):
        return int(value)
    try:
        label = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"unsupported binary label {value!r}") from exc
    if label not in (0, 1):
        raise ValueError(f"binary label must be 0 or 1, got {value!r}")
    return label


def _oracle_label(payload: Mapping[str, Any]) -> int:
    if "label" not in payload:
        raise ValueError("oracle entry is missing label")
    return _binary_label(payload["label"])


def _label_counts(rows: Iterable[Mapping[str, Any]]) -> dict[str, int]:
    return {str(label): int(count) for label, count in sorted(Counter(int(row["label"]) for row in rows).items())}
