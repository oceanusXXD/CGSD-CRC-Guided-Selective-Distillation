#!/usr/bin/env python
"""Generate balanced FEVER test and small LoRA training subsets."""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Mapping

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from algorithms.cgsd import teacher_weight
from scripts.cgsd_cli_common import binary_to_int, read_jsonl
from src.utils import write_json, write_jsonl


def _row_id(row: Mapping[str, Any]) -> str:
    sample_id = row.get("id", row.get("sample_id"))
    if sample_id is None or str(sample_id) == "":
        raise ValueError(f"row is missing id/sample_id: {row!r}")
    return str(sample_id)


def _row_label(row: Mapping[str, Any]) -> int:
    return binary_to_int(row.get("teacher_label", row.get("label", row.get("groundtruth"))), field_name=f"row {_row_id(row)!r} label")


def _choose_random(ids: Iterable[str], *, k: int, seed: int) -> list[str]:
    candidates = list(ids)
    if int(k) > len(candidates):
        raise ValueError(f"requested {k} ids from only {len(candidates)} candidates")
    random.Random(int(seed)).shuffle(candidates)
    return candidates[: int(k)]


def _label_buckets(rows: Iterable[Mapping[str, Any]]) -> dict[int, list[str]]:
    buckets = {0: [], 1: []}
    for row in rows:
        buckets[_row_label(row)].append(_row_id(row))
    for ids in buckets.values():
        ids.sort()
    return buckets


def build_label_balanced_ids(rows: list[dict[str, Any]], *, size: int, seed: int) -> list[str]:
    """Select exactly half positive and half negative ids."""
    total = int(size)
    if total <= 0 or total % 2 != 0:
        raise ValueError("size must be a positive even integer for exact 1/0 balance")
    buckets = _label_buckets(rows)
    per_label = total // 2
    selected_0 = _choose_random(buckets[0], k=per_label, seed=int(seed))
    selected_1 = _choose_random(buckets[1], k=per_label, seed=int(seed) + 1)
    merged = selected_0 + selected_1
    random.Random(int(seed) + 2).shuffle(merged)
    return merged


def _split_label_budget(total: int) -> dict[int, int]:
    low = int(total) // 2
    return {0: low, 1: int(total) - low}


def build_accept_defer_balanced_ids(
    rows: list[dict[str, Any]],
    *,
    size: int,
    accept_fraction: float,
    seed: int,
) -> list[str]:
    """Select ids with an accept/defer ratio and exact overall 1/0 balance."""
    total = int(size)
    if total <= 0 or total % 2 != 0:
        raise ValueError("size must be a positive even integer for exact 1/0 balance")
    accept_total = int(round(total * float(accept_fraction)))
    defer_total = total - accept_total
    if accept_total <= 0 or defer_total <= 0:
        raise ValueError("accept_fraction must leave at least one accept and one defer row")

    accept_rows = [row for row in rows if not bool(row.get("defer", False))]
    defer_rows = [row for row in rows if bool(row.get("defer", False))]
    accept_buckets = _label_buckets(accept_rows)
    defer_buckets = _label_buckets(defer_rows)
    accept_budget = _split_label_budget(accept_total)
    defer_budget = {label: total // 2 - accept_budget[label] for label in (0, 1)}

    selected: list[str] = []
    for label in (0, 1):
        selected.extend(_choose_random(accept_buckets[label], k=accept_budget[label], seed=int(seed) + 10 + label))
        selected.extend(_choose_random(defer_buckets[label], k=defer_budget[label], seed=int(seed) + 20 + label))
    random.Random(int(seed) + 30).shuffle(selected)
    return selected


def _materialize_train_rows(
    ids: list[str],
    *,
    rows_by_id: Mapping[str, dict[str, Any]],
    selection_role: str,
    teacher_beta: float,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for sample_id in ids:
        payload = dict(rows_by_id[sample_id])
        label = _row_label(payload)
        payload["label"] = label
        payload["groundtruth"] = label
        payload["teacher_label"] = label
        payload.setdefault("teacher_confidence", float(payload.get("parsed_confidence", 1.0) or 1.0))
        payload.setdefault("teacher_source", "groundtruth_substitute_for_real_teacher_api")
        payload.setdefault("teacher_label_source", "groundtruth")
        payload.setdefault("teacher_confidence_source", "fixed_1.0_groundtruth_substitute")
        payload["sample_weight"] = teacher_weight(float(payload["teacher_confidence"]), float(teacher_beta))
        payload["selection_round"] = 0
        payload["selection_role"] = selection_role
        rows.append(payload)
    return rows


def _counts(ids: list[str], rows_by_id: Mapping[str, dict[str, Any]]) -> dict[str, Any]:
    labels = [_row_label(rows_by_id[sample_id]) for sample_id in ids]
    accept = sum(1 for sample_id in ids if not bool(rows_by_id[sample_id].get("defer", False)))
    defer = len(ids) - accept
    return {
        "total": len(ids),
        "label_0": labels.count(0),
        "label_1": labels.count(1),
        "accept": accept,
        "defer": defer,
    }


def _write_subset(
    *,
    output_dir: Path,
    name: str,
    ids: list[str],
    rows_by_id: Mapping[str, dict[str, Any]],
    train_rows: bool,
    selection_role: str,
    teacher_beta: float,
) -> dict[str, Any]:
    write_json({"name": name, "count": len(ids), "ids": ids}, output_dir / f"{name}.ids.json")
    if train_rows:
        write_jsonl(
            _materialize_train_rows(ids, rows_by_id=rows_by_id, selection_role=selection_role, teacher_beta=teacher_beta),
            output_dir / f"{name}.train_rows.jsonl",
        )
    write_jsonl([dict(rows_by_id[sample_id]) for sample_id in ids], output_dir / f"{name}.jsonl")
    return _counts(ids, rows_by_id)


def parse_sizes(text: str) -> list[int]:
    sizes = [int(part.strip()) for part in str(text or "").split(",") if part.strip()]
    if not sizes:
        raise ValueError("--train_sizes cannot be empty")
    if any(size <= 0 or size % 2 != 0 for size in sizes):
        raise ValueError("--train_sizes must contain positive even integers")
    return sizes


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_path", default="experiments/inputs/fever/data.jsonl")
    parser.add_argument("--pool_crc_predictions_path", default="experiments/inputs/fever/round_0/pool_crc_predictions.jsonl")
    parser.add_argument("--output_dir", default="experiments/inputs/fever/balanced_lora_subsets_seed1")
    parser.add_argument("--test_size", type=int, default=10_000)
    parser.add_argument("--train_sizes", default="500,1000")
    parser.add_argument("--accept_fraction", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--teacher_beta", type=float, default=1.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    started = time.time()
    train_sizes = parse_sizes(args.train_sizes)
    if int(args.test_size) <= 0 or int(args.test_size) % 2 != 0:
        raise ValueError("--test_size must be a positive even integer")

    data_rows = read_jsonl(args.data_path)
    pool_rows = read_jsonl(args.pool_crc_predictions_path)
    rows_by_id = {_row_id(row): dict(row) for row in [*data_rows, *pool_rows]}
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    written: dict[str, Any] = {}
    test_name = f"balanced_test_{int(args.test_size)}_seed{int(args.seed)}"
    test_ids = build_label_balanced_ids(data_rows, size=int(args.test_size), seed=int(args.seed) + 1000)
    written[test_name] = _write_subset(
        output_dir=output_dir,
        name=test_name,
        ids=test_ids,
        rows_by_id=rows_by_id,
        train_rows=False,
        selection_role="balanced_test",
        teacher_beta=float(args.teacher_beta),
    )

    test_block = set(test_ids)
    train_pool = [row for row in pool_rows if _row_id(row) not in test_block]
    for size in train_sizes:
        full_name = f"full_random_balanced_{size}_seed{int(args.seed)}"
        full_ids = build_label_balanced_ids(train_pool, size=size, seed=int(args.seed) + size)
        written[full_name] = _write_subset(
            output_dir=output_dir,
            name=full_name,
            ids=full_ids,
            rows_by_id=rows_by_id,
            train_rows=True,
            selection_role="full_random_balanced",
            teacher_beta=float(args.teacher_beta),
        )

        mixed_name = f"accept15_defer85_random_balanced_{size}_seed{int(args.seed)}"
        mixed_ids = build_accept_defer_balanced_ids(
            train_pool,
            size=size,
            accept_fraction=float(args.accept_fraction),
            seed=int(args.seed) + size + 10_000,
        )
        written[mixed_name] = _write_subset(
            output_dir=output_dir,
            name=mixed_name,
            ids=mixed_ids,
            rows_by_id=rows_by_id,
            train_rows=True,
            selection_role="accept15_defer85_random_balanced",
            teacher_beta=float(args.teacher_beta),
        )

    summary = {
        "stage_name": "cgsd_make_fever_balanced_subsets",
        "seed": int(args.seed),
        "test_size": int(args.test_size),
        "train_sizes": train_sizes,
        "accept_fraction": float(args.accept_fraction),
        "defer_fraction": 1.0 - float(args.accept_fraction),
        "sets": written,
        "source_paths": {
            "data_path": str(args.data_path),
            "pool_crc_predictions_path": str(args.pool_crc_predictions_path),
        },
        "elapsed_seconds": round(time.time() - started, 2),
    }
    write_json(summary, output_dir / "balanced_subset_summary.json")
    print(json.dumps({"output_dir": str(output_dir), "sets": written}, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
