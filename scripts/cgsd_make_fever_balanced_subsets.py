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
from scripts.cgsd_make_fever_fixed_sets import approximate_k_center_greedy
from scripts.run_cgsd import load_embeddings
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


def build_random_ids(rows: list[dict[str, Any]], *, size: int, seed: int) -> list[str]:
    total = int(size)
    if total <= 0:
        raise ValueError("size must be positive")
    ids = [_row_id(row) for row in rows]
    return _choose_random(ids, k=total, seed=int(seed))


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


def build_accept_defer_random_ids(
    rows: list[dict[str, Any]],
    *,
    size: int,
    accept_fraction: float,
    seed: int,
) -> list[str]:
    total = int(size)
    if total <= 0:
        raise ValueError("size must be positive")
    accept_total = int(round(total * float(accept_fraction)))
    defer_total = total - accept_total
    if accept_total <= 0 or defer_total <= 0:
        raise ValueError("accept_fraction must leave at least one accept and one defer row")

    accept_ids = [_row_id(row) for row in rows if not bool(row.get("defer", False))]
    defer_ids = [_row_id(row) for row in rows if bool(row.get("defer", False))]
    selected = _choose_random(accept_ids, k=accept_total, seed=int(seed) + 10)
    selected.extend(_choose_random(defer_ids, k=defer_total, seed=int(seed) + 20))
    random.Random(int(seed) + 30).shuffle(selected)
    return selected


def build_accept_defer_kcenter_balanced_ids(
    rows: list[dict[str, Any]],
    *,
    size: int,
    accept_fraction: float,
    seed: int,
    embeddings_by_id: Mapping[str, Any],
    kcenter_candidate_count: int = 20_000,
    kcenter_projection_dim: int = 64,
) -> list[str]:
    """Select defer ids by k-center on the full defer pool, then fill accept to keep balance."""
    total = int(size)
    if total <= 0 or total % 2 != 0:
        raise ValueError("size must be a positive even integer for exact 1/0 balance")
    accept_total = int(round(total * float(accept_fraction)))
    defer_total = total - accept_total
    if accept_total <= 0 or defer_total <= 0:
        raise ValueError("accept_fraction must leave at least one accept and one defer row")

    accept_rows = [row for row in rows if not bool(row.get("defer", False))]
    defer_rows = [row for row in rows if bool(row.get("defer", False))]
    defer_ids = [_row_id(row) for row in defer_rows]
    rows_by_id = {_row_id(row): row for row in rows}

    defer_selected = approximate_k_center_greedy(
        sorted(defer_ids),
        embeddings_by_id,
        k=defer_total,
        seed=int(seed) + 200,
        candidate_count=len(defer_ids),
        projection_dim=int(kcenter_projection_dim),
    )
    defer_label_counts = {0: 0, 1: 0}
    selected: list[str] = list(defer_selected)
    for sample_id in defer_selected:
        defer_label_counts[_row_label(rows_by_id[sample_id])] += 1
    accept_budget = {label: total // 2 - defer_label_counts[label] for label in (0, 1)}
    if any(budget < 0 for budget in accept_budget.values()):
        raise ValueError(
            f"k-center selected too many defer rows for one label to preserve balance: {accept_budget}"
        )

    accept_buckets = _label_buckets(accept_rows)
    for label in (0, 1):
        selected.extend(
            _choose_random(
                accept_buckets[label],
                k=accept_budget[label],
                seed=int(seed) + 100 + label,
            )
        )
    random.Random(int(seed) + 300).shuffle(selected)
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


def parse_accept_fractions(text: str) -> list[float]:
    fractions = [float(part.strip()) for part in str(text or "").split(",") if part.strip()]
    if not fractions:
        raise ValueError("--accept_fraction cannot be empty")
    if any(fraction <= 0.0 or fraction >= 1.0 for fraction in fractions):
        raise ValueError("--accept_fraction values must be between 0 and 1")
    return fractions


def format_accept_defer_name(accept_fraction: float) -> str:
    accept_pct = int(round(float(accept_fraction) * 100))
    defer_pct = 100 - accept_pct
    return f"accept{accept_pct}_defer{defer_pct}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_path", default="experiments/inputs/fever/data.jsonl")
    parser.add_argument("--pool_crc_predictions_path", default="experiments/inputs/fever/round_0/pool_crc_predictions.jsonl")
    parser.add_argument("--output_dir", default="experiments/inputs/fever/balanced_lora_subsets_seed1")
    parser.add_argument("--test_size", type=int, default=10_000)
    parser.add_argument("--train_sizes", default="500,1000")
    parser.add_argument("--accept_fraction", default="0.15")
    parser.add_argument("--kcenter_accept_fraction", default="")
    parser.add_argument("--embeddings_path", default="experiments/inputs/fever/embeddings.npy")
    parser.add_argument("--kcenter_candidate_count", type=int, default=20_000)
    parser.add_argument("--kcenter_projection_dim", type=int, default=64)
    parser.add_argument("--unbalanced_only", action="store_true")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--teacher_beta", type=float, default=1.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    started = time.time()
    train_sizes = parse_sizes(args.train_sizes)
    accept_fractions = parse_accept_fractions(args.accept_fraction)
    kcenter_accept_fractions = parse_accept_fractions(args.kcenter_accept_fraction) if str(args.kcenter_accept_fraction).strip() else []
    if int(args.test_size) <= 0 or int(args.test_size) % 2 != 0:
        raise ValueError("--test_size must be a positive even integer")

    data_rows = read_jsonl(args.data_path)
    pool_rows = read_jsonl(args.pool_crc_predictions_path)
    embeddings_by_id = load_embeddings(Path(args.embeddings_path)) if kcenter_accept_fractions else None
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
        if not args.unbalanced_only:
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

            for fraction_index, accept_fraction in enumerate(accept_fractions):
                ratio_name = format_accept_defer_name(accept_fraction)
                mixed_name = f"{ratio_name}_random_balanced_{size}_seed{int(args.seed)}"
                mixed_ids = build_accept_defer_balanced_ids(
                    train_pool,
                    size=size,
                    accept_fraction=accept_fraction,
                    seed=int(args.seed) + size + 10_000 + (fraction_index * 1_000_000),
                )
                written[mixed_name] = _write_subset(
                    output_dir=output_dir,
                    name=mixed_name,
                    ids=mixed_ids,
                    rows_by_id=rows_by_id,
                    train_rows=True,
                    selection_role=f"{ratio_name}_random_balanced",
                    teacher_beta=float(args.teacher_beta),
                )
            for fraction_index, accept_fraction in enumerate(kcenter_accept_fractions):
                ratio_name = format_accept_defer_name(accept_fraction)
                kcenter_name = f"{ratio_name}_kcenter_balanced_{size}_seed{int(args.seed)}"
                kcenter_ids = build_accept_defer_kcenter_balanced_ids(
                    train_pool,
                    size=size,
                    accept_fraction=accept_fraction,
                    seed=int(args.seed) + size + 20_000 + (fraction_index * 1_000_000),
                    embeddings_by_id=embeddings_by_id or {},
                    kcenter_candidate_count=int(args.kcenter_candidate_count),
                    kcenter_projection_dim=int(args.kcenter_projection_dim),
                )
                written[kcenter_name] = _write_subset(
                    output_dir=output_dir,
                    name=kcenter_name,
                    ids=kcenter_ids,
                    rows_by_id=rows_by_id,
                    train_rows=True,
                    selection_role=f"{ratio_name}_kcenter_balanced",
                    teacher_beta=float(args.teacher_beta),
                )

        full_unbalanced_name = f"full_random_unbalanced_{size}_seed{int(args.seed)}"
        full_unbalanced_ids = build_random_ids(train_pool, size=size, seed=int(args.seed) + size + 30_000)
        written[full_unbalanced_name] = _write_subset(
            output_dir=output_dir,
            name=full_unbalanced_name,
            ids=full_unbalanced_ids,
            rows_by_id=rows_by_id,
            train_rows=True,
            selection_role="full_random_unbalanced",
            teacher_beta=float(args.teacher_beta),
        )

        for fraction_index, accept_fraction in enumerate(accept_fractions):
            ratio_name = format_accept_defer_name(accept_fraction)
            mixed_name = f"{ratio_name}_random_unbalanced_{size}_seed{int(args.seed)}"
            mixed_ids = build_accept_defer_random_ids(
                train_pool,
                size=size,
                accept_fraction=accept_fraction,
                seed=int(args.seed) + size + 40_000 + (fraction_index * 1_000_000),
            )
            written[mixed_name] = _write_subset(
                output_dir=output_dir,
                name=mixed_name,
                ids=mixed_ids,
                rows_by_id=rows_by_id,
                train_rows=True,
                selection_role=f"{ratio_name}_random_unbalanced",
                teacher_beta=float(args.teacher_beta),
            )

    summary = {
        "stage_name": "cgsd_make_fever_balanced_subsets",
        "seed": int(args.seed),
        "test_size": int(args.test_size),
        "train_sizes": train_sizes,
        "accept_fractions": accept_fractions,
        "defer_fractions": [1.0 - accept_fraction for accept_fraction in accept_fractions],
        "kcenter_accept_fractions": kcenter_accept_fractions,
        "kcenter_defer_fractions": [1.0 - accept_fraction for accept_fraction in kcenter_accept_fractions],
        "kcenter": {
            "embeddings_path": str(args.embeddings_path) if kcenter_accept_fractions else None,
            "candidate_count": int(args.kcenter_candidate_count),
            "projection_dim": int(args.kcenter_projection_dim),
            "method": "accept_defer_label_bucket_prefilter_then_projected_k_center",
        },
        "sets": written,
        "source_paths": {
            "data_path": str(args.data_path),
            "pool_crc_predictions_path": str(args.pool_crc_predictions_path),
        },
        "elapsed_seconds": round(time.time() - started, 2),
    }
    summary_name = "unbalanced_random_subset_summary.json" if args.unbalanced_only else "balanced_subset_summary.json"
    write_json(summary, output_dir / summary_name)
    print(json.dumps({"output_dir": str(output_dir), "sets": written}, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
