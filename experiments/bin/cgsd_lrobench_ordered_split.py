#!/usr/bin/env python
"""Build the ordered LROBench query-stratified CGSD split."""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

from src.binary_protocol import normalize_binary_label


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def label_value(row: dict[str, Any]) -> int:
    value = row.get("groundtruth", row.get("label"))
    return normalize_binary_label(value, field_name=f"label for row {row.get('id')!r}")


def allocation_count(n: int, ratio: float) -> int:
    if n <= 0:
        return 0
    count = int(round(float(n) * float(ratio)))
    if n >= 2:
        count = max(1, min(n - 1, count))
    return max(0, min(n, count))


def build_split(rows: list[dict[str, Any]], *, seed: int, calibration_ratio: float) -> dict[str, Any]:
    rng = random.Random(seed)
    grouped: dict[str, dict[int, list[dict[str, Any]]]] = defaultdict(lambda: {0: [], 1: []})
    for row in rows:
        grouped[str(row["query_id"])][label_value(row)].append(row)

    calibration_ids: list[str] = []
    pool_ids: list[str] = []
    query_rows: list[dict[str, Any]] = []
    for query_id in sorted(grouped):
        query_calibration: list[str] = []
        query_pool: list[str] = []
        for label in (0, 1):
            items = list(grouped[query_id][label])
            items.sort(key=lambda row: str(row["id"]))
            rng.shuffle(items)
            n_cal = allocation_count(len(items), calibration_ratio)
            cal_items = sorted(items[:n_cal], key=lambda row: str(row["id"]))
            pool_items = sorted(items[n_cal:], key=lambda row: str(row["id"]))
            query_calibration.extend(str(row["id"]) for row in cal_items)
            query_pool.extend(str(row["id"]) for row in pool_items)
        query_calibration.sort()
        query_pool.sort()
        calibration_ids.extend(query_calibration)
        pool_ids.extend(query_pool)
        total = len(query_calibration) + len(query_pool)
        positives = len(grouped[query_id][1])
        negatives = len(grouped[query_id][0])
        query_rows.append(
            {
                "query_id": query_id,
                "rows": total,
                "positive": positives,
                "negative": negatives,
                "positive_pct": round(100.0 * positives / total, 2) if total else 0.0,
                "calibration_rows": len(query_calibration),
                "pool_rows": len(query_pool),
            }
        )

    calibration_ids.sort()
    pool_ids.sort()
    all_ids = calibration_ids + pool_ids
    if len(all_ids) != len(set(all_ids)):
        raise ValueError("split contains duplicate row ids")
    if len(all_ids) != len(rows):
        raise ValueError(f"split covers {len(all_ids)} rows, expected {len(rows)}")

    return {
        "calibration_ids": calibration_ids,
        "final_calibration_ids": [],
        "pool_ids": pool_ids,
        "seed": int(seed),
        "calibration_ratio": float(calibration_ratio),
        "split_algorithm": "query_label_stratified_pooled_calibration_v1",
        "split_scope": "pooled_global_crc_threshold",
        "query_summary": query_rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_path", default="experiments/inputs/lrobench/data.jsonl")
    parser.add_argument("--output_dir", default="experiments/runs/lrobench_ordered/step2_split")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--calibration_ratio", type=float, default=0.2)
    args = parser.parse_args()

    rows = read_jsonl(Path(args.data_path))
    payload = build_split(rows, seed=args.seed, calibration_ratio=args.calibration_ratio)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "cgsd_split_ids.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with (output_dir / "split_by_query.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "query_id",
                "rows",
                "positive",
                "negative",
                "positive_pct",
                "calibration_rows",
                "pool_rows",
            ],
        )
        writer.writeheader()
        writer.writerows(payload["query_summary"])
    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "calibration_rows": len(payload["calibration_ids"]),
                "pool_rows": len(payload["pool_ids"]),
                "queries": len(payload["query_summary"]),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
