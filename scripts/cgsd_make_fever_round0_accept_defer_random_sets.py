#!/usr/bin/env python
"""Generate FEVER round0 accept/defer stratified random training sets."""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import Any, Mapping

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


def _choose_random(ids: list[str], *, k: int, seed: int) -> list[str]:
    if int(k) > len(ids):
        raise ValueError(f"requested {k} ids from only {len(ids)} candidates")
    shuffled = list(ids)
    random.Random(int(seed)).shuffle(shuffled)
    return shuffled[: int(k)]


def pool_crc_defer_rate(rows: list[dict[str, Any]]) -> float:
    """Return the CRC-identified defer rate in a pool prediction file."""
    if not rows:
        raise ValueError("pool CRC predictions cannot be empty")
    return sum(1 for row in rows if bool(row.get("defer", False))) / len(rows)


def accept_fraction_from_crc_defer_rate(defer_rate: float) -> float:
    """Compute pi_accept(r) where pi_defer(r) = r + (1-r)^2."""
    r = float(defer_rate)
    if not 0.0 <= r <= 1.0:
        raise ValueError("crc defer rate must be in [0, 1]")
    pi_defer = r + (1.0 - r) ** 2
    return max(0.0, min(1.0, 1.0 - pi_defer))


def format_accept_defer_name(accept_fraction: float) -> str:
    accept_pct = int(round(float(accept_fraction) * 100))
    defer_pct = 100 - accept_pct
    return f"accept{accept_pct}_defer{defer_pct}"


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
        label = binary_to_int(
            payload.get("teacher_label", payload.get("label", payload.get("groundtruth"))),
            field_name=f"fixed set row {sample_id!r} label",
        )
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


def _write_id_set(output_dir: Path, name: str, ids: list[str]) -> None:
    write_json({"name": name, "count": len(ids), "ids": ids}, output_dir / f"{name}.ids.json")


def _load_blocked_ids(paths: list[str]) -> set[str]:
    blocked: set[str] = set()
    for path_text in paths:
        if not path_text:
            continue
        path = Path(path_text)
        if not path.exists():
            raise FileNotFoundError(path)
        for row in read_jsonl(path):
            blocked.add(_row_id(row))
    return blocked


def parse_sizes(text: str) -> list[int]:
    sizes = [int(part.strip()) for part in str(text or "").split(",") if part.strip()]
    if not sizes:
        raise ValueError("--sizes cannot be empty")
    if any(size <= 0 for size in sizes):
        raise ValueError("--sizes must all be positive")
    return sizes


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool_crc_predictions_path", default="experiments/inputs/fever/round_0/pool_crc_predictions.jsonl")
    parser.add_argument("--calibration_predictions_path", default="experiments/inputs/fever/round_0/calibration_student_predictions.jsonl")
    parser.add_argument("--final_calibration_predictions_path", default="experiments/inputs/fever/round_0/final_calibration_student_predictions.jsonl")
    parser.add_argument("--output_dir", default="experiments/inputs/fever/fixed_sets_alpha010_t15_seed1_round0_accept15_defer85_random")
    parser.add_argument("--sizes", default="500,1000,2500,5000,20000")
    parser.add_argument("--accept_fraction_mode", choices=("fixed", "crc_formula"), default="fixed")
    parser.add_argument("--accept_fraction", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--teacher_beta", type=float, default=1.0)
    parser.add_argument("--no_block_calibration", action="store_true")
    parser.add_argument("--write_component_sets", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    started = time.time()
    sizes = parse_sizes(args.sizes)

    pool_rows = read_jsonl(args.pool_crc_predictions_path)
    observed_defer_rate = pool_crc_defer_rate(pool_rows)
    if args.accept_fraction_mode == "crc_formula":
        accept_fraction = accept_fraction_from_crc_defer_rate(observed_defer_rate)
    else:
        accept_fraction = float(args.accept_fraction)
    if not 0.0 <= accept_fraction <= 1.0:
        raise ValueError("--accept_fraction must be in [0, 1]")
    ratio_name = format_accept_defer_name(accept_fraction)

    rows_by_id = {_row_id(row): dict(row) for row in pool_rows}
    blocked_ids: set[str] = set()
    if not bool(args.no_block_calibration):
        blocked_ids = _load_blocked_ids([args.calibration_predictions_path, args.final_calibration_predictions_path])

    accept_ids = sorted(_row_id(row) for row in pool_rows if not bool(row.get("defer", False)) and _row_id(row) not in blocked_ids)
    defer_ids = sorted(_row_id(row) for row in pool_rows if bool(row.get("defer", False)) and _row_id(row) not in blocked_ids)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    set_counts: dict[str, int] = {}
    mixtures: dict[str, dict[str, int]] = {}

    for size in sizes:
        accept_count = int(round(int(size) * accept_fraction))
        defer_count = int(size) - accept_count
        accept_selected = _choose_random(accept_ids, k=accept_count, seed=int(args.seed) + int(size) * 2)
        defer_selected = _choose_random(defer_ids, k=defer_count, seed=int(args.seed) + int(size) * 2 + 1)
        merged_ids = accept_selected + defer_selected

        accept_name = f"round0_accept_random_{accept_count}_of_{size}_{ratio_name}_seed{args.seed}"
        defer_name = f"round0_defer_random_{defer_count}_of_{size}_{ratio_name}_seed{args.seed}"
        merged_name = f"round0_{ratio_name}_random_{size}_seed{args.seed}"

        if bool(args.write_component_sets):
            for name, ids in ((accept_name, accept_selected), (defer_name, defer_selected)):
                _write_id_set(output_dir, name, ids)
                write_jsonl([dict(rows_by_id[sample_id]) for sample_id in ids], output_dir / f"{name}.jsonl")
                set_counts[name] = len(ids)

        _write_id_set(output_dir, merged_name, merged_ids)
        write_jsonl([dict(rows_by_id[sample_id]) for sample_id in merged_ids], output_dir / f"{merged_name}.jsonl")
        set_counts[merged_name] = len(merged_ids)

        train_rows = _materialize_train_rows(
            merged_ids,
            rows_by_id=rows_by_id,
            selection_role=f"round0_{ratio_name}_random",
            teacher_beta=float(args.teacher_beta),
        )
        write_jsonl(train_rows, output_dir / f"{merged_name}.train_rows.jsonl")
        mixtures[merged_name] = {
            "total": len(merged_ids),
            "accept_random": len(accept_selected),
            "defer_random": len(defer_selected),
        }

    summary = {
        "stage_name": "cgsd_make_fever_round0_accept_defer_random_sets",
        "seed": int(args.seed),
        "accept_fraction_mode": str(args.accept_fraction_mode),
        "accept_defer_ratio_name": ratio_name,
        "accept_fraction": accept_fraction,
        "defer_fraction": 1.0 - accept_fraction,
        "observed_pool_crc_defer_rate": observed_defer_rate,
        "crc_formula": {
            "pi_defer": "r + (1-r)^2",
            "pi_accept": "1 - pi_defer",
            "r_source": "pool_crc_predictions defer field",
        },
        "sizes": sizes,
        "candidate_counts": {
            "accept": len(accept_ids),
            "defer": len(defer_ids),
            "blocked": len(blocked_ids),
        },
        "sets": set_counts,
        "mixtures": mixtures,
        "source_paths": {
            "pool_crc_predictions_path": str(args.pool_crc_predictions_path),
            "calibration_predictions_path": None if args.no_block_calibration else str(args.calibration_predictions_path),
            "final_calibration_predictions_path": None if args.no_block_calibration else str(args.final_calibration_predictions_path),
        },
        "write_component_sets": bool(args.write_component_sets),
        "elapsed_seconds": round(time.time() - started, 2),
    }
    write_json(summary, output_dir / "fixed_selection_summary.json")
    print(json.dumps({"output_dir": str(output_dir), "mixtures": mixtures}, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
