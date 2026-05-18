#!/usr/bin/env python
"""Generate FEVER fixed training sets from round0 CGSD outputs."""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from algorithms.cgsd import k_center_greedy, teacher_weight
from scripts.cgsd_cli_common import binary_to_int, read_jsonl
from scripts.run_cgsd import load_embeddings
from src.utils import read_json, write_json, write_jsonl


@dataclass(frozen=True)
class FixedSetBudgets:
    d_guide: int
    d_cert: int
    accept_random: int
    defer_random: int
    defer_kcenter: int


def fixed_set_budgets(*, dataset_size: int) -> dict[str, int]:
    """Return the requested FEVER fixed-set budgets for the given dataset size."""
    total = int(dataset_size)
    return {
        "D_guide": 1000,
        "D_cert": 1000,
        "accept_random": 1000,
        "defer_random": int(round(total * 0.025)),
        "defer_kcenter": int(round(total * 0.025)),
    }


def _row_id(row: Mapping[str, Any]) -> str:
    sample_id = row.get("id", row.get("sample_id"))
    if sample_id is None or str(sample_id) == "":
        raise ValueError(f"row is missing id/sample_id: {row!r}")
    return str(sample_id)


def _choose_random(ids: list[str], *, k: int, seed: int) -> list[str]:
    if int(k) > len(ids):
        raise ValueError(f"requested {k} ids from only {len(ids)} candidates")
    rows = list(ids)
    random.Random(int(seed)).shuffle(rows)
    return rows[: int(k)]


def approximate_k_center_greedy(
    candidate_ids: list[str],
    embeddings_by_id: Mapping[str, Any],
    *,
    k: int,
    seed: int,
    candidate_count: int,
    projection_dim: int,
) -> list[str]:
    """Run deterministic k-center on a bounded, projected defer candidate set."""
    requested = int(max(0, k))
    if requested == 0 or not candidate_ids:
        return []
    subset_size = max(int(candidate_count), requested)
    if len(candidate_ids) <= subset_size:
        candidate_subset = list(candidate_ids)
    else:
        candidate_subset = _choose_random(candidate_ids, k=subset_size, seed=int(seed))
    if int(projection_dim) <= 0:
        return k_center_greedy(candidate_subset, embeddings_by_id, k=requested, seed=int(seed))

    missing_ids = [sample_id for sample_id in candidate_subset if sample_id not in embeddings_by_id]
    if missing_ids:
        raise ValueError(f"missing embeddings for defer k-center ids: {missing_ids[:5]}")
    matrix = np.vstack([np.asarray(embeddings_by_id[sample_id], dtype=np.float32) for sample_id in candidate_subset])
    rng = np.random.default_rng(int(seed))
    projection = rng.standard_normal((matrix.shape[1], int(projection_dim)), dtype=np.float32)
    matrix = matrix @ projection
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms <= 1e-12] = 1.0
    matrix = (matrix / norms).astype(np.float32)

    if len(candidate_subset) <= requested:
        return candidate_subset[:requested]
    center = np.mean(matrix, axis=0, keepdims=True)
    first_index = int(np.argmax(np.sum((matrix - center) ** 2, axis=1)))
    selected_indices = [first_index]
    min_distances = np.sum((matrix - matrix[first_index]) ** 2, axis=1)
    min_distances[first_index] = -1.0
    while len(selected_indices) < requested:
        next_index = int(np.argmax(min_distances))
        selected_indices.append(next_index)
        candidate_distances = np.sum((matrix - matrix[next_index]) ** 2, axis=1)
        min_distances = np.minimum(min_distances, candidate_distances)
        min_distances[selected_indices] = -1.0
    return [candidate_subset[index] for index in selected_indices]


def select_fixed_set_ids(
    *,
    split_payload: Mapping[str, Any],
    pool_crc_rows: list[dict[str, Any]],
    budgets: FixedSetBudgets,
    seed: int,
    embeddings_by_id: Mapping[str, Any] | None,
    all_ids: list[str] | None = None,
    kcenter_candidate_count: int = 20_000,
    kcenter_projection_dim: int = 64,
) -> dict[str, list[str]]:
    """Select guide/cert ids plus two 20k train mixtures.

    Both train mixtures share the same accept-random prefix. The suffix differs:
    random defer sampling for ``defer_random_train`` and k-center defer sampling
    for ``defer_kcenter_train``.
    """
    d_guide = [str(sample_id) for sample_id in split_payload.get("calibration_ids", [])][: int(budgets.d_guide)]
    d_cert = [str(sample_id) for sample_id in split_payload.get("final_calibration_ids", [])][: int(budgets.d_cert)]
    if len(d_guide) < int(budgets.d_guide) or len(d_cert) < int(budgets.d_cert):
        if all_ids is None:
            raise ValueError("all_ids is required when existing split ids do not cover requested D_guide/D_cert")
        available = [str(sample_id) for sample_id in all_ids]
        existing = set(d_guide) | set(d_cert)
        shuffled = [sample_id for sample_id in _choose_random(available, k=len(available), seed=int(seed) + 10) if sample_id not in existing]
        while len(d_guide) < int(budgets.d_guide) and shuffled:
            d_guide.append(shuffled.pop())
        blocked_for_cert = set(d_guide) | set(d_cert)
        shuffled = [sample_id for sample_id in shuffled if sample_id not in blocked_for_cert]
        while len(d_cert) < int(budgets.d_cert) and shuffled:
            d_cert.append(shuffled.pop())
    if len(d_guide) < int(budgets.d_guide):
        raise ValueError(f"D_guide needs {budgets.d_guide} ids, found {len(d_guide)}")
    if len(d_cert) < int(budgets.d_cert):
        raise ValueError(f"D_cert needs {budgets.d_cert} ids, found {len(d_cert)}")

    holdout_ids = set(d_guide) | set(d_cert)
    accept_ids = [_row_id(row) for row in pool_crc_rows if not bool(row.get("defer", False)) and _row_id(row) not in holdout_ids]
    defer_ids = [_row_id(row) for row in pool_crc_rows if bool(row.get("defer", False)) and _row_id(row) not in holdout_ids]
    accept_ids.sort()
    defer_ids.sort()

    accept_random = _choose_random(accept_ids, k=budgets.accept_random, seed=int(seed))
    defer_random = _choose_random(defer_ids, k=budgets.defer_random, seed=int(seed) + 1)
    if embeddings_by_id is None:
        defer_kcenter = defer_ids[: int(budgets.defer_kcenter)]
    else:
        defer_kcenter = approximate_k_center_greedy(
            defer_ids,
            embeddings_by_id,
            k=int(budgets.defer_kcenter),
            seed=int(seed) + 2,
            candidate_count=int(kcenter_candidate_count),
            projection_dim=int(kcenter_projection_dim),
        )

    return {
        "D_guide": d_guide,
        "D_cert": d_cert,
        "accept_random": accept_random,
        "defer_random": defer_random,
        "defer_kcenter": defer_kcenter,
        "defer_random_train": accept_random + defer_random,
        "defer_kcenter_train": accept_random + defer_kcenter,
    }


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


def write_fixed_sets(
    *,
    output_dir: Path,
    selected: Mapping[str, list[str]],
    rows_by_id: Mapping[str, dict[str, Any]],
    teacher_beta: float,
) -> dict[str, int]:
    output_dir.mkdir(parents=True, exist_ok=True)
    names = {
        "D_guide": "D_guide_1000",
        "D_cert": "D_cert_1000",
        "accept_random": "accept_random_1000",
        "defer_random": "defer_random_4000",
        "defer_kcenter": "defer_kcenter_4000",
        "defer_random_train": "defer_random_5000",
        "defer_kcenter_train": "defer_kcenter_5000",
    }
    counts: dict[str, int] = {}
    for key, name in names.items():
        ids = list(selected[key])
        _write_id_set(output_dir, name, ids)
        counts[name] = len(ids)
        if key in {"D_guide", "D_cert"}:
            write_jsonl([dict(rows_by_id[sample_id]) for sample_id in ids], output_dir / f"{name}.jsonl")
        elif key in {"defer_random_train", "defer_kcenter_train"}:
            role = "defer_random_mixed_accept_defer" if key == "defer_random_train" else "defer_kcenter_mixed_accept_defer"
            write_jsonl(
                _materialize_train_rows(ids, rows_by_id=rows_by_id, selection_role=role, teacher_beta=teacher_beta),
                output_dir / f"{name}.train_rows.jsonl",
            )
            write_jsonl([dict(rows_by_id[sample_id]) for sample_id in ids], output_dir / f"{name}.jsonl")
        else:
            write_jsonl([dict(rows_by_id[sample_id]) for sample_id in ids], output_dir / f"{name}.jsonl")
    return counts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split_ids_path", default="experiments/inputs/fever/cgsd_split_ids.json")
    parser.add_argument("--pool_crc_predictions_path", default="experiments/inputs/fever/round_0/pool_crc_predictions.jsonl")
    parser.add_argument("--pool_student_predictions_path", default="experiments/inputs/fever/round_0/pool_student_predictions.jsonl")
    parser.add_argument("--calibration_predictions_path", default="experiments/inputs/fever/round_0/calibration_student_predictions.jsonl")
    parser.add_argument("--final_calibration_predictions_path", default="experiments/inputs/fever/round_0/final_calibration_student_predictions.jsonl")
    parser.add_argument("--embeddings_path", default="experiments/inputs/fever/embeddings.npy")
    parser.add_argument("--output_dir", default="experiments/inputs/fever/fixed_sets_alpha010_t15_seed1_mixed5k")
    parser.add_argument("--dataset_size", type=int, default=160_000)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--teacher_beta", type=float, default=1.0)
    parser.add_argument("--no_kcenter_embeddings", action="store_true")
    parser.add_argument("--kcenter_candidate_count", type=int, default=20_000)
    parser.add_argument("--kcenter_projection_dim", type=int, default=64)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    started = time.time()
    split_payload = read_json(args.split_ids_path)
    pool_crc_rows = read_jsonl(args.pool_crc_predictions_path)
    pool_student_rows = read_jsonl(args.pool_student_predictions_path)
    calibration_rows = read_jsonl(args.calibration_predictions_path)
    final_calibration_rows = read_jsonl(args.final_calibration_predictions_path)
    rows_by_id = {
        _row_id(row): dict(row)
        for row in [*pool_student_rows, *pool_crc_rows, *calibration_rows, *final_calibration_rows]
    }
    budget_payload = fixed_set_budgets(dataset_size=int(args.dataset_size))
    budgets = FixedSetBudgets(
        d_guide=budget_payload["D_guide"],
        d_cert=budget_payload["D_cert"],
        accept_random=budget_payload["accept_random"],
        defer_random=budget_payload["defer_random"],
        defer_kcenter=budget_payload["defer_kcenter"],
    )
    embeddings_by_id = None if args.no_kcenter_embeddings else load_embeddings(Path(args.embeddings_path))
    selected = select_fixed_set_ids(
        split_payload=split_payload,
        pool_crc_rows=pool_crc_rows,
        budgets=budgets,
        seed=int(args.seed),
        embeddings_by_id=embeddings_by_id,
        all_ids=sorted(rows_by_id),
        kcenter_candidate_count=int(args.kcenter_candidate_count),
        kcenter_projection_dim=int(args.kcenter_projection_dim),
    )
    output_dir = Path(args.output_dir)
    counts = write_fixed_sets(
        output_dir=output_dir,
        selected=selected,
        rows_by_id=rows_by_id,
        teacher_beta=float(args.teacher_beta),
    )
    summary = {
        "stage_name": "cgsd_make_fever_fixed_sets",
        "dataset_size": int(args.dataset_size),
        "seed": int(args.seed),
        "budgets": budget_payload,
        "sets": counts,
        "mixtures": {
            "defer_kcenter_5000": {"accept_random": len(selected["accept_random"]), "defer_kcenter": len(selected["defer_kcenter"])},
            "defer_random_5000": {"accept_random": len(selected["accept_random"]), "defer_random": len(selected["defer_random"])},
        },
        "kcenter": {
            "method": "random_defer_candidate_prefilter_then_projected_k_center",
            "candidate_count": int(args.kcenter_candidate_count),
            "projection_dim": int(args.kcenter_projection_dim),
        },
        "source_paths": {
            "split_ids_path": str(args.split_ids_path),
            "pool_crc_predictions_path": str(args.pool_crc_predictions_path),
            "pool_student_predictions_path": str(args.pool_student_predictions_path),
            "calibration_predictions_path": str(args.calibration_predictions_path),
            "final_calibration_predictions_path": str(args.final_calibration_predictions_path),
            "embeddings_path": None if args.no_kcenter_embeddings else str(args.embeddings_path),
        },
        "elapsed_seconds": round(time.time() - started, 2),
    }
    write_json(summary, output_dir / "fixed_selection_summary.json")
    print(json.dumps({"output_dir": str(output_dir), "sets": counts}, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
