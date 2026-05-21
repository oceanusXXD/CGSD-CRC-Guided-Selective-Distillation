#!/usr/bin/env python
"""Generate the documented FEVER round0 500-sample training sets."""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from algorithms.cgsd import (  # noqa: E402
    apply_crc_decisions,
    calibrate_crc,
    compute_adaptive_sampling_plan,
    select_documented_training_samples,
    summarize_crc_decisions,
    teacher_weight,
)
from scripts.cgsd_calibrate import compute_crc_sampling_statistics  # noqa: E402
from scripts.cgsd_cli_common import binary_to_int, read_jsonl  # noqa: E402
from scripts.run_cgsd import assert_embedding_coverage, load_embeddings  # noqa: E402
from src.utils import write_json, write_jsonl  # noqa: E402


METHODS = ("pool-random", "pure-accept", "pure-defer", "fixed-15-85", "crc-error-mass")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--all_predictions_path",
        default="experiments/inputs/fever/round_0/all_student_predictions.jsonl",
    )
    parser.add_argument(
        "--output_dir",
        default="experiments/inputs/fever/documented_sampling_500_seeds1_2_3_t15_alpha010",
    )
    parser.add_argument("--test_size", type=int, default=10_000)
    parser.add_argument("--guide_size", type=int, default=1000)
    parser.add_argument("--train_size", type=int, default=500)
    parser.add_argument("--seeds", default="1,2,3")
    parser.add_argument(
        "--methods",
        default=",".join(METHODS),
        help=(
            "Comma-separated methods to generate. Accepts hyphen or underscore names: "
            f"{', '.join(METHODS)}."
        ),
    )
    parser.add_argument("--temperature", type=float, default=15.0)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--embeddings_path", default="experiments/inputs/fever/embeddings.npy")
    parser.add_argument("--embedding_dim", type=int, default=0)
    parser.add_argument("--teacher_beta", type=float, default=1.0)
    parser.add_argument(
        "--write_pool_artifacts",
        action="store_true",
        help="Also write per-seed D_test and pool prediction JSONL artifacts. Disabled by default to keep outputs small.",
    )
    return parser.parse_args()


def _row_id(row: Mapping[str, Any]) -> str:
    sample_id = row.get("id", row.get("sample_id"))
    if sample_id is None or str(sample_id) == "":
        raise ValueError(f"row is missing id/sample_id: {row!r}")
    return str(sample_id)


def _label(row: Mapping[str, Any]) -> int:
    return binary_to_int(row.get("groundtruth", row.get("label")), field_name=f"FEVER row {_row_id(row)!r} label")


def make_split_ids(
    all_ids: Sequence[str],
    *,
    test_size: int,
    guide_size: int,
    seed: int,
) -> dict[str, list[str]]:
    """Create FEVER \\ (D_test union D_guide) with deterministic random IDs."""
    ids = sorted(str(sample_id) for sample_id in all_ids)
    if len(ids) != len(set(ids)):
        raise ValueError("all_predictions_path contains duplicate ids")
    if int(test_size) <= 0 or int(guide_size) <= 0:
        raise ValueError("test_size and guide_size must be positive")
    if int(test_size) + int(guide_size) >= len(ids):
        raise ValueError("test_size + guide_size must be smaller than the dataset size")
    random.Random(int(seed)).shuffle(ids)
    test_ids = ids[: int(test_size)]
    guide_ids = ids[int(test_size) : int(test_size) + int(guide_size)]
    pool_ids = ids[int(test_size) + int(guide_size) :]
    return {
        "test_ids": test_ids,
        "guide_ids": guide_ids,
        "calibration_ids": guide_ids,
        "pool_ids": pool_ids,
    }


def parse_int_csv(text: str) -> list[int]:
    values = [int(part.strip()) for part in str(text or "").split(",") if part.strip()]
    if not values:
        raise ValueError("--seeds cannot be empty")
    if len(values) != len(set(values)):
        raise ValueError("--seeds contains duplicate values")
    return values


def parse_methods_csv(text: str) -> list[str]:
    methods = [part.strip().replace("_", "-") for part in str(text or "").split(",") if part.strip()]
    if not methods:
        raise ValueError("--methods cannot be empty")
    invalid = [method for method in methods if method not in METHODS]
    if invalid:
        raise ValueError(f"--methods contains unsupported values: {invalid}; supported values: {list(METHODS)}")
    if len(methods) != len(set(methods)):
        raise ValueError("--methods contains duplicate values")
    return methods


def _subset_rows(ids: Sequence[str], rows_by_id: Mapping[str, dict[str, Any]]) -> list[dict[str, Any]]:
    return [dict(rows_by_id[str(sample_id)]) for sample_id in ids]


def _dataset_file_stem(method: str, *, train_size: int, seed: int) -> str:
    return f"{method.replace('-', '_')}_{int(train_size)}_seed{int(seed)}"


def materialize_train_rows(
    ids: Sequence[str],
    *,
    rows_by_id: Mapping[str, dict[str, Any]],
    method: str,
    teacher_beta: float,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for sample_id in ids:
        payload = dict(rows_by_id[str(sample_id)])
        label = _label(payload)
        payload["label"] = label
        payload["groundtruth"] = label
        payload["teacher_label"] = label
        payload["teacher_confidence"] = 1.0
        payload["teacher_source"] = "fever_groundtruth"
        payload["teacher_label_source"] = "fever_groundtruth"
        payload["teacher_confidence_source"] = "fixed_1.0_fever_groundtruth"
        payload["sample_weight"] = teacher_weight(1.0, float(teacher_beta))
        payload["selection_round"] = 0
        payload["selection_role"] = str(method).replace("-", "_")
        rows.append(payload)
    return rows


def label_counts(rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    counts = {"0": 0, "1": 0}
    for row in rows:
        counts[str(_label(row))] += 1
    return counts


def write_dataset(
    *,
    output_dir: Path,
    method: str,
    train_size: int,
    seed: int,
    selected_ids: Sequence[str],
    rows_by_id: Mapping[str, dict[str, Any]],
    selection_payload: dict[str, Any],
    teacher_beta: float,
    source_summary: dict[str, Any],
) -> dict[str, Any]:
    stem = _dataset_file_stem(method, train_size=train_size, seed=seed)
    raw_rows = _subset_rows(selected_ids, rows_by_id)
    train_rows = materialize_train_rows(
        selected_ids,
        rows_by_id=rows_by_id,
        method=method,
        teacher_beta=teacher_beta,
    )
    id_payload = {"name": stem, "method": method, "count": len(selected_ids), "ids": list(selected_ids)}
    summary = {
        "name": stem,
        "method": method,
        "requested_train_size": int(train_size),
        "selected_count": len(selected_ids),
        "shortfall": bool(selection_payload.get("shortfall", len(selected_ids) < int(train_size))),
        "accept_selected_count": int(selection_payload.get("selected_accept_budget", 0)),
        "defer_selected_count": int(selection_payload.get("selected_defer_budget", 0)),
        "label_counts": label_counts(train_rows),
        "ids_path": str(output_dir / f"{stem}.ids.json"),
        "jsonl_path": str(output_dir / f"{stem}.jsonl"),
        "train_rows_path": str(output_dir / f"{stem}.train_rows.jsonl"),
        "selection": selection_payload,
        "source": source_summary,
    }
    write_json(id_payload, output_dir / f"{stem}.ids.json")
    write_jsonl(raw_rows, output_dir / f"{stem}.jsonl")
    write_jsonl(train_rows, output_dir / f"{stem}.train_rows.jsonl")
    write_json(summary, output_dir / f"{stem}.summary.json")
    return summary


def generate_for_seed(
    *,
    args: argparse.Namespace,
    seed: int,
    methods: Sequence[str],
    all_rows: Sequence[dict[str, Any]],
    rows_by_id: Mapping[str, dict[str, Any]],
    output_root: Path,
    embeddings_by_id: Mapping[str, Any],
) -> dict[str, Any]:
    seed_output_dir = output_root / f"seed{int(seed)}"
    round0_dir = seed_output_dir / "round_0"
    split = make_split_ids(
        sorted(rows_by_id),
        test_size=int(args.test_size),
        guide_size=int(args.guide_size),
        seed=int(seed),
    )
    guide_rows = _subset_rows(split["guide_ids"], rows_by_id)
    pool_rows = _subset_rows(split["pool_ids"], rows_by_id)

    crc = calibrate_crc(
        guide_rows,
        alpha=float(args.alpha),
        temperature=float(args.temperature),
        embeddings_by_id=embeddings_by_id,
    )
    guide_decisions = apply_crc_decisions(
        guide_rows,
        lambda_hat=crc.lambda_hat,
        temperature=float(args.temperature),
        embeddings_by_id=embeddings_by_id,
        support_rows=guide_rows,
        crc_result=crc,
        neighbor_exclude_self=True,
    )
    pool_decisions = apply_crc_decisions(
        pool_rows,
        lambda_hat=crc.lambda_hat,
        temperature=float(args.temperature),
        embeddings_by_id=embeddings_by_id,
        support_rows=guide_rows,
        crc_result=crc,
    )
    pool_decisions_by_id = {_row_id(row): dict(row) for row in pool_decisions}
    sampling_stats = compute_crc_sampling_statistics(
        guide_decisions,
        pool_decisions,
        temperature=float(args.temperature),
        lambda_hat=float(crc.lambda_hat),
    )
    pool_summary = summarize_crc_decisions(pool_decisions)
    guide_summary = summarize_crc_decisions(guide_decisions)
    round_summary = {
        "round_index": 0,
        "student": "Qwen3-0.6B round0 zero-shot predictions",
        "temperature": float(args.temperature),
        "T": float(args.temperature),
        "alpha": float(args.alpha),
        "lambda_hat": float(crc.lambda_hat),
        "crc": crc.to_dict(),
        "pool_definition": "FEVER \\ (D_test union D_guide)",
        "split_counts": {
            "FEVER": len(all_rows),
            "D_test": len(split["test_ids"]),
            "D_guide": len(split["guide_ids"]),
            "U_pool": len(split["pool_ids"]),
        },
        "pool_summary": pool_summary,
        "guide_summary": guide_summary,
        "sampling_statistics": sampling_stats,
        "source_predictions_path": str(args.all_predictions_path),
        "embeddings_path": str(args.embeddings_path),
    }
    write_json(
        {
            **split,
            "test_size": int(args.test_size),
            "guide_size": int(args.guide_size),
            "pool_size": len(split["pool_ids"]),
            "seed": int(seed),
            "split_algorithm": f"sorted_ids_random_shuffle_test10000_guide{int(args.guide_size)}_pool_rest_v1",
        },
        seed_output_dir / "split_ids.json",
    )
    write_json(
        {"name": f"D_test_10000_seed{int(seed)}", "count": len(split["test_ids"]), "ids": split["test_ids"]},
        seed_output_dir / "D_test_10000.ids.json",
    )
    write_json(
        {
            "name": f"D_guide_{int(args.guide_size)}_seed{int(seed)}",
            "count": len(split["guide_ids"]),
            "ids": split["guide_ids"],
        },
        seed_output_dir / f"D_guide_{int(args.guide_size)}.ids.json",
    )
    if bool(args.write_pool_artifacts):
        write_jsonl(_subset_rows(split["test_ids"], rows_by_id), seed_output_dir / "D_test_10000.jsonl")
        write_jsonl(pool_rows, round0_dir / "pool_student_predictions.jsonl")
        write_jsonl(pool_decisions, round0_dir / "pool_crc_predictions.jsonl")
    write_jsonl(guide_decisions, round0_dir / "guide_crc_predictions.jsonl")
    write_json(round_summary, round0_dir / "round_summary.json")

    summaries: dict[str, Any] = {}
    source_summary = {
        "split_ids_path": str(seed_output_dir / "split_ids.json"),
        "round_summary_path": str(round0_dir / "round_summary.json"),
        "pool_crc_predictions_path": str(round0_dir / "pool_crc_predictions.jsonl")
        if bool(args.write_pool_artifacts)
        else None,
        "source_predictions_path": str(args.all_predictions_path),
        "labels": "FEVER groundtruth",
    }
    adaptive_plan = compute_adaptive_sampling_plan(
        guide_decisions,
        pool_decisions,
        budget=int(args.train_size),
        temperature=float(args.temperature),
        lambda_hat=float(crc.lambda_hat),
        alpha=float(args.alpha),
    )
    for method in methods:
        plan = adaptive_plan if method == "crc-error-mass" else None
        selection = select_documented_training_samples(
            pool_decisions,
            method=method,
            budget=int(args.train_size),
            seed=int(seed),
            blocked_ids=(),
            sampling_plan=plan,
            accept_strategy="random",
            defer_strategy="random",
            embeddings_by_id=embeddings_by_id,
        )
        summaries[method] = write_dataset(
            output_dir=seed_output_dir,
            method=method,
            train_size=int(args.train_size),
            seed=int(seed),
            selected_ids=selection.distillation_ids,
            rows_by_id=pool_decisions_by_id,
            selection_payload=selection.to_dict(),
            teacher_beta=float(args.teacher_beta),
            source_summary=source_summary,
        )

    seed_summary = {
        "seed": int(seed),
        "datasets_count": len(summaries),
        "temperature": float(args.temperature),
        "alpha": float(args.alpha),
        "split_counts": round_summary["split_counts"],
        "pool_summary": pool_summary,
        "guide_summary": guide_summary,
        "sampling_statistics": sampling_stats,
        "crc_error_mass_execution_counts": {
            "B_accept": int(adaptive_plan.B_accept),
            "B_defer": int(adaptive_plan.B_defer),
        },
        "datasets": summaries,
    }
    write_json(seed_summary, seed_output_dir / "documented_sampling_summary.json")
    return seed_summary


def main() -> None:
    args = parse_args()
    started = time.time()
    output_dir = Path(args.output_dir)
    seeds = parse_int_csv(args.seeds)
    methods = parse_methods_csv(args.methods)

    all_rows = read_jsonl(args.all_predictions_path)
    rows_by_id = {_row_id(row): dict(row) for row in all_rows}
    if len(rows_by_id) != len(all_rows):
        raise ValueError("all_predictions_path contains duplicate ids")
    if not args.embeddings_path:
        raise ValueError("--embeddings_path is required because all CRC uses neighbor-support")
    embeddings_path = Path(args.embeddings_path)
    if not embeddings_path.is_absolute() and not embeddings_path.exists():
        embeddings_path = PROJECT_ROOT / embeddings_path
    embeddings_by_id = load_embeddings(embeddings_path)
    assert_embedding_coverage(
        embeddings_by_id,
        all_rows,
        expected_dim=int(args.embedding_dim),
    )

    seed_summaries: dict[str, Any] = {}
    for seed in seeds:
        seed_summaries[f"seed{int(seed)}"] = generate_for_seed(
            args=args,
            seed=int(seed),
            methods=methods,
            all_rows=all_rows,
            rows_by_id=rows_by_id,
            output_root=output_dir,
            embeddings_by_id=embeddings_by_id,
        )

    total_datasets = sum(int(summary["datasets_count"]) for summary in seed_summaries.values())
    aggregate = {
        "stage_name": "cgsd_make_fever_documented_sampling_sets",
        "elapsed_seconds": round(time.time() - started, 2),
        "methods": list(methods),
        "seeds": seeds,
        "expected_dataset_count": len(methods) * len(seeds),
        "dataset_count": total_datasets,
        "train_size": int(args.train_size),
        "temperature": float(args.temperature),
        "alpha": float(args.alpha),
        "source_predictions_path": str(args.all_predictions_path),
        "embeddings_path": str(embeddings_path),
        "write_pool_artifacts": bool(args.write_pool_artifacts),
        "seed_summaries": seed_summaries,
    }
    write_json(aggregate, output_dir / "documented_sampling_summary.json")
    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "dataset_count": total_datasets,
                "seeds": seeds,
                "datasets_by_seed": {
                    seed_name: {
                        method: {
                            "selected": summary["selected_count"],
                            "accept": summary["accept_selected_count"],
                            "defer": summary["defer_selected_count"],
                            "shortfall": summary["shortfall"],
                        }
                        for method, summary in seed_summary["datasets"].items()
                    }
                    for seed_name, seed_summary in seed_summaries.items()
                },
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
