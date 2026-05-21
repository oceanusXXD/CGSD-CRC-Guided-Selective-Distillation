#!/usr/bin/env python
"""Generate FEVER budget-sweep training sets from cached round0 predictions."""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from algorithms.cgsd import (  # noqa: E402
    apply_crc_decisions,
    calibrate_crc,
    compute_adaptive_sampling_plan,
    score_ns_difficulty_global,
    select_documented_training_samples,
    select_ns_error_mass_samples,
    summarize_crc_decisions,
    teacher_weight,
)
from scripts.cgsd_calibrate import compute_crc_sampling_statistics  # noqa: E402
from scripts.cgsd_cli_common import binary_to_int, read_jsonl  # noqa: E402
from scripts.run_cgsd import assert_embedding_coverage, load_embeddings  # noqa: E402
from src.utils import read_json, write_json, write_jsonl  # noqa: E402


METHODS = ("pool-random", "ns-error-mass")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--split_ids_path", default="experiments/inputs/fever/cgsd_split_ids.json")
    parser.add_argument(
        "--calibration_predictions_path",
        default="experiments/inputs/fever/round_0/calibration_student_predictions.jsonl",
    )
    parser.add_argument(
        "--pool_predictions_path",
        default="experiments/inputs/fever/round_0/pool_student_predictions.jsonl",
    )
    parser.add_argument("--embeddings_path", default="experiments/inputs/fever/embeddings.npy")
    parser.add_argument("--embedding_dim", type=int, default=2560)
    parser.add_argument("--budgets", default="1500,3000,4500,6000,7500,9000")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--teacher_beta", type=float, default=1.0)
    parser.add_argument("--write_pool_crc_predictions", action="store_true", default=False)
    return parser.parse_args()


def parse_budgets(text: str) -> list[int]:
    budgets = [int(part.strip()) for part in str(text or "").split(",") if part.strip()]
    if not budgets:
        raise ValueError("--budgets cannot be empty")
    if len(budgets) != len(set(budgets)):
        raise ValueError("--budgets contains duplicate values")
    if any(budget <= 0 for budget in budgets):
        raise ValueError("--budgets must be positive")
    return budgets


def row_id(row: Mapping[str, Any]) -> str:
    sample_id = row.get("id", row.get("sample_id"))
    if sample_id is None or str(sample_id) == "":
        raise ValueError(f"row is missing id/sample_id: {row!r}")
    return str(sample_id)


def label_value(row: Mapping[str, Any]) -> int:
    return binary_to_int(row.get("groundtruth", row.get("label")), field_name=f"row {row_id(row)!r} label")


def prediction_value(row: Mapping[str, Any]) -> int:
    value = row.get("prediction")
    if value is None:
        value = int(float(row.get("score", 0.0) or 0.0) > 0.0)
    return binary_to_int(value, field_name=f"row {row_id(row)!r} prediction")


def label_counts(rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    counts = {"0": 0, "1": 0}
    for row in rows:
        counts[str(label_value(row))] += 1
    return counts


def base_error_count(rows: Sequence[Mapping[str, Any]]) -> int:
    return sum(1 for row in rows if prediction_value(row) != label_value(row))


def materialize_train_rows(
    ids: Sequence[str],
    *,
    rows_by_id: Mapping[str, dict[str, Any]],
    method: str,
    teacher_beta: float,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for sample_id in ids:
        row = dict(rows_by_id[str(sample_id)])
        label = label_value(row)
        row["label"] = label
        row["groundtruth"] = label
        row["teacher_label"] = label
        row["teacher_confidence"] = 1.0
        row["teacher_source"] = "fever_groundtruth"
        row["teacher_label_source"] = "fever_groundtruth"
        row["teacher_confidence_source"] = "fixed_1.0_fever_groundtruth"
        row["sample_weight"] = teacher_weight(1.0, float(teacher_beta))
        row["selection_round"] = 0
        row["selection_role"] = str(method).replace("-", "_")
        rows.append(row)
    return rows


def dataset_stem(method: str, *, budget: int, seed: int) -> str:
    return f"{method.replace('-', '_')}_{int(budget)}_seed{int(seed)}"


def write_dataset(
    *,
    output_dir: Path,
    method: str,
    budget: int,
    seed: int,
    selected_ids: Sequence[str],
    rows_by_id: Mapping[str, dict[str, Any]],
    selection_payload: dict[str, Any],
    teacher_beta: float,
    source_payload: dict[str, Any],
) -> dict[str, Any]:
    stem = dataset_stem(method, budget=budget, seed=seed)
    raw_rows = [dict(rows_by_id[str(sample_id)]) for sample_id in selected_ids]
    train_rows = materialize_train_rows(
        selected_ids,
        rows_by_id=rows_by_id,
        method=method,
        teacher_beta=teacher_beta,
    )
    error_count = base_error_count(train_rows)
    summary = {
        "name": stem,
        "method": method,
        "seed": int(seed),
        "requested_train_size": int(budget),
        "selected_count": len(selected_ids),
        "shortfall": bool(selection_payload.get("shortfall", len(selected_ids) < int(budget))),
        "accept_selected_count": int(selection_payload.get("selected_accept_budget", 0)),
        "defer_selected_count": int(selection_payload.get("selected_defer_budget", 0)),
        "label_counts": label_counts(train_rows),
        "base_error_count": int(error_count),
        "base_error_rate": float(error_count / len(train_rows)) if train_rows else 0.0,
        "ids_path": str(output_dir / f"{stem}.ids.json"),
        "jsonl_path": str(output_dir / f"{stem}.jsonl"),
        "train_rows_path": str(output_dir / f"{stem}.train_rows.jsonl"),
        "selection": selection_payload,
        "source": source_payload,
    }
    write_json({"name": stem, "method": method, "count": len(selected_ids), "ids": list(selected_ids)}, output_dir / f"{stem}.ids.json")
    write_jsonl(raw_rows, output_dir / f"{stem}.jsonl")
    write_jsonl(train_rows, output_dir / f"{stem}.train_rows.jsonl")
    write_json(summary, output_dir / f"{stem}.summary.json")
    return summary


def select_ns_error_mass_split(
    scored_pool_rows: Sequence[dict[str, Any]],
    *,
    plan: Any,
    budget: int,
    seed: int,
) -> tuple[list[str], dict[str, Any]]:
    accept_rows = [row for row in scored_pool_rows if not bool(row.get("defer", False))]
    defer_rows = [row for row in scored_pool_rows if bool(row.get("defer", False))]
    accept_budget = min(int(plan.B_accept), len(accept_rows))
    defer_budget = min(int(plan.B_defer), len(defer_rows))
    accept_plan = replace(
        plan,
        budget=accept_budget,
        s_accept=1.0,
        s_defer=0.0,
        B_accept=accept_budget,
        B_defer=0,
        pool_accept_count=len(accept_rows),
        pool_defer_count=0,
    )
    defer_plan = replace(
        plan,
        budget=defer_budget,
        s_accept=0.0,
        s_defer=1.0,
        B_accept=0,
        B_defer=defer_budget,
        pool_accept_count=0,
        pool_defer_count=len(defer_rows),
    )
    accept_selection = select_ns_error_mass_samples(
        accept_rows,
        sampling_plan=accept_plan,
        budget=accept_budget,
        seed=int(seed),
    )
    defer_selection = select_ns_error_mass_samples(
        defer_rows,
        sampling_plan=defer_plan,
        budget=defer_budget,
        seed=int(seed) + 1,
    )
    selected_ids = [*accept_selection.distillation_ids, *defer_selection.distillation_ids]
    payload = {
        "distillation_ids": selected_ids,
        "accept_ids": accept_selection.distillation_ids,
        "defer_ids": defer_selection.distillation_ids,
        "requested_budget": int(budget),
        "selected_budget": len(selected_ids),
        "requested_accept_budget": int(plan.B_accept),
        "requested_defer_budget": int(plan.B_defer),
        "selected_accept_budget": len(accept_selection.distillation_ids),
        "selected_defer_budget": len(defer_selection.distillation_ids),
        "selection_method": "ns-error-mass",
        "ns_weighting": "ns-error-mass",
        "accept_candidate_count": len(accept_rows),
        "defer_candidate_count": len(defer_rows),
        "shortfall": len(selected_ids) < int(budget),
        "accept_ns_selection": accept_selection.to_dict(),
        "defer_ns_selection": defer_selection.to_dict(),
    }
    return selected_ids, payload


def main() -> None:
    args = parse_args()
    started = time.time()
    budgets = parse_budgets(args.budgets)
    output_root = Path(args.output_dir)
    seed_dir = output_root / f"seed{int(args.seed)}"
    round0_dir = seed_dir / "round_0"
    seed_dir.mkdir(parents=True, exist_ok=True)
    round0_dir.mkdir(parents=True, exist_ok=True)

    split_payload = read_json(Path(args.split_ids_path))
    calibration_rows = read_jsonl(Path(args.calibration_predictions_path))
    pool_rows = read_jsonl(Path(args.pool_predictions_path))
    embeddings_path = Path(args.embeddings_path)
    if not embeddings_path.is_absolute() and not embeddings_path.exists():
        embeddings_path = PROJECT_ROOT / embeddings_path
    embeddings_by_id = load_embeddings(embeddings_path)
    assert_embedding_coverage(
        embeddings_by_id,
        [*calibration_rows, *pool_rows],
        expected_dim=int(args.embedding_dim),
    )

    crc = calibrate_crc(
        calibration_rows,
        alpha=float(args.alpha),
        temperature=float(args.temperature),
        embeddings_by_id=embeddings_by_id,
    )
    calibration_decisions = apply_crc_decisions(
        calibration_rows,
        lambda_hat=float(crc.lambda_hat),
        temperature=float(args.temperature),
        embeddings_by_id=embeddings_by_id,
        support_rows=calibration_rows,
        crc_result=crc,
        neighbor_exclude_self=True,
    )
    pool_decisions = apply_crc_decisions(
        pool_rows,
        lambda_hat=float(crc.lambda_hat),
        temperature=float(args.temperature),
        embeddings_by_id=embeddings_by_id,
        support_rows=calibration_rows,
        crc_result=crc,
    )
    pool_summary = summarize_crc_decisions(pool_decisions)
    guide_summary = summarize_crc_decisions(calibration_decisions)
    sampling_statistics = compute_crc_sampling_statistics(
        calibration_decisions,
        pool_decisions,
        temperature=float(args.temperature),
        lambda_hat=float(crc.lambda_hat),
    )
    ns_scoring = score_ns_difficulty_global(
        pool_decisions,
        calibration_decisions,
        embeddings_by_id=embeddings_by_id,
        e_all=float(sampling_statistics["e_all"]),
    )
    rows_by_id = {row_id(row): dict(row) for row in ns_scoring.pool_rows}
    round_summary = {
        "round_index": 0,
        "student": "Qwen3-0.6B round0 cached predictions",
        "temperature": float(args.temperature),
        "T": float(args.temperature),
        "alpha": float(args.alpha),
        "lambda_hat": float(crc.lambda_hat),
        "crc": crc.to_dict(),
        "split_counts": {
            "D_guide": len(calibration_rows),
            "D_cert": len(split_payload.get("final_calibration_ids", [])),
            "U_pool": len(pool_rows),
        },
        "pool_summary": pool_summary,
        "guide_summary": guide_summary,
        "sampling_statistics": sampling_statistics,
        "ns_scoring": ns_scoring.to_dict(),
        "source_predictions": {
            "calibration_predictions_path": str(args.calibration_predictions_path),
            "pool_predictions_path": str(args.pool_predictions_path),
        },
        "embeddings_path": str(embeddings_path),
    }
    write_json(split_payload, seed_dir / "split_ids.json")
    write_json(round_summary, round0_dir / "round_summary.json")
    write_jsonl(calibration_decisions, round0_dir / "guide_crc_predictions.jsonl")
    if bool(args.write_pool_crc_predictions):
        write_jsonl(pool_decisions, round0_dir / "pool_crc_predictions.jsonl")

    source_payload = {
        "split_ids_path": str(seed_dir / "split_ids.json"),
        "round_summary_path": str(round0_dir / "round_summary.json"),
        "pool_crc_predictions_path": str(round0_dir / "pool_crc_predictions.jsonl")
        if bool(args.write_pool_crc_predictions)
        else None,
        "calibration_predictions_path": str(args.calibration_predictions_path),
        "pool_predictions_path": str(args.pool_predictions_path),
        "labels": "FEVER groundtruth",
    }
    dataset_summaries: dict[str, dict[str, Any]] = {}
    for budget in budgets:
        plan = compute_adaptive_sampling_plan(
            calibration_decisions,
            pool_decisions,
            budget=int(budget),
            temperature=float(args.temperature),
            lambda_hat=float(crc.lambda_hat),
            alpha=float(args.alpha),
        )
        for method in METHODS:
            if method == "ns-error-mass":
                selected_ids, selection_payload = select_ns_error_mass_split(
                    ns_scoring.pool_rows,
                    plan=plan,
                    budget=int(budget),
                    seed=int(args.seed),
                )
            else:
                selection = select_documented_training_samples(
                    ns_scoring.pool_rows,
                    method=method,
                    budget=int(budget),
                    seed=int(args.seed),
                    blocked_ids=(),
                    sampling_plan=None,
                    accept_strategy="random",
                    defer_strategy="random",
                    embeddings_by_id=embeddings_by_id,
                )
                selected_ids = selection.distillation_ids
                selection_payload = selection.to_dict()
            summary = write_dataset(
                output_dir=seed_dir,
                method=method,
                budget=int(budget),
                seed=int(args.seed),
                selected_ids=selected_ids,
                rows_by_id=rows_by_id,
                selection_payload=selection_payload,
                teacher_beta=float(args.teacher_beta),
                source_payload=source_payload,
            )
            summary["sampling_plan"] = plan.to_dict() if method == "ns-error-mass" else None
            dataset_summaries[summary["name"]] = summary
            write_json(summary, seed_dir / f"{summary['name']}.summary.json")

    aggregate = {
        "stage_name": "cgsd_make_fever_budget_sweep_sets",
        "elapsed_seconds": round(time.time() - started, 2),
        "seed": int(args.seed),
        "budgets": budgets,
        "methods": list(METHODS),
        "dataset_count": len(dataset_summaries),
        "temperature": float(args.temperature),
        "alpha": float(args.alpha),
        "round_summary_path": str(round0_dir / "round_summary.json"),
        "pool_summary": pool_summary,
        "guide_summary": guide_summary,
        "sampling_statistics": sampling_statistics,
        "ns_scoring": ns_scoring.to_dict(),
        "datasets": dataset_summaries,
    }
    write_json(aggregate, output_root / "budget_sweep_summary.json")
    print(
        json.dumps(
            {
                "output_dir": str(output_root),
                "dataset_count": len(dataset_summaries),
                "temperature": float(args.temperature),
                "alpha": float(args.alpha),
                "pool_defer_rate": pool_summary["defer_rate"],
                "datasets": {
                    name: {
                        "n": summary["selected_count"],
                        "accept": summary["accept_selected_count"],
                        "defer": summary["defer_selected_count"],
                        "base_error_rate": summary["base_error_rate"],
                        "label_counts": summary["label_counts"],
                    }
                    for name, summary in dataset_summaries.items()
                },
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
