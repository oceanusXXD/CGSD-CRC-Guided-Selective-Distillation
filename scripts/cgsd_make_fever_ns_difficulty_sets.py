#!/usr/bin/env python
"""从缓存的 round0 预测生成 NS/CRC 训练集。"""

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
    select_ns_difficulty_global_samples,
    select_ns_error_mass_samples,
    summarize_crc_decisions,
    teacher_weight,
)
from scripts.cgsd_cli_common import binary_to_int, read_jsonl  # noqa: E402
from scripts.run_cgsd import assert_embedding_coverage, load_embeddings  # noqa: E402
from src.utils import write_json, write_jsonl  # noqa: E402


METHODS = ("random", "crc-error-mass", "ns-difficulty-global", "ns-error-mass")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--calibration_predictions_path",
        default="experiments/inputs/fever/qwen17b_round0_base/round_0/calibration_student_predictions.jsonl",
    )
    parser.add_argument(
        "--pool_predictions_path",
        default="experiments/inputs/fever/qwen17b_round0_base/round_0/pool_student_predictions.jsonl",
    )
    parser.add_argument(
        "--output_dir",
        default="experiments/inputs/fever/qwen17b_round0_t1_alpha010_ns_difficulty_global_3000_seeds1_2",
    )
    parser.add_argument("--embeddings_path", default="experiments/inputs/fever/embeddings.npy")
    parser.add_argument("--embedding_dim", type=int, default=0)
    parser.add_argument("--train_size", type=int, default=3000)
    parser.add_argument("--seeds", default="1,2")
    parser.add_argument("--methods", default=",".join(METHODS))
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--teacher_beta", type=float, default=1.0)
    parser.add_argument("--teacher_source_name", default="groundtruth")
    return parser.parse_args()


def resolve_path(path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute() or candidate.exists():
        return candidate
    return PROJECT_ROOT / candidate


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


def _row_id(row: Mapping[str, Any]) -> str:
    sample_id = row.get("id", row.get("sample_id"))
    if sample_id is None or str(sample_id) == "":
        raise ValueError(f"row is missing id/sample_id: {row!r}")
    return str(sample_id)


def _label(row: Mapping[str, Any]) -> int:
    return binary_to_int(row.get("groundtruth", row.get("label")), field_name=f"FEVER row {_row_id(row)!r} label")


def materialize_train_rows(
    ids: Sequence[str],
    *,
    rows_by_id: Mapping[str, dict[str, Any]],
    method: str,
    teacher_beta: float,
    teacher_source_name: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    source_name = str(teacher_source_name or "groundtruth")
    for sample_id in ids:
        payload = dict(rows_by_id[str(sample_id)])
        label = _label(payload)
        payload["label"] = label
        payload["groundtruth"] = label
        payload["teacher_label"] = label
        payload["teacher_confidence"] = 1.0
        payload["teacher_source"] = source_name
        payload["teacher_label_source"] = source_name
        payload["teacher_confidence_source"] = f"fixed_1.0_{source_name}"
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
    scoring_payload: dict[str, Any],
    teacher_beta: float,
    teacher_source_name: str,
    source_summary: dict[str, Any],
) -> dict[str, Any]:
    stem = f"{method.replace('-', '_')}_{int(train_size)}_seed{int(seed)}"
    raw_rows = [dict(rows_by_id[str(sample_id)]) for sample_id in selected_ids]
    train_rows = materialize_train_rows(
        selected_ids,
        rows_by_id=rows_by_id,
        method=method,
        teacher_beta=teacher_beta,
        teacher_source_name=teacher_source_name,
    )
    accept_count = sum(1 for row in raw_rows if not bool(row.get("defer", False)))
    defer_count = sum(1 for row in raw_rows if bool(row.get("defer", False)))
    summary = {
        "name": stem,
        "method": method,
        "requested_train_size": int(train_size),
        "selected_count": len(selected_ids),
        "shortfall": bool(selection_payload.get("shortfall", len(selected_ids) < int(train_size))),
        "accept_selected_count": int(accept_count),
        "defer_selected_count": int(defer_count),
        "label_counts": label_counts(train_rows),
        "ids_path": str(output_dir / f"{stem}.ids.json"),
        "jsonl_path": str(output_dir / f"{stem}.jsonl"),
        "train_rows_path": str(output_dir / f"{stem}.train_rows.jsonl"),
        "selection": selection_payload,
        "scoring": scoring_payload,
        "source": source_summary,
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
    train_size: int,
    seed: int,
    selection_method: str = "ns-error-mass",
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
        "requested_budget": int(train_size),
        "selected_budget": len(selected_ids),
        "requested_accept_budget": int(plan.B_accept),
        "requested_defer_budget": int(plan.B_defer),
        "selected_accept_budget": len(accept_selection.distillation_ids),
        "selected_defer_budget": len(defer_selection.distillation_ids),
        "selection_method": str(selection_method),
        "ns_weighting": "ns-error-mass",
        "accept_candidate_count": len(accept_rows),
        "defer_candidate_count": len(defer_rows),
        "shortfall": len(selected_ids) < int(train_size),
        "accept_ns_selection": accept_selection.to_dict(),
        "defer_ns_selection": defer_selection.to_dict(),
    }
    return selected_ids, payload


def main() -> None:
    args = parse_args()
    started = time.time()
    output_dir = resolve_path(args.output_dir)
    calibration_path = resolve_path(args.calibration_predictions_path)
    pool_path = resolve_path(args.pool_predictions_path)
    embeddings_path = resolve_path(args.embeddings_path)
    seeds = parse_int_csv(args.seeds)
    methods = parse_methods_csv(args.methods)

    guide_rows = read_jsonl(calibration_path)
    pool_rows = read_jsonl(pool_path)
    if not args.embeddings_path:
        raise ValueError("--embeddings_path is required because all CRC uses neighbor-support")
    embeddings_by_id = load_embeddings(embeddings_path)
    assert_embedding_coverage(
        embeddings_by_id,
        [*guide_rows, *pool_rows],
        expected_dim=int(args.embedding_dim),
    )

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
    plan = compute_adaptive_sampling_plan(
        guide_decisions,
        pool_decisions,
        budget=int(args.train_size),
        temperature=float(args.temperature),
        lambda_hat=float(crc.lambda_hat),
        alpha=float(args.alpha),
    )
    scored = score_ns_difficulty_global(
        pool_decisions,
        guide_decisions,
        embeddings_by_id=embeddings_by_id,
        e_all=float(plan.e_all),
    )
    scored_pool_by_id = {_row_id(row): dict(row) for row in scored.pool_rows}
    source_summary = {
        "calibration_predictions_path": str(calibration_path),
        "pool_predictions_path": str(pool_path),
        "embeddings_path": str(embeddings_path),
        "temperature": float(args.temperature),
        "alpha": float(args.alpha),
        "crc": crc.to_dict(),
        "guide_summary": summarize_crc_decisions(guide_decisions),
        "pool_summary": summarize_crc_decisions(pool_decisions),
        "sampling_plan": plan.to_dict(),
    }

    seed_summaries: dict[str, Any] = {}
    for seed in seeds:
        seed_output_dir = output_dir / f"seed{int(seed)}"
        seed_outputs: dict[str, Any] = {}
        for method in methods:
            if method == "random":
                selection = select_documented_training_samples(
                    scored.pool_rows,
                    method="pool-random",
                    budget=int(args.train_size),
                    seed=int(seed),
                )
                selected_ids = selection.distillation_ids
                selection_payload = selection.to_dict()
                selection_payload["selection_method"] = "random"
            elif method == "crc-error-mass":
                selection = select_documented_training_samples(
                    scored.pool_rows,
                    method="crc-error-mass",
                    budget=int(args.train_size),
                    seed=int(seed),
                    sampling_plan=plan,
                    accept_strategy="random",
                    defer_strategy="random",
                )
                selected_ids = selection.distillation_ids
                selection_payload = selection.to_dict()
            elif method == "ns-difficulty-global":
                selection = select_ns_difficulty_global_samples(
                    scored.pool_rows,
                    sampling_plan=plan,
                    budget=int(args.train_size),
                    seed=int(seed),
                )
                selected_ids = selection.distillation_ids
                selection_payload = selection.to_dict()
            elif method == "ns-error-mass":
                selected_ids, selection_payload = select_ns_error_mass_split(
                    scored.pool_rows,
                    plan=plan,
                    train_size=int(args.train_size),
                    seed=int(seed),
                    selection_method=method,
                )
            else:
                raise ValueError(f"unsupported method: {method}")
            seed_outputs[method] = write_dataset(
                output_dir=seed_output_dir,
                method=method,
                train_size=int(args.train_size),
                seed=int(seed),
                selected_ids=selected_ids,
                rows_by_id=scored_pool_by_id,
                selection_payload=selection_payload,
                scoring_payload=scored.to_dict(),
                teacher_beta=float(args.teacher_beta),
                teacher_source_name=str(args.teacher_source_name),
                source_summary=source_summary,
            )
        seed_summaries[f"seed{int(seed)}"] = seed_outputs

    aggregate = {
        "stage_name": "cgsd_make_fever_ns_difficulty_sets",
        "elapsed_seconds": round(time.time() - started, 2),
        "methods": methods,
        "seeds": seeds,
        "train_size": int(args.train_size),
        "temperature": float(args.temperature),
        "alpha": float(args.alpha),
        "source": source_summary,
        "scoring": scored.to_dict(),
        "seed_summaries": seed_summaries,
    }
    write_json(aggregate, output_dir / "ns_sampling_summary.json")
    print(json.dumps({"output_dir": str(output_dir), "methods": methods, "seeds": seeds, "train_size": int(args.train_size)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
