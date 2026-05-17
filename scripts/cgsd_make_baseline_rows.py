#!/usr/bin/env python
"""为 CGSD 实验 2 生成 baseline 训练行。"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from algorithms.cgsd import k_center_greedy, teacher_weight
from scripts.cgsd_cli_common import (
    add_stage_cache_args,
    binary_to_int,
    embedding_usage_payload,
    estimate_query_document_prompt_tokens,
    input_artifact_path,
    output_artifact_path,
    output_dir_from_arg,
    print_existing_stage_result,
    read_jsonl,
    stage_cache_decision,
    summarize_teacher_label_usage,
    write_stage_usage,
)
from scripts.run_cgsd import load_embeddings
from src.utils import read_json, write_json, write_jsonl


STRATEGIES = ("random", "uncertainty", "k-center", "defer-random")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--round_index", type=int, default=0)
    parser.add_argument("--strategy", choices=STRATEGIES, required=True)
    parser.add_argument("--budget", type=int, required=True)
    parser.add_argument("--split_ids_path", default=None)
    parser.add_argument("--pool_student_predictions_path", default=None)
    parser.add_argument("--pool_crc_predictions_path", default=None)
    parser.add_argument("--embeddings_path", default=None)
    parser.add_argument("--embedding_dim", type=int, default=1024)
    parser.add_argument("--train_rows_output_path", default=None)
    parser.add_argument("--summary_path", default=None)
    parser.add_argument("--usage_path", default=None)
    parser.add_argument("--teacher_beta", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    add_stage_cache_args(parser)
    return parser.parse_args()


def _row_id(row: dict[str, Any]) -> str:
    sample_id = row.get("id", row.get("sample_id"))
    if sample_id is None or str(sample_id) == "":
        raise ValueError(f"prediction row is missing id/sample_id: {row!r}")
    return str(sample_id)


def _routing_score(row: dict[str, Any]) -> float:
    for key in ("routing_score", "score", "confidence"):
        value = row.get(key)
        if value is not None:
            return float(value)
    raise ValueError(f"uncertainty baseline needs routing_score or score on row {_row_id(row)!r}")


def _label_from_row(row: dict[str, Any]) -> int:
    return binary_to_int(
        row.get("teacher_label", row.get("label", row.get("groundtruth"))),
        field_name=f"baseline row {_row_id(row)!r} label",
    )


def _confidence_from_row(row: dict[str, Any]) -> float:
    return float(row.get("teacher_confidence", row.get("parsed_confidence", 1.0)) or 1.0)


def _without_calibration(rows: list[dict[str, Any]], split_payload: dict[str, Any]) -> list[dict[str, Any]]:
    calibration_ids = {str(sample_id) for sample_id in split_payload.get("calibration_ids", [])}
    pool_ids = {str(sample_id) for sample_id in split_payload.get("pool_ids", [])}
    filtered: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in rows:
        sample_id = _row_id(row)
        if sample_id in seen or sample_id in calibration_ids:
            continue
        if pool_ids and sample_id not in pool_ids:
            continue
        seen.add(sample_id)
        filtered.append(dict(row))
    return filtered


def _merge_defer_flags(
    candidate_rows: list[dict[str, Any]],
    pool_crc_rows: list[dict[str, Any]] | None,
) -> list[dict[str, Any]]:
    if pool_crc_rows is None:
        return candidate_rows
    crc_by_id = {_row_id(row): row for row in pool_crc_rows}
    merged: list[dict[str, Any]] = []
    for row in candidate_rows:
        sample_id = _row_id(row)
        payload = dict(row)
        if sample_id in crc_by_id:
            payload["defer"] = bool(crc_by_id[sample_id].get("defer", False))
            if "routing_score" in crc_by_id[sample_id]:
                payload.setdefault("routing_score", crc_by_id[sample_id]["routing_score"])
        merged.append(payload)
    return merged


def select_candidate_rows(
    *,
    strategy: str,
    candidate_rows: list[dict[str, Any]],
    budget: int,
    seed: int,
    embeddings_by_id: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    requested = int(max(0, budget))
    if requested == 0:
        return []
    if strategy == "random":
        rows = list(candidate_rows)
        random.Random(seed).shuffle(rows)
        return rows[:requested]
    if strategy == "uncertainty":
        return sorted(candidate_rows, key=lambda row: (_routing_score(row), _row_id(row)))[:requested]
    if strategy == "defer-random":
        rows = [row for row in candidate_rows if bool(row.get("defer", False))]
        random.Random(seed).shuffle(rows)
        return rows[:requested]
    if strategy == "k-center":
        if embeddings_by_id is None:
            raise ValueError("--embeddings_path is required for --strategy k-center")
        candidate_ids = [_row_id(row) for row in candidate_rows]
        selected_ids = k_center_greedy(candidate_ids, embeddings_by_id, k=requested, seed=int(seed))
        by_id = {_row_id(row): row for row in candidate_rows}
        return [by_id[sample_id] for sample_id in selected_ids]
    raise ValueError(f"unsupported strategy: {strategy!r}")


def materialize_training_rows(
    selected_rows: list[dict[str, Any]],
    *,
    strategy: str,
    round_index: int,
    teacher_beta: float,
) -> list[dict[str, Any]]:
    train_rows: list[dict[str, Any]] = []
    for row in selected_rows:
        payload = dict(row)
        label = _label_from_row(payload)
        payload["label"] = label
        payload["groundtruth"] = label
        payload["teacher_label"] = label
        payload.setdefault("teacher_confidence", _confidence_from_row(payload))
        payload.setdefault("teacher_source", "groundtruth_substitute_for_real_teacher_api")
        payload.setdefault("teacher_label_source", "groundtruth")
        payload.setdefault("teacher_confidence_source", "fixed_1.0_groundtruth_substitute")
        payload["sample_weight"] = teacher_weight(float(payload["teacher_confidence"]), float(teacher_beta))
        payload["selection_round"] = int(round_index)
        payload["selection_role"] = f"baseline_{strategy.replace('-', '_')}"
        train_rows.append(payload)
    return train_rows


def make_baseline_rows(
    *,
    strategy: str,
    budget: int,
    split_ids_path: str | Path,
    pool_student_predictions_path: str | Path,
    pool_crc_predictions_path: str | Path | None,
    embeddings_path: str | Path | None,
    output_path: str | Path,
    summary_path: str | Path,
    seed: int,
    teacher_beta: float,
    round_index: int = 0,
    usage_path: str | Path | None = None,
    embedding_dim: int = 1024,
) -> list[dict[str, Any]]:
    split_payload = read_json(split_ids_path)
    student_rows = read_jsonl(pool_student_predictions_path)
    crc_rows = read_jsonl(pool_crc_predictions_path) if pool_crc_predictions_path is not None else None
    candidate_rows = _merge_defer_flags(_without_calibration(student_rows, split_payload), crc_rows)
    embeddings_by_id = load_embeddings(Path(embeddings_path)) if embeddings_path is not None else None
    selected_source_rows = select_candidate_rows(
        strategy=strategy,
        candidate_rows=candidate_rows,
        budget=int(budget),
        seed=int(seed),
        embeddings_by_id=embeddings_by_id,
    )
    train_rows = materialize_training_rows(
        selected_source_rows,
        strategy=strategy,
        round_index=int(round_index),
        teacher_beta=float(teacher_beta),
    )
    write_jsonl(train_rows, output_path)
    summary = {
        "stage_name": "cgsd_make_baseline_rows",
        "round_index": int(round_index),
        "strategy": strategy,
        "budget": int(budget),
        "candidate_rows": len(candidate_rows),
        "selected_rows": len(train_rows),
        "seed": int(seed),
        "teacher_beta": float(teacher_beta),
        "train_rows_output_path": str(output_path),
        "pool_student_predictions_path": str(pool_student_predictions_path),
        "pool_crc_predictions_path": str(pool_crc_predictions_path) if pool_crc_predictions_path else None,
        "embeddings_path": str(embeddings_path) if embeddings_path else None,
    }
    write_json(summary, summary_path)
    if usage_path is not None:
        teacher_usage = summarize_teacher_label_usage(train_rows, purpose=f"baseline_{strategy}_selected_rows")
        usage_payload: dict[str, Any] = {
            "stage_name": "cgsd_make_baseline_rows",
            "round_index": int(round_index),
            "strategy": strategy,
            "student_model_calls": 0,
            "selected_rows": len(train_rows),
            "candidate_rows": len(candidate_rows),
            "estimated_selected_row_tokens": estimate_query_document_prompt_tokens(train_rows),
            "teacher_label_usage": teacher_usage,
            "groundtruth_substitute_calls": teacher_usage["groundtruth_substitute_calls"],
            "teacher_api_file_calls": teacher_usage["teacher_api_file_calls"],
        }
        if embeddings_path is not None:
            usage_payload["embedding"] = embedding_usage_payload(
                embedding_source=embeddings_path,
                row_count=len(candidate_rows),
                embedding_dim=int(embedding_dim),
                purpose=f"baseline_{strategy}_selection",
            )
        write_stage_usage(usage_path, usage_payload)
    return train_rows


def main() -> None:
    args = parse_args()
    output_dir = output_dir_from_arg(args.output_dir)
    round_dir = output_dir / f"round_{args.round_index}"
    split_ids_path = input_artifact_path(args.split_ids_path, output_dir / "cgsd_split_ids.json")
    pool_student_predictions_path = input_artifact_path(
        args.pool_student_predictions_path,
        round_dir / "pool_student_predictions.jsonl",
    )
    pool_crc_predictions_path = (
        input_artifact_path(args.pool_crc_predictions_path, round_dir / "pool_crc_predictions.jsonl")
        if args.pool_crc_predictions_path or args.strategy == "defer-random"
        else None
    )
    embeddings_path = (
        input_artifact_path(args.embeddings_path, PROJECT_ROOT / str(args.embeddings_path))
        if args.embeddings_path or args.strategy == "k-center"
        else None
    )
    train_rows_output_path = output_artifact_path(
        args.train_rows_output_path,
        output_dir / "cgsd_train_rows.jsonl",
    )
    summary_path = output_artifact_path(
        args.summary_path,
        round_dir / "baseline_selection_summary.json",
    )
    usage_path = output_artifact_path(args.usage_path, round_dir / "baseline_selection_usage.json")
    if args.show_result:
        print_existing_stage_result(stage_name="cgsd_make_baseline_rows", summary_path=summary_path)
        return
    cache_decision = stage_cache_decision(
        stage_name="cgsd_make_baseline_rows",
        required_outputs=[train_rows_output_path, summary_path, usage_path],
        cache_policy=args.cache_policy,
    )
    if cache_decision.cache_hit:
        print_existing_stage_result(stage_name="cgsd_make_baseline_rows", summary_path=summary_path)
        return

    selected = make_baseline_rows(
        strategy=args.strategy,
        budget=int(args.budget),
        split_ids_path=split_ids_path,
        pool_student_predictions_path=pool_student_predictions_path,
        pool_crc_predictions_path=pool_crc_predictions_path,
        embeddings_path=embeddings_path,
        output_path=train_rows_output_path,
        summary_path=summary_path,
        usage_path=usage_path,
        seed=int(args.seed),
        teacher_beta=float(args.teacher_beta),
        round_index=int(args.round_index),
        embedding_dim=int(args.embedding_dim),
    )
    print(json.dumps({"strategy": args.strategy, "selected_rows": len(selected)}, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
