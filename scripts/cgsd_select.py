#!/usr/bin/env python
"""为单个 CGSD round 选择 DBDS 蒸馏样本。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.cgsd_cli_common import (
    add_stage_cache_args,
    binary_to_int,
    embedding_usage_payload,
    estimate_query_document_prompt_tokens,
    input_artifact_path,
    load_anchor_ids,
    load_selected_train_rows,
    load_split_ids,
    output_dir_from_arg,
    output_artifact_path,
    print_existing_stage_result,
    read_jsonl,
    stage_cache_decision,
    summarize_teacher_label_usage,
    write_stage_usage,
)
from algorithms.cgsd import select_dbds_samples, teacher_weight
from scripts.run_cgsd import assert_embedding_coverage, load_embeddings
from src.utils import read_json, write_json, write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--round_index", type=int, required=True)
    parser.add_argument("--split_ids_path", default=None)
    parser.add_argument("--pool_crc_predictions_path", default=None)
    parser.add_argument("--pool_student_predictions_path", default=None)
    parser.add_argument("--round_summary_path", default=None)
    parser.add_argument("--selected_rows_input_path", default=None)
    parser.add_argument("--selected_train_rows_output_path", default=None)
    parser.add_argument("--cumulative_train_rows_output_path", default=None)
    parser.add_argument("--anchor_candidate_pool_path", default=None)
    parser.add_argument("--selection_summary_path", default=None)
    parser.add_argument("--usage_path", default=None)
    parser.add_argument("--embeddings_path", required=True)
    parser.add_argument("--embedding_dim", type=int, default=1024)
    parser.add_argument("--budget", type=int, default=None)
    parser.add_argument("--budget_schedule", default="250,150,100")
    parser.add_argument("--delta", type=float, default=0.1)
    parser.add_argument("--easy_anchor_ratio", type=float, default=None)
    parser.add_argument("--anchor_count", type=int, default=None)
    parser.add_argument("--anchor_ids_path", default=None)
    parser.add_argument("--teacher_beta", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    add_stage_cache_args(parser)
    return parser.parse_args()


def budget_for_round(*, round_index: int, budget: int | None, budget_schedule: str) -> int:
    """从 CLI 预算或预算表得到当前 selection stage 的预算。"""
    if budget is not None:
        return int(budget)
    values = [int(part.strip()) for part in str(budget_schedule or "").split(",") if part.strip()]
    if int(round_index) < 0 or int(round_index) >= len(values):
        raise ValueError("--budget is required when --round_index is outside --budget_schedule")
    return int(values[int(round_index)])


def anchor_candidate_rows(
    pool_decisions: list[dict[str, object]],
    *,
    blocked_ids: set[str],
    cached_anchor_ids: set[str] | None,
) -> list[dict[str, object]]:
    """导出可缓存的 easy-anchor 候选池。

    候选池只包含非 defer、student 预测正确且高置信的样本；如果用户传入
    `--anchor_ids_path`，这里再和该缓存集合取交集，确保本轮只从指定集合选。
    """
    candidates: list[dict[str, object]] = []
    for row in pool_decisions:
        sample_id = str(row["id"])
        if sample_id in blocked_ids or bool(row.get("defer", False)):
            continue
        if cached_anchor_ids is not None and sample_id not in cached_anchor_ids:
            continue
        label_value = row.get("label", row.get("groundtruth"))
        if label_value is None or binary_to_int(row["prediction"], field_name="anchor prediction") != binary_to_int(
            label_value,
            field_name="anchor label",
        ):
            continue
        candidates.append(
            {
                "id": sample_id,
                "routing_score": float(row.get("routing_score", 0.0) or 0.0),
                "prediction": binary_to_int(row["prediction"], field_name="anchor prediction"),
                "label": binary_to_int(label_value, field_name="anchor label"),
            }
        )
    candidates.sort(key=lambda row: (-float(row["routing_score"]), str(row["id"])))
    return candidates


def main() -> None:
    args = parse_args()
    output_dir = output_dir_from_arg(args.output_dir)
    round_dir = output_dir / f"round_{args.round_index}"
    pool_crc_predictions_path = input_artifact_path(
        args.pool_crc_predictions_path,
        round_dir / "pool_crc_predictions.jsonl",
    )
    pool_student_predictions_path = input_artifact_path(
        args.pool_student_predictions_path,
        round_dir / "pool_student_predictions.jsonl",
    )
    round_summary_path = input_artifact_path(args.round_summary_path, round_dir / "round_summary.json")
    selected_train_rows_output_path = output_artifact_path(
        args.selected_train_rows_output_path,
        round_dir / "selected_train_rows.jsonl",
    )
    cumulative_train_rows_output_path = output_artifact_path(
        args.cumulative_train_rows_output_path,
        output_dir / "cgsd_train_rows.jsonl",
    )
    anchor_candidate_pool_path = output_artifact_path(
        args.anchor_candidate_pool_path,
        round_dir / "anchor_candidate_pool.jsonl",
    )
    selection_summary_path = output_artifact_path(args.selection_summary_path, round_dir / "selection_summary.json")
    usage_path = output_artifact_path(args.usage_path, round_dir / "select_usage.json")
    if args.show_result:
        print_existing_stage_result(stage_name="cgsd_select", summary_path=selection_summary_path)
        return
    cache_decision = stage_cache_decision(
        stage_name="cgsd_select",
        required_outputs=[
            selected_train_rows_output_path,
            cumulative_train_rows_output_path,
            anchor_candidate_pool_path,
            selection_summary_path,
            usage_path,
        ],
        cache_policy=args.cache_policy,
    )
    if cache_decision.cache_hit:
        print_existing_stage_result(stage_name="cgsd_select", summary_path=selection_summary_path)
        return

    split_payload = (
        read_json(input_artifact_path(args.split_ids_path, output_dir / "cgsd_split_ids.json"))
        if args.split_ids_path
        else load_split_ids(output_dir)
    )
    pool_decisions = read_jsonl(pool_crc_predictions_path)
    prediction_by_id = {
        str(row["id"]): row
        for row in read_jsonl(pool_student_predictions_path)
    }
    budget = budget_for_round(
        round_index=int(args.round_index),
        budget=args.budget,
        budget_schedule=args.budget_schedule,
    )
    calibration_set = {str(sample_id) for sample_id in split_payload["calibration_ids"]}
    selected_rows_input = (
        read_jsonl(input_artifact_path(args.selected_rows_input_path, output_dir / "cgsd_train_rows.jsonl"))
        if args.selected_rows_input_path
        else load_selected_train_rows(output_dir)
    )
    selected_train_rows = {str(row["id"]): row for row in selected_rows_input}
    blocked_ids = set(selected_train_rows) | calibration_set
    anchor_count = (
        int(args.anchor_count)
        if args.anchor_count is not None
        else None
    )
    anchor_ids_path = args.anchor_ids_path
    anchor_candidate_ids = load_anchor_ids(anchor_ids_path) if anchor_ids_path else None
    easy_anchor_ratio = float(args.easy_anchor_ratio) if args.easy_anchor_ratio is not None else 0.1
    embeddings_path = input_artifact_path(args.embeddings_path, PROJECT_ROOT / str(args.embeddings_path))
    embeddings_by_id = load_embeddings(embeddings_path)
    assert_embedding_coverage(embeddings_by_id, pool_decisions, expected_dim=int(args.embedding_dim))
    defer_ids = [str(row["id"]) for row in pool_decisions if bool(row.get("defer", False))]
    cached_anchor_set = set(anchor_candidate_ids) if anchor_candidate_ids is not None else None
    write_jsonl(
        anchor_candidate_rows(
            pool_decisions,
            blocked_ids=blocked_ids,
            cached_anchor_ids=cached_anchor_set,
        ),
        anchor_candidate_pool_path,
    )
    round_summary = read_json(round_summary_path)
    selection = select_dbds_samples(
        pool_decisions,
        defer_ids=defer_ids,
        already_selected_ids=blocked_ids,
        budget=budget,
        lambda_hat=float(round_summary["lambda_hat"]),
        embeddings_by_id=embeddings_by_id,
        delta=float(args.delta),
        easy_anchor_ratio=easy_anchor_ratio,
        anchor_count=anchor_count,
        anchor_candidate_ids=anchor_candidate_ids,
        seed=int(args.seed) + int(args.round_index),
    )
    selected_round_rows: list[dict[str, object]] = []
    for sample_id in selection.distillation_ids + selection.anchor_ids:
        row = dict(prediction_by_id[sample_id])
        confidence = float(row.get("parsed_confidence", row.get("teacher_confidence", 1.0)) or 1.0)
        row["sample_weight"] = teacher_weight(confidence, float(args.teacher_beta))
        row["selection_round"] = int(args.round_index)
        row["selection_role"] = "easy_anchor" if sample_id in set(selection.anchor_ids) else "dbds_defer"
        selected_train_rows[sample_id] = row
        selected_round_rows.append(row)

    selection_payload = selection.to_dict()
    selection_payload["effective_anchor_count_arg"] = anchor_count
    selection_payload["effective_easy_anchor_ratio"] = easy_anchor_ratio
    selection_payload["anchor_ids_path"] = str(anchor_ids_path) if anchor_ids_path else None
    selection_summary = {
        "stage_name": "cgsd_select",
        "round_index": int(args.round_index),
        "source_round_summary_path": str(round_summary_path),
        "source_lambda_hat": float(round_summary["lambda_hat"]),
        "selection": selection_payload,
        "selected_train_rows_output_path": str(selected_train_rows_output_path),
        "cumulative_train_rows_output_path": str(cumulative_train_rows_output_path),
        "anchor_candidate_pool_path": str(anchor_candidate_pool_path),
    }
    write_json(selection_summary, selection_summary_path)
    write_jsonl(selected_round_rows, selected_train_rows_output_path)
    write_jsonl(list(selected_train_rows.values()), cumulative_train_rows_output_path)
    teacher_usage = summarize_teacher_label_usage(selected_round_rows, purpose="distillation_selected_rows")
    write_stage_usage(
        usage_path,
        {
            "stage_name": "cgsd_select",
            "round_index": int(args.round_index),
            "cache": cache_decision.to_dict(),
            "student_model_calls": 0,
            "selected_rows": len(selected_round_rows),
            "defer_candidate_rows": len(defer_ids),
            "estimated_selected_row_tokens": estimate_query_document_prompt_tokens(selected_round_rows),
            "teacher_label_usage": teacher_usage,
            "groundtruth_substitute_calls": teacher_usage["groundtruth_substitute_calls"],
            "teacher_api_file_calls": teacher_usage["teacher_api_file_calls"],
            "embedding": embedding_usage_payload(
                embedding_source=embeddings_path,
                row_count=len(defer_ids),
                embedding_dim=int(args.embedding_dim),
                purpose="dbds_defer_candidate_selection",
            ),
        },
    )
    print(json.dumps(selection.to_dict(), ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
