#!/usr/bin/env python
"""为单个 CGSD round 从 defer 集选择 k-Center 蒸馏样本。

本阶段只消费 CRC 已经写出的 `pool_crc_predictions.jsonl`。候选样本必须
满足 `defer=true`，并且不在已标注训练集和 D_guide 中。选出的样本会写入
本轮 `selected_train_rows.jsonl`，同时并入累计训练集 `cgsd_train_rows.jsonl`。
"""

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
    embedding_usage_payload,
    estimate_query_document_prompt_tokens,
    input_artifact_path,
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
from algorithms.cgsd import select_defer_k_center_samples, teacher_weight
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
    parser.add_argument("--selection_summary_path", default=None)
    parser.add_argument("--usage_path", default=None)
    parser.add_argument("--embeddings_path", required=True)
    parser.add_argument("--embedding_dim", type=int, default=1024)
    parser.add_argument("--budget", type=int, default=None)
    parser.add_argument("--budget_schedule", default="250,150,100")
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
    selection_summary_path = output_artifact_path(args.selection_summary_path, round_dir / "selection_summary.json")
    usage_path = output_artifact_path(args.usage_path, round_dir / "select_usage.json")
    if args.show_result:
        print_existing_stage_result(stage_name="cgsd_select", summary_path=selection_summary_path)
        return
    cache_decision = stage_cache_decision(
        stage_name="cgsd_select",
        required_outputs=[
            selected_train_rows_output_path,
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
    embeddings_path = input_artifact_path(args.embeddings_path, PROJECT_ROOT / str(args.embeddings_path))
    embeddings_by_id = load_embeddings(embeddings_path)
    assert_embedding_coverage(embeddings_by_id, pool_decisions, expected_dim=int(args.embedding_dim))
    # 只从当前 CRC 判定为 defer 的样本中选择；accept 样本默认由小模型处理，
    # 不消耗本轮标注预算。
    defer_ids = [str(row["id"]) for row in pool_decisions if bool(row.get("defer", False))]
    round_summary = read_json(round_summary_path)
    selection = select_defer_k_center_samples(
        pool_decisions,
        defer_ids=defer_ids,
        already_selected_ids=blocked_ids,
        budget=budget,
        embeddings_by_id=embeddings_by_id,
        seed=int(args.seed) + int(args.round_index),
    )
    selected_round_rows: list[dict[str, object]] = []
    for sample_id in selection.distillation_ids:
        row = dict(prediction_by_id[sample_id])
        confidence = float(row.get("parsed_confidence", row.get("teacher_confidence", 1.0)) or 1.0)
        row["sample_weight"] = teacher_weight(confidence, float(args.teacher_beta))
        row["selection_round"] = int(args.round_index)
        row["selection_role"] = "defer_k_center"
        selected_train_rows[sample_id] = row
        selected_round_rows.append(row)

    selection_payload = selection.to_dict()
    selection_summary = {
        "stage_name": "cgsd_select",
        "round_index": int(args.round_index),
        "source_round_summary_path": str(round_summary_path),
        "source_lambda_hat": float(round_summary["lambda_hat"]),
        "selection": selection_payload,
        "selected_train_rows_output_path": str(selected_train_rows_output_path),
        "cumulative_train_rows_output_path": str(cumulative_train_rows_output_path),
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
                purpose="defer_k_center_candidate_selection",
            ),
        },
    )
    print(json.dumps(selection.to_dict(), ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
