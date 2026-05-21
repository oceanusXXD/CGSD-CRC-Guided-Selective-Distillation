#!/usr/bin/env python
"""为单个 CGSD round 按自适应 accept/defer 比例选择蒸馏样本。

本阶段消费 CRC 已经写出的 `pool_crc_predictions.jsonl`，并根据更新方法中的
r_U、c_crc、eta_crc 公式把预算拆成 accept anchor 和 defer hard samples。
选出的样本会写入本轮 `selected_train_rows.jsonl`，同时并入累计训练集
`cgsd_train_rows.jsonl`。
"""

from __future__ import annotations

import argparse
import math
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
from algorithms.cgsd import (
    apply_crc_decisions,
    compute_adaptive_sampling_plan,
    select_documented_training_samples,
    teacher_weight,
)
from scripts.run_cgsd import assert_embedding_coverage, load_embeddings
from src.utils import read_json, write_json, write_jsonl


ACCEPT_STRATEGIES = ("random", "high-confidence")
DEFER_STRATEGIES = ("random", "k-center")
SELECTION_METHODS = ("crc-error-mass", "pool-random", "pure-accept", "pure-defer", "fixed-15-85")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--round_index", type=int, required=True)
    parser.add_argument("--split_ids_path", default=None)
    parser.add_argument("--calibration_predictions_path", default=None)
    parser.add_argument("--pool_crc_predictions_path", default=None)
    parser.add_argument("--pool_student_predictions_path", default=None)
    parser.add_argument("--round_summary_path", default=None)
    parser.add_argument("--selected_rows_input_path", default=None)
    parser.add_argument("--selected_train_rows_output_path", default=None)
    parser.add_argument("--cumulative_train_rows_output_path", default=None)
    parser.add_argument("--selection_summary_path", default=None)
    parser.add_argument("--usage_path", default=None)
    parser.add_argument("--embeddings_path", default=None)
    parser.add_argument("--embedding_dim", type=int, default=1024)
    parser.add_argument("--budget", type=int, default=None)
    parser.add_argument("--budget_schedule", default="250,150,100")
    parser.add_argument("--teacher_beta", type=float, default=1.0)
    parser.add_argument("--selection_method", choices=SELECTION_METHODS, default="crc-error-mass")
    parser.add_argument("--accept_strategy", choices=ACCEPT_STRATEGIES, default="random")
    parser.add_argument("--defer_strategy", choices=DEFER_STRATEGIES, default="random")
    parser.add_argument("--selection_buffer_multiplier", type=float, default=1.0)
    parser.add_argument("--teacher_confidence_filter", action="store_true", default=False)
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


def selection_budget_with_buffer(*, budget: int, buffer_multiplier: float) -> int:
    multiplier = float(buffer_multiplier)
    if not math.isfinite(multiplier) or multiplier < 1.0:
        raise ValueError("--selection_buffer_multiplier must be a finite value >= 1.0")
    return int(math.ceil(int(budget) * multiplier))


def teacher_confidence(row: dict[str, object]) -> float:
    return float(row.get("parsed_confidence", row.get("teacher_confidence", 1.0)) or 1.0)


def merge_prediction_and_decision(
    *,
    sample_id: str,
    prediction_by_id: dict[str, dict[str, object]],
    decision_by_id: dict[str, dict[str, object]],
) -> dict[str, object]:
    row = dict(prediction_by_id.get(sample_id, {}))
    row.update(decision_by_id[sample_id])
    return row


def keep_highest_teacher_confidence(rows: list[dict[str, object]], *, budget: int) -> list[dict[str, object]]:
    if len(rows) <= int(budget):
        return rows
    ranked = sorted(rows, key=lambda row: (-teacher_confidence(row), str(row.get("id", row.get("sample_id", "")))))
    keep_ids = {str(row.get("id", row.get("sample_id", ""))) for row in ranked[: int(budget)]}
    return [row for row in rows if str(row.get("id", row.get("sample_id", ""))) in keep_ids]


def split_excluded_ids(split_payload: dict[str, object]) -> set[str]:
    """汇总不能进入训练 pool 的 split ID。

    当前 prepare stage 已经让 `pool_ids` 排除了 guide/final calibration；
    这里额外兼容带 test set 的 FEVER 固定划分，防止手工输入的
    `pool_crc_predictions.jsonl` 混入测试或认证样本。
    """
    excluded: set[str] = set()
    for key in (
        "calibration_ids",
        "guide_ids",
        "test_ids",
        "d_test_ids",
        "D_test_ids",
        "final_calibration_ids",
        "cert_ids",
        "D_cert_ids",
    ):
        values = split_payload.get(key, [])
        if isinstance(values, list):
            excluded.update(str(sample_id) for sample_id in values)
    return excluded


def main() -> None:
    args = parse_args()
    output_dir = output_dir_from_arg(args.output_dir)
    round_dir = output_dir / f"round_{args.round_index}"
    pool_crc_predictions_path = input_artifact_path(
        args.pool_crc_predictions_path,
        round_dir / "pool_crc_predictions.jsonl",
    )
    calibration_predictions_path = input_artifact_path(
        args.calibration_predictions_path,
        round_dir / "calibration_student_predictions.jsonl",
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
    decision_by_id = {str(row["id"]): row for row in pool_decisions}
    prediction_by_id = {
        str(row["id"]): row
        for row in read_jsonl(pool_student_predictions_path)
    }
    budget = budget_for_round(
        round_index=int(args.round_index),
        budget=args.budget,
        budget_schedule=args.budget_schedule,
    )
    split_excluded_set = split_excluded_ids(split_payload)
    selected_rows_input = (
        read_jsonl(input_artifact_path(args.selected_rows_input_path, output_dir / "cgsd_train_rows.jsonl"))
        if args.selected_rows_input_path
        else load_selected_train_rows(output_dir)
    )
    selected_train_rows = {str(row["id"]): row for row in selected_rows_input}
    blocked_ids = set(selected_train_rows) | split_excluded_set
    round_summary = read_json(round_summary_path)
    crc_payload = round_summary.get("crc", {})
    neighbor_support_enabled = bool(
        crc_payload.get("neighbor_support_enabled", False)
        if isinstance(crc_payload, dict)
        else False
    )
    if not neighbor_support_enabled:
        raise ValueError("cgsd_select requires neighbor-support CRC; rerun cgsd_calibrate with --embeddings_path")
    embeddings_by_id = None
    embeddings_path = None
    needs_embeddings = (
        str(args.selection_method) == "crc-error-mass"
        or str(args.defer_strategy) == "k-center"
        or neighbor_support_enabled
    )
    if args.embeddings_path:
        embeddings_path = input_artifact_path(args.embeddings_path, PROJECT_ROOT / str(args.embeddings_path))
        embeddings_by_id = load_embeddings(embeddings_path)
        assert_embedding_coverage(embeddings_by_id, pool_decisions, expected_dim=int(args.embedding_dim))
    elif needs_embeddings:
        raise ValueError("--embeddings_path is required for k-center selection or neighbor-support CRC")

    candidate_budget = selection_budget_with_buffer(
        budget=budget,
        buffer_multiplier=float(args.selection_buffer_multiplier),
    )
    sampling_plan = None
    if str(args.selection_method) == "crc-error-mass":
        calibration_predictions = read_jsonl(calibration_predictions_path)
        calibration_decisions = apply_crc_decisions(
            calibration_predictions,
            lambda_hat=float(round_summary["lambda_hat"]),
            temperature=float(round_summary["temperature"]),
            embeddings_by_id=embeddings_by_id,
            support_rows=calibration_predictions,
            crc_result=crc_payload if isinstance(crc_payload, dict) else None,
            neighbor_exclude_self=True,
        )
        sampling_plan = compute_adaptive_sampling_plan(
            calibration_decisions,
            pool_decisions,
            budget=candidate_budget,
            temperature=float(round_summary["temperature"]),
            lambda_hat=float(round_summary["lambda_hat"]),
            alpha=float(round_summary.get("alpha", crc_payload.get("alpha", 0.0) if isinstance(crc_payload, dict) else 0.0)),
        )
    selection = select_documented_training_samples(
        pool_decisions,
        method=str(args.selection_method),
        budget=candidate_budget,
        seed=int(args.seed) + int(args.round_index),
        blocked_ids=blocked_ids,
        sampling_plan=sampling_plan,
        accept_strategy=str(args.accept_strategy),
        defer_strategy=str(args.defer_strategy),
        embeddings_by_id=embeddings_by_id,
    )
    selected_round_rows: list[dict[str, object]] = []
    for sample_id in selection.distillation_ids:
        row = merge_prediction_and_decision(
            sample_id=sample_id,
            prediction_by_id=prediction_by_id,
            decision_by_id=decision_by_id,
        )
        confidence = teacher_confidence(row)
        row["sample_weight"] = teacher_weight(confidence, float(args.teacher_beta))
        row["selection_round"] = int(args.round_index)
        side = "defer" if bool(row.get("defer", False)) else "accept"
        row["selection_role"] = f"{args.selection_method}_{side}".replace("-", "_")
        selected_round_rows.append(row)

    prefilter_selected_count = len(selected_round_rows)
    if args.teacher_confidence_filter:
        selected_round_rows = keep_highest_teacher_confidence(selected_round_rows, budget=budget)
    for row in selected_round_rows:
        selected_train_rows[str(row["id"])] = row

    selection_payload = selection.to_dict()
    selection_summary = {
        "stage_name": "cgsd_select",
        "round_index": int(args.round_index),
        "source_round_summary_path": str(round_summary_path),
        "source_lambda_hat": float(round_summary["lambda_hat"]),
        "selection_method": str(args.selection_method),
        "sampling_statistics": sampling_plan.to_dict() if sampling_plan is not None else None,
        "selection": selection_payload,
        "target_budget": int(budget),
        "candidate_budget": int(candidate_budget),
        "prefilter_selected_count": int(prefilter_selected_count),
        "teacher_confidence_filter": bool(args.teacher_confidence_filter),
        "final_selected_count": len(selected_round_rows),
        "accept_selected_count": sum(1 for row in selected_round_rows if not bool(row.get("defer", False))),
        "defer_selected_count": sum(1 for row in selected_round_rows if bool(row.get("defer", False))),
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
            "selection_method": str(args.selection_method),
            "accept_candidate_rows": int(selection.accept_candidate_count),
            "defer_candidate_rows": int(selection.defer_candidate_count),
            "estimated_selected_row_tokens": estimate_query_document_prompt_tokens(selected_round_rows),
            "teacher_label_usage": teacher_usage,
            "groundtruth_substitute_calls": teacher_usage["groundtruth_substitute_calls"],
            "teacher_api_file_calls": teacher_usage["teacher_api_file_calls"],
            "embedding": embedding_usage_payload(
                embedding_source=embeddings_path or "not_used",
                row_count=int(selection.defer_candidate_count) if embeddings_path else 0,
                embedding_dim=int(args.embedding_dim),
                purpose=f"{args.selection_method}_defer_{args.defer_strategy}",
            ),
        },
    )
    print(json.dumps(selection.to_dict(), ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
