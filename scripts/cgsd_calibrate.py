#!/usr/bin/env python
"""校准单个 CGSD round 的 CRC 阈值。"""

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
    estimate_query_document_prompt_tokens,
    input_artifact_path,
    output_dir_from_arg,
    output_artifact_path,
    print_existing_stage_result,
    read_jsonl,
    stage_cache_decision,
    summarize_teacher_label_usage,
    write_stage_usage,
)
from algorithms.cgsd import (
    DEFAULT_TEMPERATURE_GRID,
    apply_crc_decisions,
    calibrate_crc,
    choose_temperature,
    evaluate_stop_criteria,
    summarize_crc_decisions,
)
from src.metrics import compute_binary_metrics
from src.utils import read_json, write_json, write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--round_index", type=int, required=True)
    parser.add_argument("--calibration_predictions_path", default=None)
    parser.add_argument("--pool_predictions_path", default=None)
    parser.add_argument("--pool_crc_predictions_path", default=None)
    parser.add_argument("--round_summary_path", default=None)
    parser.add_argument("--previous_round_summary_path", default=None)
    parser.add_argument("--previous_selection_summary_path", default=None)
    parser.add_argument("--train_rows_path", default=None)
    parser.add_argument("--alpha", type=float, default=0.07)
    parser.add_argument("--temperatures", default=",".join(str(value) for value in DEFAULT_TEMPERATURE_GRID))
    parser.add_argument("--threshold", type=float, default=0.0)
    parser.add_argument("--total_budget", type=int, default=500)
    parser.add_argument("--budget_schedule", default="250,150,100")
    parser.add_argument("--max_rounds", type=int, default=3)
    parser.add_argument("--min_delta_defer_rate", type=float, default=0.005)
    parser.add_argument("--no_economic_stop", action="store_true", default=False)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--usage_path", default=None)
    add_stage_cache_args(parser)
    return parser.parse_args()


def parse_float_csv(text: str) -> list[float]:
    """解析 CLI 传入的温度网格。"""
    values = [float(part.strip()) for part in str(text or "").split(",") if part.strip()]
    if not values:
        raise ValueError("--temperatures cannot be empty")
    return values


def parse_int_csv(text: str) -> list[int]:
    """解析 CLI 传入的每轮预算表。"""
    values = [int(part.strip()) for part in str(text or "").split(",") if part.strip()]
    if not values:
        raise ValueError("--budget_schedule cannot be empty")
    return values


def fixed_temperature_for_round(
    *,
    round_index: int,
    explicit_temperature: float | None,
    previous_round_summary_path: Path | None,
) -> tuple[float | None, str]:
    """解析后续 round 必须复用的固定温度。

    严格实验应显式传 `--temperature 15`。round0 保留温度网格搜索只为
    兼容诊断实验；round1 及以后只能复用已固定温度，避免后续轮次
    悄悄重新扫描温度并改变 routing score 口径。
    """
    if explicit_temperature is not None:
        return float(explicit_temperature), "cli_arg"
    if int(round_index) == 0:
        return None, "round0_temperature_search"
    if previous_round_summary_path is not None and previous_round_summary_path.exists():
        previous_summary = read_json(previous_round_summary_path)
        if "temperature" in previous_summary:
            return float(previous_summary["temperature"]), "previous_round_summary"
    raise RuntimeError(
        "round_index > 0 requires a fixed temperature from --temperature or the previous round_summary.json"
    )


def selected_count_for_stop(
    *,
    previous_selection_summary_path: Path,
    previous_summary: dict[str, object],
    train_rows: list[dict[str, object]],
    previous_round_index: int,
) -> int:
    """读取上一轮 selection 的样本数，用于停止标准的成本项。"""
    if previous_selection_summary_path.exists():
        selection = read_json(previous_selection_summary_path).get("selection", {})
    else:
        selection = previous_summary.get("selection", {})
    if isinstance(selection, dict) and selection:
        return int(selection.get("selected_budget", 0) or 0) + int(selection.get("anchor_budget", 0) or 0)
    count = 0
    for row in train_rows:
        try:
            selection_round = int(row.get("selection_round", -1) or -1)
        except (TypeError, ValueError):
            continue
        if selection_round == int(previous_round_index):
            count += 1
    return count


def main() -> None:
    args = parse_args()
    output_dir = output_dir_from_arg(args.output_dir)
    round_dir = output_dir / f"round_{args.round_index}"
    calibration_predictions_path = input_artifact_path(
        args.calibration_predictions_path,
        round_dir / "calibration_student_predictions.jsonl",
    )
    pool_predictions_path = input_artifact_path(args.pool_predictions_path, round_dir / "pool_student_predictions.jsonl")
    pool_crc_predictions_path = output_artifact_path(
        args.pool_crc_predictions_path,
        round_dir / "pool_crc_predictions.jsonl",
    )
    round_summary_path = output_artifact_path(args.round_summary_path, round_dir / "round_summary.json")
    usage_path = output_artifact_path(args.usage_path, round_dir / "calibrate_usage.json")
    if args.show_result:
        print_existing_stage_result(stage_name="cgsd_calibrate", summary_path=round_summary_path)
        return
    cache_decision = stage_cache_decision(
        stage_name="cgsd_calibrate",
        required_outputs=[pool_crc_predictions_path, round_summary_path, usage_path],
        cache_policy=args.cache_policy,
    )
    if cache_decision.cache_hit:
        print_existing_stage_result(stage_name="cgsd_calibrate", summary_path=round_summary_path)
        return

    calibration_predictions = read_jsonl(calibration_predictions_path)
    pool_predictions = read_jsonl(pool_predictions_path)

    previous_summary_path = (
        input_artifact_path(
            args.previous_round_summary_path,
            output_dir / f"round_{args.round_index - 1}" / "round_summary.json",
        )
        if args.previous_round_summary_path
        else output_dir / f"round_{args.round_index - 1}" / "round_summary.json"
    )
    previous_selection_summary_path = input_artifact_path(
        args.previous_selection_summary_path,
        output_dir / f"round_{args.round_index - 1}" / "selection_summary.json",
    )
    fixed_temperature, temperature_source = fixed_temperature_for_round(
        round_index=int(args.round_index),
        explicit_temperature=args.temperature,
        previous_round_summary_path=previous_summary_path,
    )
    if fixed_temperature is not None:
        temperature = float(fixed_temperature)
        crc_result = calibrate_crc(
            calibration_predictions,
            alpha=float(args.alpha),
            temperature=temperature,
        )
        temperature_payload = {
            "temperature": temperature,
            "crc": crc_result.to_dict(),
            "source": temperature_source,
        }
    else:
        temperature_choice = choose_temperature(
            calibration_predictions,
            pool_predictions,
            alpha=float(args.alpha),
            temperatures=parse_float_csv(args.temperatures),
        )
        temperature = float(temperature_choice.temperature)
        crc_result = temperature_choice.crc
        temperature_payload = temperature_choice.to_dict()
        temperature_payload["source"] = temperature_source

    pool_decisions = apply_crc_decisions(
        pool_predictions,
        lambda_hat=crc_result.lambda_hat,
        temperature=temperature,
    )
    summary = summarize_crc_decisions(pool_decisions)
    metrics = compute_binary_metrics(
        [binary_to_int(row["label"], field_name="pool prediction label") for row in pool_predictions],
        [float(row["score"]) for row in pool_predictions],
        threshold=float(args.threshold),
    )
    record = {
        "round_index": int(args.round_index),
        "crc": crc_result.to_dict(),
        "temperature_choice": temperature_payload,
        "temperature": float(temperature),
        "lambda_hat": float(crc_result.lambda_hat),
        "pool_summary": summary,
        "pool_metrics": metrics,
    }
    if args.round_index > 0 and previous_summary_path.exists():
        previous_summary = read_json(previous_summary_path)
        train_rows_path = input_artifact_path(args.train_rows_path, output_dir / "cgsd_train_rows.jsonl")
        train_rows = read_jsonl(train_rows_path) if train_rows_path.exists() else []
        previous_selected_count = selected_count_for_stop(
            previous_selection_summary_path=previous_selection_summary_path,
            previous_summary=previous_summary,
            train_rows=train_rows,
            previous_round_index=int(args.round_index) - 1,
        )
        total_selected_count = len(train_rows) if train_rows else previous_selected_count
        budget_schedule = parse_int_csv(args.budget_schedule)
        stop_decision = evaluate_stop_criteria(
            previous_defer_rate=float(previous_summary["pool_summary"]["defer_rate"]),
            current_defer_rate=float(summary["defer_rate"]),
            round_selected_count=previous_selected_count,
            total_selected_count=total_selected_count,
            total_budget=int(args.total_budget),
            completed_rounds=int(args.round_index),
            max_rounds=len(budget_schedule) if budget_schedule else int(args.max_rounds),
            pool_size=len(pool_predictions),
            min_delta_defer_rate=float(args.min_delta_defer_rate),
            use_economic_stop=not bool(args.no_economic_stop),
        )
        record["stop_after_round"] = bool(stop_decision.should_stop)
        record["stop_decision"] = stop_decision.to_dict()
        record["stop_reason"] = ",".join(stop_decision.reasons)
        record["stop_decision_scope"] = (
            "本轮训练后的预测和 CRC 校准完成后记录；"
            "本 stage 只写出停止判断，不启动或停止其他 stage"
        )
    write_jsonl(pool_decisions, pool_crc_predictions_path)
    write_json(record, round_summary_path)
    calibration_teacher_usage = summarize_teacher_label_usage(
        calibration_predictions,
        purpose="crc_calibration_teacher_labels",
    )
    write_stage_usage(
        usage_path,
        {
            "stage_name": "cgsd_calibrate",
            "round_index": int(args.round_index),
            "cache": cache_decision.to_dict(),
            "student_model_calls": 0,
            "crc_calibration_rows": len(calibration_predictions),
            "pool_decision_rows": len(pool_predictions),
            "estimated_crc_input_tokens": estimate_query_document_prompt_tokens(calibration_predictions),
            "teacher_label_usage": calibration_teacher_usage,
            "groundtruth_substitute_calls": calibration_teacher_usage["groundtruth_substitute_calls"],
            "teacher_api_file_calls": calibration_teacher_usage["teacher_api_file_calls"],
            "pool_crc_predictions_path": str(pool_crc_predictions_path),
            "round_summary_path": str(round_summary_path),
        },
    )
    print(json.dumps(record, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
