#!/usr/bin/env python
"""从 CRC accept/defer 结果中选择训练样本。

输入是 `guide_crc_predictions.jsonl`、`pool_crc_predictions.jsonl` 和
`crc_summary.json`。`--method random` 从 pool 均匀随机抽样；`--method
crc-error-mass` 会先根据 guide 错误浓缩度和 pool defer 率拆分 accept/defer
预算，再分别抽样。输出本轮 `selected_train_rows.jsonl` 和累计
`cgsd_train_rows.jsonl`，后者直接供训练脚本读取。
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
    input_artifact_path,
    load_selected_train_rows,
    output_artifact_path,
    output_dir_from_arg,
    print_existing_stage_result,
    read_jsonl,
    stage_cache_decision,
    summarize_teacher_label_usage,
    write_stage_usage,
)
from src.crc import compute_crc_error_mass_plan, select_training_ids
from src.utils import read_json, write_json, write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--round_index", type=int, required=True)
    parser.add_argument("--guide_crc_predictions_path", default=None)
    parser.add_argument("--pool_crc_predictions_path", default=None)
    parser.add_argument("--crc_summary_path", default=None)
    parser.add_argument("--selected_train_rows_output_path", default=None)
    parser.add_argument("--cumulative_train_rows_output_path", default=None)
    parser.add_argument("--selection_summary_path", default=None)
    parser.add_argument("--usage_path", default=None)
    parser.add_argument("--method", choices=["random", "crc-error-mass"], default="crc-error-mass")
    parser.add_argument("--budget", type=int, required=True)
    parser.add_argument("--accept_strategy", choices=["random", "high-confidence"], default="random")
    parser.add_argument("--defer_strategy", choices=["random"], default="random")
    parser.add_argument("--seed", type=int, default=42)
    add_stage_cache_args(parser)
    return parser.parse_args()


def _merge_by_id(rows: list[dict[str, object]]) -> dict[str, dict[str, object]]:
    return {str(row["id"]): row for row in rows}


def main() -> None:
    args = parse_args()
    output_dir = output_dir_from_arg(args.output_dir)
    round_dir = output_dir / f"round_{args.round_index}"
    guide_crc_predictions_path = input_artifact_path(
        args.guide_crc_predictions_path,
        round_dir / "guide_crc_predictions.jsonl",
    )
    pool_crc_predictions_path = input_artifact_path(
        args.pool_crc_predictions_path,
        round_dir / "pool_crc_predictions.jsonl",
    )
    crc_summary_path = input_artifact_path(args.crc_summary_path, round_dir / "crc_summary.json")
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
        print_existing_stage_result(stage_name="cgsd_select_data", summary_path=selection_summary_path)
        return

    cache_decision = stage_cache_decision(
        stage_name="cgsd_select_data",
        required_outputs=[selected_train_rows_output_path, cumulative_train_rows_output_path, selection_summary_path, usage_path],
        cache_policy=args.cache_policy,
    )
    if cache_decision.cache_hit:
        print_existing_stage_result(stage_name="cgsd_select_data", summary_path=selection_summary_path)
        return

    guide_crc_rows = read_jsonl(guide_crc_predictions_path)
    pool_crc_rows = read_jsonl(pool_crc_predictions_path)
    crc_summary = read_json(crc_summary_path)
    existing_rows = load_selected_train_rows(output_dir)
    blocked_ids = {str(row["id"]) for row in existing_rows}
    plan = None
    if args.method == "crc-error-mass":
        plan = compute_crc_error_mass_plan(
            guide_crc_rows,
            pool_crc_rows,
            budget=int(args.budget),
            temperature=float(crc_summary["temperature"]),
            lambda_hat=float(crc_summary["lambda_hat"]),
            alpha=float(crc_summary["alpha"]),
        )
    selection = select_training_ids(
        pool_crc_rows,
        method=str(args.method),
        budget=int(args.budget),
        seed=int(args.seed) + int(args.round_index),
        blocked_ids=blocked_ids,
        crc_error_mass_plan=plan,
        accept_strategy=str(args.accept_strategy),
        defer_strategy=str(args.defer_strategy),
    )
    pool_by_id = _merge_by_id(pool_crc_rows)
    selected_rows: list[dict[str, object]] = []
    for sample_id in selection.selected_ids:
        row = dict(pool_by_id[sample_id])
        row["selection_round"] = int(args.round_index)
        row["selection_method"] = str(args.method)
        row["selection_side"] = "defer" if bool(row.get("defer", False)) else "accept"
        row.setdefault("label", row.get("groundtruth"))
        selected_rows.append(row)

    cumulative_rows_by_id = {str(row["id"]): row for row in existing_rows}
    for row in selected_rows:
        cumulative_rows_by_id[str(row["id"])] = row
    cumulative_rows = list(cumulative_rows_by_id.values())
    summary = {
        "stage_name": "cgsd_select_data",
        "round_index": int(args.round_index),
        "method": str(args.method),
        "budget": int(args.budget),
        "selection": selection.to_dict(),
        "crc_error_mass_plan": plan.to_dict() if plan is not None else None,
        "selected_train_rows_output_path": str(selected_train_rows_output_path),
        "cumulative_train_rows_output_path": str(cumulative_train_rows_output_path),
    }
    write_jsonl(selected_rows, selected_train_rows_output_path)
    write_jsonl(cumulative_rows, cumulative_train_rows_output_path)
    write_json(summary, selection_summary_path)
    teacher_usage = summarize_teacher_label_usage(selected_rows, purpose="selected_training_rows")
    write_stage_usage(
        usage_path,
        {
            "stage_name": "cgsd_select_data",
            "round_index": int(args.round_index),
            "cache": cache_decision.to_dict(),
            "selected_rows": len(selected_rows),
            "cumulative_rows": len(cumulative_rows),
            "selection_method": str(args.method),
            "teacher_label_usage": teacher_usage,
            "groundtruth_substitute_calls": teacher_usage["groundtruth_substitute_calls"],
            "teacher_api_file_calls": teacher_usage["teacher_api_file_calls"],
        },
    )
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
