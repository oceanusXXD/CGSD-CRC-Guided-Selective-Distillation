#!/usr/bin/env python
"""根据已保存的 round 输出生成最终 CGSD 部署决策。

该阶段不重新校准阈值，只读取某一轮的 `round_summary.json` 和
`pool_crc_predictions.jsonl`：accept 样本采用小模型输出，defer 样本标记为
需要 teacher 或外部系统处理。严格最终认证应另行使用 D_cert 校准阈值。
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
    estimate_query_document_prompt_tokens,
    input_artifact_path,
    load_selected_train_rows,
    output_dir_from_arg,
    output_artifact_path,
    print_existing_stage_result,
    read_jsonl,
    stage_cache_decision,
    summarize_teacher_label_usage,
    train_label_snapshot,
    write_stage_usage,
)
from algorithms.cgsd import build_deployment_rows
from src.utils import read_json, write_json, write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--round_index", type=int, default=None)
    parser.add_argument("--round_summary_path", default=None)
    parser.add_argument("--pool_crc_predictions_path", default=None)
    parser.add_argument("--train_label_snapshot_path", default=None)
    parser.add_argument("--train_rows_path", default=None)
    parser.add_argument("--deployment_decisions_path", default=None)
    parser.add_argument("--summary_path", default=None)
    parser.add_argument("--usage_path", default=None)
    add_stage_cache_args(parser)
    return parser.parse_args()


def choose_best_round(output_dir: Path) -> int:
    """选择已校准 round 中保存 defer rate 最小的一轮。"""
    candidates: list[tuple[float, int]] = []
    for summary_path in sorted(output_dir.glob("round_*/round_summary.json")):
        try:
            round_index = int(summary_path.parent.name.split("_", 1)[1])
            payload = read_json(summary_path)
            defer_rate = float(payload["pool_summary"]["defer_rate"])
        except (KeyError, ValueError, IndexError):
            continue
        candidates.append((defer_rate, round_index))
    if not candidates:
        raise RuntimeError("no round_*/round_summary.json files found; run cgsd_calibrate.py first")
    return sorted(candidates, key=lambda item: (item[0], item[1]))[0][1]


def main() -> None:
    args = parse_args()
    output_dir = output_dir_from_arg(args.output_dir)
    deployment_decisions_path = output_artifact_path(args.deployment_decisions_path, output_dir / "deployment_decisions.jsonl")
    summary_path = output_artifact_path(args.summary_path, output_dir / "cgsd_summary.json")
    usage_path = output_artifact_path(args.usage_path, output_dir / "finalize_usage.json")
    if args.show_result:
        print_existing_stage_result(stage_name="cgsd_finalize", summary_path=summary_path)
        return
    cache_decision = stage_cache_decision(
        stage_name="cgsd_finalize",
        required_outputs=[deployment_decisions_path, summary_path, usage_path],
        cache_policy=args.cache_policy,
    )
    if cache_decision.cache_hit:
        print_existing_stage_result(stage_name="cgsd_finalize", summary_path=summary_path)
        return

    if args.round_index is None:
        round_index = choose_best_round(output_dir)
    else:
        round_index = int(args.round_index)

    round_dir = output_dir / f"round_{round_index}"
    round_summary_path = input_artifact_path(args.round_summary_path, round_dir / "round_summary.json")
    pool_crc_predictions_path = input_artifact_path(args.pool_crc_predictions_path, round_dir / "pool_crc_predictions.jsonl")
    train_snapshot_path = input_artifact_path(args.train_label_snapshot_path, round_dir / "train_label_snapshot.json")
    round_summary = read_json(round_summary_path)
    final_pool_rows = read_jsonl(pool_crc_predictions_path)
    train_rows_for_summary = (
        read_jsonl(input_artifact_path(args.train_rows_path, output_dir / "cgsd_train_rows.jsonl"))
        if args.train_rows_path
        else load_selected_train_rows(output_dir)
    )
    if train_snapshot_path.exists():
        train_label_by_id = read_json(train_snapshot_path)
    else:
        train_label_by_id = train_label_snapshot(train_rows_for_summary)
    deployment_rows = build_deployment_rows(
        final_pool_rows,
        train_label_by_id=train_label_by_id,
        lambda_hat=float(round_summary["lambda_hat"]),
        temperature=float(round_summary["temperature"]),
    )
    final_checkpoint_dir = None if round_index == 0 else str(round_dir / "model")
    summary = {
        "best_round_index": round_index,
        "best_lambda_hat": float(round_summary["lambda_hat"]),
        "best_temperature": float(round_summary["temperature"]),
        "best_pool_summary": round_summary["pool_summary"],
        "teacher_train_calls": len(train_label_by_id),
        "teacher_train_calls_total_spent": len(train_rows_for_summary) if train_rows_for_summary else len(train_label_by_id),
        "teacher_defer_calls": sum(1 for row in deployment_rows if bool(row.get("teacher_required", False))),
        "final_model_source": "zero_shot_base" if round_index == 0 else "lora_adapter",
        "final_checkpoint_dir": final_checkpoint_dir,
    }
    write_jsonl(deployment_rows, deployment_decisions_path)
    write_json(summary, summary_path)
    defer_rows = [row for row in deployment_rows if row.get("deployment_source") == "teacher_defer"]
    train_reuse_rows = [row for row in deployment_rows if row.get("deployment_source") == "teacher_train_label"]
    teacher_defer_usage = summarize_teacher_label_usage(defer_rows, purpose="deployment_defer_teacher_calls")
    write_stage_usage(
        usage_path,
        {
            "stage_name": "cgsd_finalize",
            "round_index": int(round_index),
            "cache": cache_decision.to_dict(),
            "student_model_calls": 0,
            "deployment_rows": len(deployment_rows),
            "student_accept_rows": sum(1 for row in deployment_rows if row.get("deployment_source") == "student_accept"),
            "teacher_train_label_reuse_rows": len(train_reuse_rows),
            "teacher_defer_usage": teacher_defer_usage,
            "groundtruth_substitute_calls": teacher_defer_usage["groundtruth_substitute_calls"],
            "teacher_api_file_calls": teacher_defer_usage["teacher_api_file_calls"],
            "estimated_defer_prompt_tokens": estimate_query_document_prompt_tokens(defer_rows),
            "deployment_decisions_path": str(deployment_decisions_path),
            "summary_path": str(summary_path),
        },
    )
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
