#!/usr/bin/env python

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.cli_common import (
    add_stage_cache_args,
    input_artifact_path,
    output_artifact_path,
    output_dir_from_arg,
    print_existing_stage_result,
    read_jsonl,
    stage_cache_decision,
    write_stage_usage,
)
from src.crc import (
    apply_crc_defer_set,
    calibrate_crc,
    compute_pcss_plan,
    summarize_crc_decisions,
)
from src.metrics import compute_binary_metrics
from src.utils import write_json, write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--round_index", type=int, required=True)
    parser.add_argument("--guide_predictions_path", default=None)
    parser.add_argument("--pool_predictions_path", default=None)
    parser.add_argument("--guide_crc_predictions_path", default=None)
    parser.add_argument("--pool_crc_predictions_path", default=None)
    parser.add_argument("--crc_summary_path", default=None)
    parser.add_argument("--usage_path", default=None)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--temperature", type=float, default=15.0)
    parser.add_argument("--selection_budget", type=int, default=0)
    add_stage_cache_args(parser)
    return parser.parse_args()


def _labels(rows: list[dict[str, object]]) -> list[int]:
    return [int(row.get("label", row.get("groundtruth"))) for row in rows]


def _scores(rows: list[dict[str, object]]) -> list[float]:
    return [float(row["score"]) for row in rows]


def main() -> None:
    args = parse_args()
    output_dir = output_dir_from_arg(args.output_dir)
    round_dir = output_dir / f"round_{args.round_index}"
    guide_predictions_path = input_artifact_path(
        args.guide_predictions_path,
        round_dir / "guide_student_predictions.jsonl",
    )
    pool_predictions_path = input_artifact_path(
        args.pool_predictions_path,
        round_dir / "pool_student_predictions.jsonl",
    )
    guide_crc_predictions_path = output_artifact_path(
        args.guide_crc_predictions_path,
        round_dir / "guide_crc_predictions.jsonl",
    )
    pool_crc_predictions_path = output_artifact_path(
        args.pool_crc_predictions_path,
        round_dir / "pool_crc_predictions.jsonl",
    )
    crc_summary_path = output_artifact_path(args.crc_summary_path, round_dir / "crc_summary.json")
    usage_path = output_artifact_path(args.usage_path, round_dir / "crc_usage.json")
    if args.show_result:
        print_existing_stage_result(stage_name="compute_crc", summary_path=crc_summary_path)
        return

    cache_decision = stage_cache_decision(
        stage_name="compute_crc",
        required_outputs=[guide_crc_predictions_path, pool_crc_predictions_path, crc_summary_path, usage_path],
        cache_policy=args.cache_policy,
    )
    if cache_decision.cache_hit:
        print_existing_stage_result(stage_name="compute_crc", summary_path=crc_summary_path)
        return

    guide_predictions = read_jsonl(guide_predictions_path)
    pool_predictions = read_jsonl(pool_predictions_path)
    crc_result = calibrate_crc(
        guide_predictions,
        alpha=float(args.alpha),
        temperature=float(args.temperature),
    )
    guide_decisions = apply_crc_defer_set(
        guide_predictions,
        lambda_hat=crc_result.lambda_hat,
        temperature=crc_result.temperature,
    )
    pool_decisions = apply_crc_defer_set(
        pool_predictions,
        lambda_hat=crc_result.lambda_hat,
        temperature=crc_result.temperature,
    )
    selection_plan = None
    if int(args.selection_budget) > 0:
        selection_plan = compute_pcss_plan(
            guide_decisions,
            pool_decisions,
            budget=int(args.selection_budget),
            temperature=crc_result.temperature,
            lambda_hat=crc_result.lambda_hat,
            alpha=crc_result.alpha,
        )
    summary = {
        "round_index": int(args.round_index),
        "crc": crc_result.to_dict(),
        "temperature": float(crc_result.temperature),
        "alpha": float(crc_result.alpha),
        "lambda_hat": float(crc_result.lambda_hat),
        "guide_summary": summarize_crc_decisions(guide_decisions),
        "pool_summary": summarize_crc_decisions(pool_decisions),
        "pool_metrics": compute_binary_metrics(_labels(pool_predictions), _scores(pool_predictions)),
        "pcss_plan": selection_plan.to_dict() if selection_plan is not None else None,
        "crc_error_mass_plan": None,
        "guide_predictions_path": str(guide_predictions_path),
        "pool_predictions_path": str(pool_predictions_path),
        "guide_crc_predictions_path": str(guide_crc_predictions_path),
        "pool_crc_predictions_path": str(pool_crc_predictions_path),
    }
    write_jsonl(guide_decisions, guide_crc_predictions_path)
    write_jsonl(pool_decisions, pool_crc_predictions_path)
    write_json(summary, crc_summary_path)
    write_stage_usage(
        usage_path,
        {
            "stage_name": "compute_crc",
            "round_index": int(args.round_index),
            "cache": cache_decision.to_dict(),
            "guide_rows": len(guide_predictions),
            "pool_rows": len(pool_predictions),
            "student_model_calls": 0,
            "crc_summary_path": str(crc_summary_path),
        },
    )
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
