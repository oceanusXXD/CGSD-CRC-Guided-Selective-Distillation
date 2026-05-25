"""Shared implementation for the three training-row selection entrypoints."""

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
    load_selected_train_rows,
    output_artifact_path,
    output_dir_from_arg,
    print_existing_stage_result,
    read_jsonl,
    stage_cache_decision,
    summarize_teacher_label_usage,
    write_stage_usage,
)
from src.binary_protocol import normalize_binary_label
from src.crc import compute_crc_error_mass_plan, compute_pcss_plan, routing_score_from_margin, select_training_ids
from src.utils import read_json, write_json, write_jsonl


def parse_selection_args(
    *,
    description: str,
    include_error_mass_options: bool = False,
) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--round_index", type=int, required=True)
    parser.add_argument("--guide_crc_predictions_path", default=None)
    parser.add_argument("--pool_crc_predictions_path", default=None)
    parser.add_argument("--crc_summary_path", default=None)
    parser.add_argument("--selected_train_rows_output_path", default=None)
    parser.add_argument("--cumulative_train_rows_output_path", default=None)
    parser.add_argument("--selection_summary_path", default=None)
    parser.add_argument("--usage_path", default=None)
    parser.add_argument("--budget", type=int, required=True)
    if include_error_mass_options:
        parser.add_argument("--accept_strategy", choices=["random", "high-confidence"], default="random")
        parser.add_argument("--defer_strategy", choices=["random", "high-confidence"], default="random")
    else:
        parser.set_defaults(accept_strategy="random", defer_strategy="random")
    parser.add_argument("--seed", type=int, default=42)
    add_stage_cache_args(parser)
    return parser.parse_args()


def _merge_by_id(rows: list[dict[str, object]]) -> dict[str, dict[str, object]]:
    return {str(row["id"]): row for row in rows}


def _proxy_prediction(row: dict[str, object]) -> int:
    if "prediction" in row:
        return normalize_binary_label(row["prediction"], field_name="row prediction")
    return 1 if float(row.get("score", 0.0) or 0.0) > 0.0 else 0


def _routing_score(row: dict[str, object], *, temperature: float) -> float:
    if row.get("routing_score") is not None:
        return float(row["routing_score"])
    if row.get("score") is not None:
        return routing_score_from_margin(float(row["score"]), temperature)
    raise ValueError(f"row missing routing_score/score: {row!r}")


def _stratum_priority_check(
    *,
    label: int,
    candidate_ids: list[str],
    selected_ids: list[str],
    rows_by_id: dict[str, dict[str, object]],
    temperature: float,
) -> dict[str, object]:
    selected_scores = {
        sample_id: _routing_score(rows_by_id[sample_id], temperature=temperature)
        for sample_id in selected_ids
    }
    selected_id_set = set(selected_ids)
    sorted_selected_ids = sorted(selected_ids, key=lambda sample_id: (selected_scores[sample_id], sample_id))
    selected_ordered_by_score = selected_ids == sorted_selected_ids
    unselected_scores = [
        _routing_score(rows_by_id[sample_id], temperature=temperature)
        for sample_id in candidate_ids
        if sample_id not in selected_id_set
    ]
    max_selected_score = max(selected_scores.values()) if selected_scores else None
    min_unselected_score = min(unselected_scores) if unselected_scores else None
    score_boundary_passed = (
        max_selected_score is None
        or min_unselected_score is None
        or float(max_selected_score) <= float(min_unselected_score) + 1e-12
    )
    return {
        "label": int(label),
        "candidate_count": int(len(candidate_ids)),
        "selected_count": int(len(selected_ids)),
        "selected_ordered_by_score": bool(selected_ordered_by_score),
        "score_boundary_passed": bool(score_boundary_passed),
        "max_selected_routing_score": max_selected_score,
        "min_unselected_routing_score": min_unselected_score,
        "passed": bool(selected_ordered_by_score and score_boundary_passed),
    }


def _pcss_training_property_checks(
    *,
    selection: object,
    pcss_plan: object,
    pool_by_id: dict[str, dict[str, object]],
    blocked_ids: set[str],
) -> dict[str, object]:
    selected_ids = list(selection.selected_ids)
    selected_id_set = set(selected_ids)
    selected_budget = len(selected_ids)
    selected_label1_count = len(selection.label1_ids)
    selected_label0_count = len(selection.label0_ids)
    selected_proxy_label1_rate = float(selected_label1_count / selected_budget) if selected_budget else 0.0
    capacity_limited = (
        int(selection.requested_label0_budget) != int(pcss_plan.target_label0_budget)
        or int(selection.requested_label1_budget) != int(pcss_plan.target_label1_budget)
    )
    distribution_within_rounding = (
        abs(float(selected_label1_count) - selected_budget * float(pcss_plan.p_hat_1)) <= 0.5
    )
    feasible_budget_match = (
        selected_label0_count == int(selection.requested_label0_budget)
        and selected_label1_count == int(selection.requested_label1_budget)
    )

    candidate_ids_by_label = {
        label: [
            sample_id
            for sample_id, row in pool_by_id.items()
            if sample_id not in blocked_ids and _proxy_prediction(row) == label
        ]
        for label in (0, 1)
    }
    label0_priority = _stratum_priority_check(
        label=0,
        candidate_ids=candidate_ids_by_label[0],
        selected_ids=list(selection.label0_ids),
        rows_by_id=pool_by_id,
        temperature=float(pcss_plan.temperature),
    )
    label1_priority = _stratum_priority_check(
        label=1,
        candidate_ids=candidate_ids_by_label[1],
        selected_ids=list(selection.label1_ids),
        rows_by_id=pool_by_id,
        temperature=float(pcss_plan.temperature),
    )

    selected_from_pool = all(sample_id in pool_by_id for sample_id in selected_ids)
    blocked_excluded = all(sample_id not in blocked_ids for sample_id in selected_ids)
    unique_ids = len(selected_id_set) == len(selected_ids)
    feasible_candidate_budget = min(
        int(selection.requested_budget),
        int(selection.label0_candidate_count) + int(selection.label1_candidate_count),
    )
    feasible_budget_filled = int(selection.selected_budget) == feasible_candidate_budget

    return {
        "property_1_distribution_consistency": {
            "target_guide_label1_rate": float(pcss_plan.p_hat_1),
            "selected_proxy_label1_rate": selected_proxy_label1_rate,
            "selected_label0_count": int(selected_label0_count),
            "selected_label1_count": int(selected_label1_count),
            "target_label0_budget": int(pcss_plan.target_label0_budget),
            "target_label1_budget": int(pcss_plan.target_label1_budget),
            "actual_label0_budget": int(selection.requested_label0_budget),
            "actual_label1_budget": int(selection.requested_label1_budget),
            "capacity_limited": bool(capacity_limited),
            "within_rounding_tolerance": bool(distribution_within_rounding),
            "feasible_budget_match": bool(feasible_budget_match),
            "passed": bool(feasible_budget_match and (distribution_within_rounding or capacity_limited)),
        },
        "property_2_uncertainty_priority": {
            "label0": label0_priority,
            "label1": label1_priority,
            "passed": bool(label0_priority["passed"] and label1_priority["passed"]),
        },
        "property_3_budget_and_uniqueness": {
            "selected_budget": int(selection.selected_budget),
            "requested_budget": int(selection.requested_budget),
            "feasible_candidate_budget": int(feasible_candidate_budget),
            "unique_ids": bool(unique_ids),
            "selected_from_pool": bool(selected_from_pool),
            "blocked_excluded": bool(blocked_excluded),
            "feasible_budget_filled": bool(feasible_budget_filled),
            "passed": bool(unique_ids and selected_from_pool and blocked_excluded and feasible_budget_filled),
        },
    }


def run_selection(args: argparse.Namespace, *, method: str) -> dict[str, object]:
    output_dir = output_dir_from_arg(args.output_dir)
    round_dir = output_dir / f"round_{args.round_index}"
    stage_name = f"select_{method.replace('-', '_')}"
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
        output_dir / "train_rows.jsonl",
    )
    selection_summary_path = output_artifact_path(args.selection_summary_path, round_dir / "selection_summary.json")
    usage_path = output_artifact_path(args.usage_path, round_dir / "select_usage.json")
    if args.show_result:
        print_existing_stage_result(stage_name=stage_name, summary_path=selection_summary_path)
        return {}

    cache_decision = stage_cache_decision(
        stage_name=stage_name,
        required_outputs=[selected_train_rows_output_path, cumulative_train_rows_output_path, selection_summary_path, usage_path],
        cache_policy=args.cache_policy,
    )
    if cache_decision.cache_hit:
        print_existing_stage_result(stage_name=stage_name, summary_path=selection_summary_path)
        return {}

    guide_crc_rows = read_jsonl(guide_crc_predictions_path)
    pool_crc_rows = read_jsonl(pool_crc_predictions_path)
    crc_summary = read_json(crc_summary_path)
    existing_rows = load_selected_train_rows(output_dir)
    blocked_ids = {str(row["id"]) for row in existing_rows}
    pcss_plan = None
    crc_error_mass_plan = None
    if method == "pcss":
        pcss_plan = compute_pcss_plan(
            guide_crc_rows,
            pool_crc_rows,
            budget=int(args.budget),
            temperature=float(crc_summary["temperature"]),
            lambda_hat=float(crc_summary["lambda_hat"]),
            alpha=float(crc_summary["alpha"]),
        )
    elif method == "crc-error-mass":
        crc_error_mass_plan = compute_crc_error_mass_plan(
            guide_crc_rows,
            pool_crc_rows,
            budget=int(args.budget),
            temperature=float(crc_summary["temperature"]),
            lambda_hat=float(crc_summary["lambda_hat"]),
            alpha=float(crc_summary["alpha"]),
    )
    selection = select_training_ids(
        pool_crc_rows,
        method=method,
        budget=int(args.budget),
        seed=int(args.seed) + int(args.round_index),
        blocked_ids=blocked_ids,
        crc_error_mass_plan=crc_error_mass_plan,
        pcss_plan=pcss_plan,
        accept_strategy=str(args.accept_strategy),
        defer_strategy=str(args.defer_strategy),
    )
    pool_by_id = _merge_by_id(pool_crc_rows)
    selected_label0_ids = set(selection.label0_ids)
    selected_label1_ids = set(selection.label1_ids)
    selected_rows: list[dict[str, object]] = []
    for sample_id in selection.selected_ids:
        row = dict(pool_by_id[sample_id])
        row["selection_round"] = int(args.round_index)
        row["selection_method"] = method
        row["selection_side"] = "defer" if bool(row.get("defer", False)) else "accept"
        if sample_id in selected_label1_ids:
            row["selection_stratum"] = "proxy_label_1"
        elif sample_id in selected_label0_ids:
            row["selection_stratum"] = "proxy_label_0"
        row.setdefault("label", row.get("groundtruth"))
        selected_rows.append(row)

    cumulative_rows_by_id = {str(row["id"]): row for row in existing_rows}
    for row in selected_rows:
        cumulative_rows_by_id[str(row["id"])] = row
    cumulative_rows = list(cumulative_rows_by_id.values())
    training_set_property_checks = None
    if method == "pcss" and pcss_plan is not None:
        training_set_property_checks = _pcss_training_property_checks(
            selection=selection,
            pcss_plan=pcss_plan,
            pool_by_id=pool_by_id,
            blocked_ids=blocked_ids,
        )
    summary = {
        "stage_name": stage_name,
        "round_index": int(args.round_index),
        "method": method,
        "budget": int(args.budget),
        "selection": selection.to_dict(),
        "pcss_plan": pcss_plan.to_dict() if pcss_plan is not None else None,
        "crc_error_mass_plan": crc_error_mass_plan.to_dict() if crc_error_mass_plan is not None else None,
        "training_set_property_checks": training_set_property_checks,
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
            "stage_name": stage_name,
            "round_index": int(args.round_index),
            "cache": cache_decision.to_dict(),
            "selected_rows": len(selected_rows),
            "cumulative_rows": len(cumulative_rows),
            "selection_method": method,
            "teacher_label_usage": teacher_usage,
            "groundtruth_substitute_calls": teacher_usage["groundtruth_substitute_calls"],
            "teacher_api_file_calls": teacher_usage["teacher_api_file_calls"],
        },
    )
    return summary


def main(
    *,
    method: str,
    description: str,
    include_error_mass_options: bool = False,
) -> None:
    args = parse_selection_args(
        description=description,
        include_error_mass_options=include_error_mass_options,
    )
    summary = run_selection(args, method=method)
    if not summary:
        return
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
