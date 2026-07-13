from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.cli_common import add_stage_cache_args, input_artifact_path, load_selected_train_rows, output_artifact_path, output_dir_from_arg, print_existing_stage_result, read_jsonl, stage_cache_decision, summarize_teacher_label_usage, write_stage_usage
from mias_dcms.binary_protocol import normalize_binary_label
from mias_dcms.crc import adaptive_label1_probability, compute_adaptive_selection_plan, compute_crc_error_mass_plan, compute_pcss_plan, routing_score_from_margin, select_training_ids
from mias_dcms.utils import read_json, write_json, write_jsonl


def parse_selection_args(*, description: str, include_error_mass_options: bool=False) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--round_index', type=int, required=True)
    parser.add_argument('--guide_crc_predictions_path', default=None)
    parser.add_argument('--pool_crc_predictions_path', default=None)
    parser.add_argument('--crc_summary_path', default=None)
    parser.add_argument('--selected_train_rows_output_path', default=None)
    parser.add_argument('--cumulative_train_rows_output_path', default=None)
    parser.add_argument('--selection_summary_path', default=None)
    parser.add_argument('--usage_path', default=None)
    parser.add_argument('--budget', type=int, required=True)
    if include_error_mass_options:
        parser.add_argument('--accept_strategy', choices=['random', 'high-confidence'], default='random')
        parser.add_argument('--defer_strategy', choices=['random', 'high-confidence'], default='random')
    else:
        parser.set_defaults(accept_strategy='random', defer_strategy='random')
    parser.add_argument('--seed', type=int, default=42)
    add_stage_cache_args(parser)
    return parser.parse_args()


def _merge_by_id(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(row['id']): row for row in rows}


def _proxy_prediction(row: dict[str, Any]) -> int:
    if 'prediction' in row:
        return normalize_binary_label(row['prediction'], field_name='row prediction')
    return 1 if float(row.get('score', 0.0) or 0.0) > 0.0 else 0


def _routing_score(row: dict[str, Any], *, temperature: float) -> float:
    if row.get('routing_score') is not None:
        return float(row['routing_score'])
    if row.get('score') is not None:
        return routing_score_from_margin(float(row['score']), temperature)
    raise ValueError(f'row missing routing_score/score: {row!r}')


def _stratum_priority_check(*, label: int, candidate_ids: list[str], selected_ids: list[str], rows_by_id: dict[str, dict[str, Any]], temperature: float) -> dict[str, Any]:
    selected_scores = {
        sample_id: _routing_score(rows_by_id[sample_id], temperature=temperature)
        for sample_id in selected_ids
    }
    selected_id_set = set(selected_ids)
    unselected_scores = [
        _routing_score(rows_by_id[sample_id], temperature=temperature)
        for sample_id in candidate_ids
        if sample_id not in selected_id_set
    ]
    max_selected_score = max(selected_scores.values()) if selected_scores else None
    min_unselected_score = min(unselected_scores) if unselected_scores else None
    selected_ordered_by_score = selected_ids == sorted(selected_ids, key=lambda sample_id: (selected_scores[sample_id], sample_id))
    score_boundary_passed = max_selected_score is None or min_unselected_score is None or float(max_selected_score) <= float(min_unselected_score) + 1e-12
    return {
        'label': int(label),
        'candidate_count': len(candidate_ids),
        'selected_count': len(selected_ids),
        'selected_ordered_by_score': bool(selected_ordered_by_score),
        'score_boundary_passed': bool(score_boundary_passed),
        'max_selected_routing_score': max_selected_score,
        'min_unselected_routing_score': min_unselected_score,
        'passed': bool(selected_ordered_by_score and score_boundary_passed),
    }


def _selection_integrity(selection: Any, selected_ids: list[str], pool_by_id: dict[str, dict[str, Any]], blocked_ids: set[str], *, risk_plan_valid: bool | None=None) -> dict[str, Any]:
    unique_ids = len(set(selected_ids)) == len(selected_ids)
    selected_from_pool = all(sample_id in pool_by_id for sample_id in selected_ids)
    blocked_excluded = all(sample_id not in blocked_ids for sample_id in selected_ids)
    feasible_candidate_budget = min(
        int(selection.requested_budget),
        int(selection.label0_candidate_count) + int(selection.label1_candidate_count),
    )
    feasible_budget_filled = int(selection.selected_budget) == feasible_candidate_budget
    passed = unique_ids and selected_from_pool and blocked_excluded and feasible_budget_filled
    if risk_plan_valid is not None:
        passed = passed and bool(risk_plan_valid)
    result = {
        'selected_budget': int(selection.selected_budget),
        'requested_budget': int(selection.requested_budget),
        'feasible_candidate_budget': int(feasible_candidate_budget),
        'unique_ids': bool(unique_ids),
        'selected_from_pool': bool(selected_from_pool),
        'blocked_excluded': bool(blocked_excluded),
        'feasible_budget_filled': bool(feasible_budget_filled),
        'passed': bool(passed),
    }
    if risk_plan_valid is not None:
        result['risk_plan_valid'] = bool(risk_plan_valid)
    return result


def _candidate_ids_by_proxy_label(pool_by_id: dict[str, dict[str, Any]], blocked_ids: set[str]) -> dict[int, list[str]]:
    return {
        label: [
            sample_id
            for sample_id, row in pool_by_id.items()
            if sample_id not in blocked_ids and _proxy_prediction(row) == label
        ]
        for label in (0, 1)
    }


def _pcss_training_property_checks(*, selection: Any, pcss_plan: Any, pool_by_id: dict[str, dict[str, Any]], blocked_ids: set[str]) -> dict[str, Any]:
    selected_ids = list(selection.selected_ids)
    selected_budget = len(selected_ids)
    selected_label1_count = len(selection.label1_ids)
    selected_label0_count = len(selection.label0_ids)
    capacity_limited = int(selection.requested_label0_budget) != int(pcss_plan.target_label0_budget) or int(selection.requested_label1_budget) != int(pcss_plan.target_label1_budget)
    distribution_within_rounding = abs(float(selected_label1_count) - selected_budget * float(pcss_plan.p_hat_1)) <= 0.5
    feasible_budget_match = selected_label0_count == int(selection.requested_label0_budget) and selected_label1_count == int(selection.requested_label1_budget)
    candidate_ids_by_label = _candidate_ids_by_proxy_label(pool_by_id, blocked_ids)
    label0_priority = _stratum_priority_check(label=0, candidate_ids=candidate_ids_by_label[0], selected_ids=list(selection.label0_ids), rows_by_id=pool_by_id, temperature=float(pcss_plan.temperature))
    label1_priority = _stratum_priority_check(label=1, candidate_ids=candidate_ids_by_label[1], selected_ids=list(selection.label1_ids), rows_by_id=pool_by_id, temperature=float(pcss_plan.temperature))
    return {
        'property_1_distribution_consistency': {
            'target_guide_label1_rate': float(pcss_plan.p_hat_1),
            'selected_proxy_label1_rate': float(selected_label1_count / selected_budget) if selected_budget else 0.0,
            'selected_label0_count': int(selected_label0_count),
            'selected_label1_count': int(selected_label1_count),
            'target_label0_budget': int(pcss_plan.target_label0_budget),
            'target_label1_budget': int(pcss_plan.target_label1_budget),
            'actual_label0_budget': int(selection.requested_label0_budget),
            'actual_label1_budget': int(selection.requested_label1_budget),
            'capacity_limited': bool(capacity_limited),
            'within_rounding_tolerance': bool(distribution_within_rounding),
            'feasible_budget_match': bool(feasible_budget_match),
            'passed': bool(feasible_budget_match and (distribution_within_rounding or capacity_limited)),
        },
        'property_2_uncertainty_priority': {
            'label0': label0_priority,
            'label1': label1_priority,
            'passed': bool(label0_priority['passed'] and label1_priority['passed']),
        },
        'property_3_budget_and_uniqueness': _selection_integrity(selection, selected_ids, pool_by_id, blocked_ids),
    }


def _confidence_bucket_index(routing_score: float, *, bucket_count: int) -> int:
    if bucket_count <= 1:
        return 0
    clipped = min(0.999999, max(0.5, float(routing_score)))
    return min(bucket_count - 1, max(0, int(((clipped - 0.5) / 0.5) * bucket_count)))


def _bucket_counts(sample_ids: list[str], routing_scores: dict[str, float], bucket_count: int) -> dict[str, int]:
    counts: dict[int, int] = {}
    for sample_id in sample_ids:
        bucket = _confidence_bucket_index(routing_scores[sample_id], bucket_count=bucket_count)
        counts[bucket] = counts.get(bucket, 0) + 1
    return {str(bucket): count for bucket, count in sorted(counts.items())}


def _risk_plan_valid(adaptive_plan: Any) -> bool:
    weights = [float(adaptive_plan.w_uncertainty), float(adaptive_plan.w_alignment), float(adaptive_plan.w_diversity)]
    return (
        0.0 <= float(adaptive_plan.risk_strength) <= 1.0
        and int(adaptive_plan.bucket_count) >= 1
        and all(weight >= 0.0 for weight in weights)
        and sum(weights) > 0.0
    )


def _adaptive_training_property_checks(*, selection: Any, adaptive_plan: Any, pool_by_id: dict[str, dict[str, Any]], blocked_ids: set[str]) -> dict[str, Any]:
    selected_ids = list(selection.selected_ids)
    selected_budget = len(selected_ids)
    selected_label1_count = len(selection.label1_ids)
    selected_label0_count = len(selection.label0_ids)
    selected_soft_label1_mass = sum(adaptive_label1_probability(pool_by_id[sample_id], adaptive_plan) for sample_id in selected_ids)
    selected_soft_label0_mass = float(selected_budget) - selected_soft_label1_mass
    hard_label_repair_enabled = bool(float(adaptive_plan.guide_proxy_reliability) >= 0.5)
    actual_target_label1_mass = float(selected_budget) * float(adaptive_plan.target_yes_rate)
    actual_target_label0_mass = float(selected_budget) - actual_target_label1_mass
    capacity_limited = int(selection.requested_label0_budget) != int(adaptive_plan.target_label0_budget) or int(selection.requested_label1_budget) != int(adaptive_plan.target_label1_budget)
    distribution_within_rounding = abs(float(selected_soft_label1_mass) - actual_target_label1_mass) <= 0.5
    feasible_budget_match = abs(selected_soft_label0_mass - actual_target_label0_mass) <= 0.5 and abs(selected_soft_label1_mass - actual_target_label1_mass) <= 0.5
    candidate_ids = [sample_id for sample_id in pool_by_id if sample_id not in blocked_ids]
    routing_scores = {
        sample_id: _routing_score(pool_by_id[sample_id], temperature=float(adaptive_plan.temperature))
        for sample_id in candidate_ids
    }
    finite_selected_scores = all(
        math.isfinite(float(routing_scores[sample_id]))
        and math.isfinite(float(adaptive_label1_probability(pool_by_id[sample_id], adaptive_plan)))
        for sample_id in selected_ids
    )
    expected_selection = select_training_ids(list(pool_by_id.values()), method='adaptive', budget=int(selection.requested_budget), seed=0, blocked_ids=blocked_ids, adaptive_plan=adaptive_plan)
    risk_plan_valid = _risk_plan_valid(adaptive_plan)
    return {
        'property_1_risk_adaptive_distribution': {
            'risk_level': adaptive_plan.risk_level,
            'risk_strength': float(adaptive_plan.risk_strength),
            'guide_proxy_balanced_accuracy': float(adaptive_plan.guide_proxy_balanced_accuracy),
            'guide_proxy_reliability': float(adaptive_plan.guide_proxy_reliability),
            'guide_score_brier': float(adaptive_plan.guide_score_brier),
            'guide_base_brier': float(adaptive_plan.guide_base_brier),
            'guide_score_reliability': float(adaptive_plan.guide_score_reliability),
            'label_probability_reliability': float(adaptive_plan.label_probability_reliability),
            'score_calibration_slope': float(adaptive_plan.score_calibration_slope),
            'score_calibration_intercept': float(adaptive_plan.score_calibration_intercept),
            'hard_label_repair_enabled': hard_label_repair_enabled,
            'target_soft_label1_rate': float(adaptive_plan.target_yes_rate),
            'planned_target_soft_label0_mass': float(adaptive_plan.target_label0_mass),
            'planned_target_soft_label1_mass': float(adaptive_plan.target_label1_mass),
            'actual_target_soft_label0_mass': actual_target_label0_mass,
            'actual_target_soft_label1_mass': actual_target_label1_mass,
            'actual_hard_target_label1_budget': int(selection.requested_label1_budget) if hard_label_repair_enabled else None,
            'selected_soft_label1_rate': float(selected_soft_label1_mass / selected_budget) if selected_budget else 0.0,
            'selected_soft_label0_mass': selected_soft_label0_mass,
            'selected_soft_label1_mass': selected_soft_label1_mass,
            'selected_hard_proxy_label1_rate': float(selected_label1_count / selected_budget) if selected_budget else 0.0,
            'selected_label0_count': int(selected_label0_count),
            'selected_label1_count': int(selected_label1_count),
            'target_label0_budget': int(adaptive_plan.target_label0_budget),
            'target_label1_budget': int(adaptive_plan.target_label1_budget),
            'actual_label0_budget': int(selection.requested_label0_budget),
            'actual_label1_budget': int(selection.requested_label1_budget),
            'capacity_limited': bool(capacity_limited),
            'within_rounding_tolerance': bool(distribution_within_rounding),
            'feasible_budget_match': bool(feasible_budget_match),
            'passed': bool(feasible_budget_match and (distribution_within_rounding or capacity_limited)),
        },
        'property_2_refined_set_fixed_point': {
            'finite_selected_scores': bool(finite_selected_scores),
            'recomputed_selection_matches_selected': bool(selected_ids == expected_selection.selected_ids),
            'local_swap_refinement_enabled': True,
            'pool_bucket_counts': _bucket_counts(candidate_ids, routing_scores, int(adaptive_plan.bucket_count)),
            'selected_bucket_counts': _bucket_counts(selected_ids, routing_scores, int(adaptive_plan.bucket_count)),
            'selected_ids': selected_ids,
            'expected_ids': list(expected_selection.selected_ids),
            'passed': bool(finite_selected_scores and selected_ids == expected_selection.selected_ids),
        },
        'property_3_budget_and_uniqueness': _selection_integrity(selection, selected_ids, pool_by_id, blocked_ids, risk_plan_valid=risk_plan_valid),
    }


def _artifact_paths(args: argparse.Namespace, round_dir: Path, output_dir: Path) -> dict[str, Path]:
    return {
        'guide_crc_predictions': input_artifact_path(args.guide_crc_predictions_path, round_dir / 'guide_crc_predictions.jsonl'),
        'pool_crc_predictions': input_artifact_path(args.pool_crc_predictions_path, round_dir / 'pool_crc_predictions.jsonl'),
        'crc_summary': input_artifact_path(args.crc_summary_path, round_dir / 'crc_summary.json'),
        'selected_train_rows': output_artifact_path(args.selected_train_rows_output_path, round_dir / 'selected_train_rows.jsonl'),
        'cumulative_train_rows': output_artifact_path(args.cumulative_train_rows_output_path, output_dir / 'train_rows.jsonl'),
        'selection_summary': output_artifact_path(args.selection_summary_path, round_dir / 'selection_summary.json'),
        'usage': output_artifact_path(args.usage_path, round_dir / 'select_usage.json'),
    }


def _selection_plan(method: str, guide_crc_rows: list[dict[str, Any]], pool_crc_rows: list[dict[str, Any]], crc_summary: dict[str, Any], budget: int) -> tuple[Any, Any, Any]:
    kwargs = {
        'budget': int(budget),
        'temperature': float(crc_summary['temperature']),
        'lambda_hat': float(crc_summary['lambda_hat']),
        'alpha': float(crc_summary['alpha']),
    }
    if method == 'pcss':
        return compute_pcss_plan(guide_crc_rows, pool_crc_rows, **kwargs), None, None
    if method == 'crc-error-mass':
        return None, compute_crc_error_mass_plan(guide_crc_rows, pool_crc_rows, **kwargs), None
    if method == 'adaptive':
        return None, None, compute_adaptive_selection_plan(guide_crc_rows, pool_crc_rows, **kwargs)
    return None, None, None


def _selected_rows(selection: Any, pool_by_id: dict[str, dict[str, Any]], *, method: str, round_index: int) -> list[dict[str, Any]]:
    selected_label0_ids = set(selection.label0_ids)
    selected_label1_ids = set(selection.label1_ids)
    rows: list[dict[str, Any]] = []
    for sample_id in selection.selected_ids:
        row = dict(pool_by_id[sample_id])
        row['selection_round'] = int(round_index)
        row['selection_method'] = method
        row['selection_side'] = 'defer' if bool(row.get('defer', False)) else 'accept'
        if sample_id in selected_label1_ids:
            row['selection_stratum'] = 'proxy_label_1'
        elif sample_id in selected_label0_ids:
            row['selection_stratum'] = 'proxy_label_0'
        row.setdefault('label', row.get('groundtruth'))
        rows.append(row)
    return rows


def _training_property_checks(*, method: str, selection: Any, pcss_plan: Any, adaptive_plan: Any, pool_by_id: dict[str, dict[str, Any]], blocked_ids: set[str]) -> dict[str, Any] | None:
    if method == 'pcss' and pcss_plan is not None:
        return _pcss_training_property_checks(selection=selection, pcss_plan=pcss_plan, pool_by_id=pool_by_id, blocked_ids=blocked_ids)
    if method == 'adaptive' and adaptive_plan is not None:
        return _adaptive_training_property_checks(selection=selection, adaptive_plan=adaptive_plan, pool_by_id=pool_by_id, blocked_ids=blocked_ids)
    return None


def run_selection(args: argparse.Namespace, *, method: str) -> dict[str, Any]:
    output_dir = output_dir_from_arg(args.output_dir)
    round_dir = output_dir / f'round_{args.round_index}'
    stage_name = f"select_{method.replace('-', '_')}"
    paths = _artifact_paths(args, round_dir, output_dir)
    if args.show_result:
        print_existing_stage_result(stage_name=stage_name, summary_path=paths['selection_summary'])
        return {}
    cache_decision = stage_cache_decision(
        stage_name=stage_name,
        required_outputs=[paths['selected_train_rows'], paths['cumulative_train_rows'], paths['selection_summary'], paths['usage']],
        cache_policy=args.cache_policy,
    )
    if cache_decision.cache_hit:
        print_existing_stage_result(stage_name=stage_name, summary_path=paths['selection_summary'])
        return {}
    guide_crc_rows = read_jsonl(paths['guide_crc_predictions'])
    pool_crc_rows = read_jsonl(paths['pool_crc_predictions'])
    crc_summary = read_json(paths['crc_summary'])
    existing_rows = load_selected_train_rows(output_dir)
    blocked_ids = {str(row['id']) for row in existing_rows}
    pcss_plan, crc_error_mass_plan, adaptive_plan = _selection_plan(method, guide_crc_rows, pool_crc_rows, crc_summary, int(args.budget))
    selection = select_training_ids(
        pool_crc_rows,
        method=method,
        budget=int(args.budget),
        seed=int(args.seed) + int(args.round_index),
        blocked_ids=blocked_ids,
        crc_error_mass_plan=crc_error_mass_plan,
        pcss_plan=pcss_plan,
        adaptive_plan=adaptive_plan,
        accept_strategy=str(args.accept_strategy),
        defer_strategy=str(args.defer_strategy),
    )
    pool_by_id = _merge_by_id(pool_crc_rows)
    selected_rows = _selected_rows(selection, pool_by_id, method=method, round_index=int(args.round_index))
    cumulative_rows_by_id = {str(row['id']): row for row in existing_rows}
    cumulative_rows_by_id.update({str(row['id']): row for row in selected_rows})
    training_set_property_checks = _training_property_checks(method=method, selection=selection, pcss_plan=pcss_plan, adaptive_plan=adaptive_plan, pool_by_id=pool_by_id, blocked_ids=blocked_ids)
    summary = {
        'stage_name': stage_name,
        'round_index': int(args.round_index),
        'method': method,
        'budget': int(args.budget),
        'selection': selection.to_dict(),
        'pcss_plan': pcss_plan.to_dict() if pcss_plan is not None else None,
        'adaptive_plan': adaptive_plan.to_dict() if adaptive_plan is not None else None,
        'crc_error_mass_plan': crc_error_mass_plan.to_dict() if crc_error_mass_plan is not None else None,
        'training_set_property_checks': training_set_property_checks,
        'selected_train_rows_output_path': str(paths['selected_train_rows']),
        'cumulative_train_rows_output_path': str(paths['cumulative_train_rows']),
    }
    write_jsonl(selected_rows, paths['selected_train_rows'])
    write_jsonl(list(cumulative_rows_by_id.values()), paths['cumulative_train_rows'])
    write_json(summary, paths['selection_summary'])
    teacher_usage = summarize_teacher_label_usage(selected_rows, purpose='selected_training_rows')
    write_stage_usage(
        paths['usage'],
        {
            'stage_name': stage_name,
            'round_index': int(args.round_index),
            'cache': cache_decision.to_dict(),
            'selected_rows': len(selected_rows),
            'cumulative_rows': len(cumulative_rows_by_id),
            'selection_method': method,
            'teacher_label_usage': teacher_usage,
            'groundtruth_substitute_calls': teacher_usage['groundtruth_substitute_calls'],
            'teacher_api_file_calls': teacher_usage['teacher_api_file_calls'],
        },
    )
    return summary


def main(*, method: str, description: str, include_error_mass_options: bool=False) -> None:
    summary = run_selection(parse_selection_args(description=description, include_error_mass_options=include_error_mass_options), method=method)
    if summary:
        print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
