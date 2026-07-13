from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mias_dcms.selection import rank_normalize_utilities, solve_dcms_with_slack
from mias_dcms.preference_selection_metrics import (
    build_preference_selection_metrics,
    materialize_preference_group_fields,
)
from mias_dcms.utils import read_jsonl, write_json, write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run DCMS over a sample-level candidate JSONL file."
    )
    parser.add_argument("--input_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--budget", type=int, required=True)
    parser.add_argument("--target_moments", required=True)
    parser.add_argument("--slack_grid", required=True)
    parser.add_argument("--kappa", type=float, required=True)
    parser.add_argument("--rounding_seed", type=int, default=None)
    parser.add_argument("--id_field", default="sample_id")
    parser.add_argument("--score_field", default="score")
    parser.add_argument("--group_field", default="groups")
    parser.add_argument(
        "--selection_group_field",
        default="",
        help="Optional observable field that may contribute at most one candidate per selected batch.",
    )
    parser.add_argument("--use_rank_normalization", action="store_true", default=False)
    parser.add_argument(
        "--audit_group_fields",
        default="",
        help="Optional comma-separated categorical fields to expose in preference selection metrics.",
    )
    return parser.parse_args()


def _parse_target_moments(value: str, *, rows: list[dict[str, Any]], group_field: str) -> dict[str, float]:
    if str(value).strip().lower() == "pool":
        memberships = _group_membership(rows, group_field)
        groups = sorted({group for membership in memberships for group in membership})
        row_count = len(memberships)
        if row_count == 0:
            raise ValueError("cannot derive pool target moments from an empty candidate set")
        return {
            group: sum(float(membership.get(group, 0.0)) for membership in memberships) / row_count
            for group in groups
        }
    payload = json.loads(value)
    if not isinstance(payload, dict):
        raise ValueError("target_moments must be a JSON object or 'pool'")
    return {str(key): float(item) for key, item in payload.items()}


def _parse_slack_grid(value: str) -> list[float]:
    slacks = [float(item.strip()) for item in value.split(",") if item.strip()]
    if not slacks:
        raise ValueError("slack_grid must contain at least one value")
    return slacks


def _group_membership(rows: list[dict[str, Any]], group_field: str) -> list[dict[str, float]]:
    memberships: list[dict[str, float]] = []
    for row in rows:
        groups = row[group_field]
        if not isinstance(groups, dict):
            raise ValueError(f"group field {group_field!r} must contain an object")
        memberships.append({str(key): float(value) for key, value in groups.items()})
    return memberships


def main() -> None:
    args = parse_args()
    rows = read_jsonl(Path(args.input_path))
    if not rows:
        raise ValueError("candidate rows must not be empty")
    selection_group_field = str(args.selection_group_field).strip()
    candidate_rows, excluded_candidate_ids = _collapse_selection_groups(
        rows,
        id_field=str(args.id_field),
        score_field=str(args.score_field),
        selection_group_field=selection_group_field,
    )
    sample_ids = [str(row[args.id_field]) for row in candidate_rows]
    raw_scores = [float(row[args.score_field]) for row in candidate_rows]
    utilities = rank_normalize_utilities(raw_scores) if args.use_rank_normalization else raw_scores
    result = solve_dcms_with_slack(
        sample_ids=sample_ids,
        utilities=utilities,
        group_membership=_group_membership(candidate_rows, str(args.group_field)),
        budget=int(args.budget),
        target_moments=_parse_target_moments(
            str(args.target_moments),
            rows=rows,
            group_field=str(args.group_field),
        ),
        slack_grid=_parse_slack_grid(str(args.slack_grid)),
        kappa=float(args.kappa),
        rounding_seed=args.rounding_seed,
    )

    output_dir = Path(args.output_dir)
    selected_payload = {
        "selected_ids": list(result.selected_ids),
        "budget": int(args.budget),
        "selected_count": len(result.selected_ids),
    }
    audit_group_fields = tuple(
        field.strip() for field in str(args.audit_group_fields).split(",") if field.strip()
    )
    candidate_values = {
        sample_id: (raw_score, utility)
        for sample_id, raw_score, utility in zip(sample_ids, raw_scores, utilities, strict=True)
    }
    propensity_rows = [
        {
            **materialize_preference_group_fields(rows[index], group_fields=audit_group_fields),
            "sample_id": sample_id,
            "base_score": raw_score,
            "utility": utility,
            "q_propensity": float(result.q_propensity.get(sample_id, 0.0)),
            "selected": int(result.selection_indicator.get(sample_id, 0)),
            **_audit_group_values(rows[index], audit_group_fields),
        }
        for index, row in enumerate(rows)
        for sample_id, raw_score, utility in [
            (
                str(row[args.id_field]),
                float(row[args.score_field]),
                candidate_values.get(str(row[args.id_field]), (float(row[args.score_field]), 0.0))[1],
            )
        ]
    ]
    summary = {
        "input_path": str(args.input_path),
        "budget": int(args.budget),
        "pool_size": len(rows),
        "selection_candidate_count": len(candidate_rows),
        "selection_group_field": selection_group_field or None,
        "excluded_same_group_candidate_count": len(excluded_candidate_ids),
        "selected_count": len(result.selected_ids),
        "selected_slack": result.selected_slack,
        "utility_retained": result.utility_retained,
        "max_constraint_violation": result.max_constraint_violation,
        "continuous_moments": dict(result.continuous_moments),
        "rounded_moments": dict(result.rounded_moments),
        "robust_lower_moments": dict(result.robust_lower_moments),
        "robust_upper_moments": dict(result.robust_upper_moments),
        "solver_status": result.solver_status,
        "rounding_seed": result.rounding_seed,
        "slack_trace": [
            {
                "slack": trace.slack,
                "feasible": trace.feasible,
                "utility_retained": trace.utility_retained,
                "max_constraint_violation": trace.max_constraint_violation,
                "expected_moments": dict(trace.expected_moments),
                "meets_utility_threshold": trace.meets_utility_threshold,
                "solver_status": trace.solver_status,
            }
            for trace in result.slack_trace
        ],
    }
    if audit_group_fields:
        summary["selection_metrics"] = build_preference_selection_metrics(
            propensity_rows,
            method="dcms",
            group_fields=audit_group_fields,
            constraint_violation=float(result.max_constraint_violation),
            utility_retained=float(result.utility_retained),
        )

    write_json(selected_payload, output_dir / "selected_ids.json")
    write_jsonl(propensity_rows, output_dir / "propensity.jsonl")
    write_json(summary, output_dir / "selection_summary.json")
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))


def _audit_group_values(row: dict[str, Any], fields: tuple[str, ...]) -> dict[str, str]:
    materialized = materialize_preference_group_fields(row, group_fields=fields)
    return {
        field: str(materialized[field])
        for field in fields
        if field in materialized and materialized[field] is not None
    }


def _collapse_selection_groups(
    rows: list[dict[str, Any]],
    *,
    id_field: str,
    score_field: str,
    selection_group_field: str,
) -> tuple[list[dict[str, Any]], set[str]]:
    if not selection_group_field:
        return [dict(row) for row in rows], set()
    representatives: dict[str, dict[str, Any]] = {}
    excluded: set[str] = set()
    for row in rows:
        payload = dict(row)
        sample_id = str(payload[id_field])
        group_id = str(payload.get(selection_group_field, ""))
        if not group_id:
            raise ValueError(f"rows are missing selection group field {selection_group_field!r}")
        current = representatives.get(group_id)
        if current is None:
            representatives[group_id] = payload
            continue
        current_key = (-float(current[score_field]), str(current[id_field]))
        candidate_key = (-float(payload[score_field]), sample_id)
        if candidate_key < current_key:
            excluded.add(str(current[id_field]))
            representatives[group_id] = payload
        else:
            excluded.add(sample_id)
    return sorted(representatives.values(), key=lambda row: str(row[id_field])), excluded


if __name__ == "__main__":
    main()
