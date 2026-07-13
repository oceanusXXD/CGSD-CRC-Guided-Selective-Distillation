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
    parser.add_argument("--use_rank_normalization", action="store_true", default=False)
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
    sample_ids = [str(row[args.id_field]) for row in rows]
    raw_scores = [float(row[args.score_field]) for row in rows]
    utilities = rank_normalize_utilities(raw_scores) if args.use_rank_normalization else raw_scores
    result = solve_dcms_with_slack(
        sample_ids=sample_ids,
        utilities=utilities,
        group_membership=_group_membership(rows, str(args.group_field)),
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
    propensity_rows = [
        {
            "sample_id": sample_id,
            "base_score": raw_score,
            "utility": utility,
            "q_propensity": float(result.q_propensity[sample_id]),
            "selected": int(result.selection_indicator[sample_id]),
        }
        for sample_id, raw_score, utility in zip(sample_ids, raw_scores, utilities)
    ]
    summary = {
        "input_path": str(args.input_path),
        "budget": int(args.budget),
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

    write_json(selected_payload, output_dir / "selected_ids.json")
    write_jsonl(propensity_rows, output_dir / "propensity.jsonl")
    write_json(summary, output_dir / "selection_summary.json")
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
