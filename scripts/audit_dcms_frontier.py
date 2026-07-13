from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mias_dcms.selection import dcms_utility_coverage_frontier, rank_normalize_utilities
from mias_dcms.utils import read_jsonl, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit DCMS utility-retention versus coverage-deviation frontier over a slack grid."
    )
    parser.add_argument("--input_path", required=True)
    parser.add_argument("--output_path", required=True)
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


def _parse_target_moments(value: str) -> dict[str, float]:
    payload = json.loads(value)
    if not isinstance(payload, dict):
        raise ValueError("target_moments must be a JSON object")
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
    frontier = dcms_utility_coverage_frontier(
        sample_ids=sample_ids,
        utilities=utilities,
        group_membership=_group_membership(rows, str(args.group_field)),
        budget=int(args.budget),
        target_moments=_parse_target_moments(str(args.target_moments)),
        slack_grid=_parse_slack_grid(str(args.slack_grid)),
        kappa=float(args.kappa),
        rounding_seed=args.rounding_seed,
    )
    payload = {
        "input_path": str(args.input_path),
        "budget": int(args.budget),
        "score_field": str(args.score_field),
        "group_field": str(args.group_field),
        "use_rank_normalization": bool(args.use_rank_normalization),
        **frontier.as_dict(),
    }
    write_json(payload, Path(args.output_path))
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
