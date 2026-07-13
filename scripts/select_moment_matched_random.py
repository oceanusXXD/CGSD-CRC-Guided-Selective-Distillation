from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mias_dcms.selectors import moment_matched_random
from mias_dcms.utils import read_jsonl, write_json, write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Select a random batch constrained to match pre-declared group moments."
    )
    parser.add_argument("--input_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--budget", type=int, required=True)
    parser.add_argument("--target_moments", required=True)
    parser.add_argument("--tolerance", type=float, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--id_field", default="sample_id")
    parser.add_argument("--group_field", default="groups")
    return parser.parse_args()


def _parse_target_moments(value: str) -> dict[str, float]:
    payload = json.loads(value)
    if not isinstance(payload, dict):
        raise ValueError("target_moments must be a JSON object")
    return {str(key): float(item) for key, item in payload.items()}


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
    result = moment_matched_random(
        sample_ids=sample_ids,
        group_membership=_group_membership(rows, str(args.group_field)),
        budget=int(args.budget),
        target_moments=_parse_target_moments(str(args.target_moments)),
        tolerance=float(args.tolerance),
        seed=int(args.seed),
    )

    output_dir = Path(args.output_dir)
    selected_payload = {
        "selected_ids": list(result.selected_ids),
        "budget": int(result.budget),
        "selected_count": len(result.selected_ids),
    }
    membership_rows = [
        {
            "sample_id": sample_id,
            "selected": int(result.selection_indicator[sample_id]),
            "groups": row[str(args.group_field)],
        }
        for sample_id, row in zip(sample_ids, rows)
    ]
    summary = {
        "method": "moment_matched_random",
        "input_path": str(args.input_path),
        "budget": int(result.budget),
        "selected_count": len(result.selected_ids),
        "seed": int(result.seed),
        "target_moments": dict(result.target_moments),
        "rounded_moments": dict(result.rounded_moments),
        "max_constraint_violation": float(result.max_constraint_violation),
        "solver_status": result.solver_status,
    }

    write_json(selected_payload, output_dir / "selected_ids.json")
    write_jsonl(membership_rows, output_dir / "membership.jsonl")
    write_json(summary, output_dir / "selection_summary.json")
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
