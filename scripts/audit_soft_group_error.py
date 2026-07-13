from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mias_dcms.soft_group_error import soft_group_error_audit
from mias_dcms.utils import read_jsonl, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare nominal and robust DCMS constraints under observed soft-group memberships."
    )
    parser.add_argument("--input_path", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--budget", type=int, required=True)
    parser.add_argument("--target_moments", required=True)
    parser.add_argument("--tolerance", type=float, required=True)
    parser.add_argument("--rounding_seed", type=int, default=None)
    parser.add_argument("--id_field", default="sample_id")
    parser.add_argument("--score_field", default="score")
    parser.add_argument("--group_field", default="groups")
    parser.add_argument("--lower_field", default="membership_lower")
    parser.add_argument("--upper_field", default="membership_upper")
    parser.add_argument("--observed_field", default="observed_membership")
    return parser.parse_args()


def _parse_target_moments(value: str) -> dict[str, float]:
    payload = json.loads(value)
    if not isinstance(payload, dict):
        raise ValueError("target_moments must be a JSON object")
    return {str(key): float(item) for key, item in payload.items()}


def _membership(rows: list[dict[str, Any]], field: str) -> list[dict[str, float]]:
    values: list[dict[str, float]] = []
    for row in rows:
        payload = row[field]
        if not isinstance(payload, dict):
            raise ValueError(f"field {field!r} must contain an object")
        values.append({str(key): float(value) for key, value in payload.items()})
    return values


def main() -> None:
    args = parse_args()
    rows = read_jsonl(Path(args.input_path))
    audit = soft_group_error_audit(
        sample_ids=[str(row[args.id_field]) for row in rows],
        utilities=[float(row[args.score_field]) for row in rows],
        group_membership=_membership(rows, str(args.group_field)),
        membership_lower=_membership(rows, str(args.lower_field)),
        membership_upper=_membership(rows, str(args.upper_field)),
        observed_membership=_membership(rows, str(args.observed_field)),
        budget=int(args.budget),
        target_moments=_parse_target_moments(str(args.target_moments)),
        tolerance=float(args.tolerance),
        rounding_seed=args.rounding_seed,
    )
    payload = {
        "input_path": str(args.input_path),
        "budget": int(args.budget),
        "target_moments": _parse_target_moments(str(args.target_moments)),
        "tolerance": float(args.tolerance),
        **audit.as_dict(),
    }
    write_json(payload, Path(args.output_path))
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
