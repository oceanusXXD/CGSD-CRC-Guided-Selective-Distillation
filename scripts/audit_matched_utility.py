from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mias_dcms.composition import matched_utility_report
from mias_dcms.utils import read_jsonl, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit whether two selections are utility-matched while comparing group coverage."
    )
    parser.add_argument("--baseline_path", required=True)
    parser.add_argument("--treatment_path", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--target_moments", required=True)
    parser.add_argument("--utility_field", default="utility")
    parser.add_argument("--group_field", default="group")
    parser.add_argument("--mean_tolerance", type=float, default=0.02)
    parser.add_argument("--quantile_tolerance", type=float, default=0.05)
    parser.add_argument("--bins", type=int, default=4)
    return parser.parse_args()


def _parse_target_moments(value: str) -> dict[str, float]:
    payload = json.loads(value)
    if not isinstance(payload, dict):
        raise ValueError("target_moments must be a JSON object")
    return {str(key): float(item) for key, item in payload.items()}


def _utilities(rows: list[dict[str, Any]], utility_field: str) -> list[float]:
    return [float(row[utility_field]) for row in rows]


def _group_moments(rows: list[dict[str, Any]], group_field: str) -> dict[str, float]:
    if not rows:
        raise ValueError("rows must not be empty")
    counts = Counter(str(row[group_field]) for row in rows)
    total = float(len(rows))
    return {group: count / total for group, count in sorted(counts.items())}


def main() -> None:
    args = parse_args()
    baseline_rows = read_jsonl(Path(args.baseline_path))
    treatment_rows = read_jsonl(Path(args.treatment_path))
    target_moments = _parse_target_moments(str(args.target_moments))

    baseline_moments = _group_moments(baseline_rows, str(args.group_field))
    treatment_moments = _group_moments(treatment_rows, str(args.group_field))
    report = matched_utility_report(
        baseline_utilities=_utilities(baseline_rows, str(args.utility_field)),
        treatment_utilities=_utilities(treatment_rows, str(args.utility_field)),
        baseline_moments=baseline_moments,
        treatment_moments=treatment_moments,
        target_moments=target_moments,
        mean_tolerance=float(args.mean_tolerance),
        quantile_tolerance=float(args.quantile_tolerance),
        bins=int(args.bins),
    )

    payload = {
        "baseline_path": str(args.baseline_path),
        "treatment_path": str(args.treatment_path),
        "target_moments": target_moments,
        "baseline_moments": baseline_moments,
        "treatment_moments": treatment_moments,
        **report.as_dict(),
    }
    write_json(payload, Path(args.output_path))
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
