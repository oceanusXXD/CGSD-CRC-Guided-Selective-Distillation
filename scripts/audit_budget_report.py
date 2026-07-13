from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mias_dcms.budgeting import BudgetInputs, build_budget_report, compare_budget_reports
from mias_dcms.utils import read_jsonl, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit fair supervision budget, evaluation resources, and compute costs."
    )
    parser.add_argument("--input_path", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--train_token_tolerance", type=int, default=0)
    return parser.parse_args()


def _budget_inputs(row: dict[str, Any]) -> BudgetInputs:
    return BudgetInputs(
        method=str(row["method"]),
        seed_label_count=int(row.get("seed_label_count", 0)),
        active_label_count=int(row.get("active_label_count", 0)),
        guide_label_count=int(row.get("guide_label_count", 0)),
        calibration_label_count=int(row.get("calibration_label_count", 0)),
        group_estimator_label_count=int(row.get("group_estimator_label_count", 0)),
        evaluation_label_count=int(row.get("evaluation_label_count", 0)),
        certification_label_count=int(row.get("certification_label_count", 0)),
        judge_calls=int(row.get("judge_calls", 0)),
        train_tokens=int(row.get("train_tokens", 0)),
        selector_compute_seconds=float(row.get("selector_compute_seconds", 0.0)),
    )


def main() -> None:
    args = parse_args()
    reports = [build_budget_report(_budget_inputs(row)) for row in read_jsonl(Path(args.input_path))]
    comparison = compare_budget_reports(
        reports,
        train_token_tolerance=int(args.train_token_tolerance),
    )
    payload = {
        "input_path": str(args.input_path),
        "method_count": len(reports),
        "reports": [report.as_dict() for report in reports],
        "comparison": comparison,
    }
    write_json(payload, Path(args.output_path))
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
