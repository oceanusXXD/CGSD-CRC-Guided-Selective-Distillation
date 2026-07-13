from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mias_dcms.soft_groups import (
    build_soft_group_intervals_from_rows,
    interval_coverage_report,
    soft_group_calibration_report,
)
from mias_dcms.utils import read_jsonl, write_json, write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build soft group mean membership and robust lower/upper intervals from ensemble draws."
    )
    parser.add_argument("--input_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--id_field", default="sample_id")
    parser.add_argument("--draws_field", default="ensemble_memberships")
    parser.add_argument("--calibration_path", default=None)
    parser.add_argument("--observed_field", default="observed_membership")
    parser.add_argument("--confidence", type=float, default=0.95)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = read_jsonl(Path(args.input_path))
    report = build_soft_group_intervals_from_rows(
        rows,
        id_field=str(args.id_field),
        draws_field=str(args.draws_field),
        confidence=float(args.confidence),
    )
    output_dir = Path(args.output_dir)
    write_jsonl([row.as_dict() for row in report.rows], output_dir / "soft_group_membership.jsonl")
    summary = {
        "input_path": str(args.input_path),
        "id_field": str(args.id_field),
        "draws_field": str(args.draws_field),
        "groups": list(report.groups),
        "sample_count": int(report.sample_count),
        "confidence": float(report.confidence),
    }
    if args.calibration_path:
        calibration_rows = read_jsonl(Path(args.calibration_path))
        observed_by_id = {
            str(row[str(args.id_field)]): row[str(args.observed_field)]
            for row in calibration_rows
        }
        observed_memberships = [
            observed_by_id[row.sample_id]
            for row in report.rows
        ]
        membership_calibration = soft_group_calibration_report(
            predicted_memberships=[row.group_membership for row in report.rows],
            observed_memberships=observed_memberships,
        )
        interval_coverage = interval_coverage_report(
            report.rows,
            observed_memberships=observed_memberships,
        )
        calibration_payload = {
            "calibration_path": str(args.calibration_path),
            "observed_field": str(args.observed_field),
            "membership_calibration": membership_calibration.as_dict(),
            "interval_coverage": interval_coverage.as_dict(),
        }
        write_json(calibration_payload, output_dir / "calibration_summary.json")
        summary["calibration_summary_path"] = str(output_dir / "calibration_summary.json")
    write_json(summary, output_dir / "summary.json")
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
