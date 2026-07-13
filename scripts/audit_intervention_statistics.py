from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mias_dcms.intervention_statistics import audit_intervention_response_statistics
from mias_dcms.utils import read_jsonl, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit intervention response curves for monotonicity, slope CI, and hidden failed settings."
    )
    parser.add_argument("--input_path", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--expected_settings", required=True)
    parser.add_argument("--minimum_values", type=int, default=5)
    parser.add_argument("--setting_field", default="setting")
    parser.add_argument("--status_field", default="status")
    parser.add_argument("--failure_reason_field", default="failure_reason")
    parser.add_argument("--intervention_value_field", default="intervention_value")
    parser.add_argument("--response_field", default="target_group_propensity")
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument("--resamples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = read_jsonl(Path(args.input_path))
    report = audit_intervention_response_statistics(
        rows,
        expected_settings=_parse_csv(args.expected_settings),
        minimum_values=int(args.minimum_values),
        setting_field=str(args.setting_field),
        status_field=str(args.status_field),
        failure_reason_field=str(args.failure_reason_field),
        intervention_value_field=str(args.intervention_value_field),
        response_field=str(args.response_field),
        confidence=float(args.confidence),
        resamples=int(args.resamples),
        seed=int(args.seed),
    )
    payload = report.as_dict()
    payload["input_path"] = str(args.input_path)
    write_json(payload, Path(args.output_path))
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    raise SystemExit(0 if payload["is_ready"] else 1)


def _parse_csv(value: str) -> list[str]:
    return [item.strip() for item in str(value).split(",") if item.strip()]


if __name__ == "__main__":
    main()
