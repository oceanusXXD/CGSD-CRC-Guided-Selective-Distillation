from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mias_dcms.auditing import mias_selection_audit
from mias_dcms.utils import read_jsonl, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit sample-level MIAS selection metrics from a JSONL file."
    )
    parser.add_argument("--input_path", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--group_field", required=True)
    parser.add_argument("--selected_field", default="selected")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--method", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--budget", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--config_hash", required=True)
    return parser.parse_args()


def build_summary(args: argparse.Namespace, rows: list[dict[str, Any]]) -> dict[str, Any]:
    audit = mias_selection_audit(
        rows,
        group_field=str(args.group_field),
        selected_field=str(args.selected_field),
    )
    audit_payload = audit.as_dict()
    return {
        "dataset": str(args.dataset),
        "method": str(args.method),
        "model": str(args.model),
        "budget": int(args.budget),
        "seed": int(args.seed),
        "config_hash": str(args.config_hash),
        "pool_size": audit.pool_size,
        "selected_size": audit.selected_size,
        "selection_metrics": {
            "acquisition_tv": audit.acquisition_tv,
            "maximum_propensity_ratio": audit.maximum_propensity_ratio,
            "total_absolute_prediction_error": audit.total_absolute_prediction_error,
        },
        "cost_metrics": {
            "oracle_label_calls": audit.selected_size,
        },
        "groups": audit_payload["groups"],
    }


def main() -> None:
    args = parse_args()
    rows = read_jsonl(Path(args.input_path))
    summary = build_summary(args, rows)
    write_json(summary, Path(args.output_path))
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
