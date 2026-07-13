from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mias_dcms.dpo_run_pack import (
    DPO_MAIN_METHODS,
    validate_dpo_run_pack,
    validate_paper_artifact_manifest,
)
from mias_dcms.utils import read_json, read_jsonl, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate DPO main-result run records and optional paper artifact manifest."
    )
    parser.add_argument("--run_records_path", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--datasets", required=True)
    parser.add_argument("--models", required=True)
    parser.add_argument("--budgets", required=True)
    parser.add_argument("--seeds", required=True)
    parser.add_argument("--methods", default=",".join(DPO_MAIN_METHODS))
    parser.add_argument("--paper_manifest_path")
    parser.add_argument("--expected_figures", default="fig1,fig2,fig3")
    parser.add_argument("--expected_tables", default="table1,table2,table3")
    parser.add_argument("--expected_seed_count", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = read_jsonl(Path(args.run_records_path))
    report = validate_dpo_run_pack(
        rows,
        expected_datasets=_parse_csv(args.datasets),
        expected_models=_parse_csv(args.models),
        expected_budgets=[int(value) for value in _parse_csv(args.budgets)],
        expected_seeds=[int(value) for value in _parse_csv(args.seeds)],
        required_methods=_parse_csv(args.methods),
    )
    payload = report.as_dict()
    payload["run_records_path"] = str(args.run_records_path)

    if args.paper_manifest_path:
        manifest = read_json(Path(args.paper_manifest_path))
        manifest_issues = validate_paper_artifact_manifest(
            manifest,
            expected_figures=_parse_csv(args.expected_figures),
            expected_tables=_parse_csv(args.expected_tables),
            expected_seed_count=args.expected_seed_count,
        )
        payload["paper_manifest_path"] = str(args.paper_manifest_path)
        payload["paper_manifest_issues"] = manifest_issues
        payload["issues"] = [*payload["issues"], *manifest_issues]
        payload["is_ready"] = not payload["issues"]

    write_json(payload, Path(args.output_path))
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    raise SystemExit(0 if payload["is_ready"] else 1)


def _parse_csv(value: str) -> list[str]:
    return [item.strip() for item in str(value).split(",") if item.strip()]


if __name__ == "__main__":
    main()
