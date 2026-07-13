from __future__ import annotations

import argparse
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mias_dcms.preference_dcms_inputs import build_preference_dcms_candidate_rows
from mias_dcms.utils import read_jsonl, write_json, write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert selector-safe preference baseline scores into DCMS candidate rows."
    )
    parser.add_argument("--input_path", type=Path, required=True)
    parser.add_argument("--output_path", type=Path, required=True)
    parser.add_argument("--summary_path", type=Path)
    parser.add_argument("--method", required=True)
    parser.add_argument("--group_fields", default="")
    parser.add_argument("--group_field")
    parser.add_argument("--id_field", default="sample_id")
    parser.add_argument("--score_field")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    group_fields = tuple(field.strip() for field in str(args.group_fields).split(",") if field.strip())
    rows = read_jsonl(args.input_path)
    candidates = build_preference_dcms_candidate_rows(
        rows,
        method=str(args.method),
        group_fields=group_fields,
        group_field=args.group_field,
        id_field=str(args.id_field),
        score_field=args.score_field,
    )
    write_jsonl(candidates, args.output_path)

    summary_path = args.summary_path or args.output_path.with_suffix(".summary.json")
    score_field = str(args.score_field or f"{str(args.method).strip().lower().replace('-', '_')}_score")
    write_json(
        {
            "input_path": str(args.input_path),
            "output_path": str(args.output_path),
            "candidate_count": len(candidates),
            "method": str(args.method).strip().lower().replace("-", "_"),
            "score_field": score_field,
            "group_field": args.group_field,
            "group_fields": list(group_fields),
        },
        summary_path,
    )


if __name__ == "__main__":
    main()
