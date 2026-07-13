from __future__ import annotations

import argparse
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mias_dcms.preference_acquisition_audit import audit_preference_acquisition
from mias_dcms.utils import read_jsonl, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit first-round preference acquisition coverage across observable attributes."
    )
    parser.add_argument("--input_path", type=Path, required=True)
    parser.add_argument("--output_path", type=Path, required=True)
    parser.add_argument("--method", required=True)
    parser.add_argument("--group_fields", required=True)
    parser.add_argument("--selected_field", default="selected")
    parser.add_argument("--random_reference_path", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    group_fields = tuple(field.strip() for field in str(args.group_fields).split(",") if field.strip())
    rows = read_jsonl(args.input_path)
    random_reference_rows = (
        read_jsonl(args.random_reference_path)
        if args.random_reference_path is not None
        else None
    )
    summary = audit_preference_acquisition(
        rows,
        method=str(args.method),
        group_fields=group_fields,
        selected_field=str(args.selected_field),
        random_reference_rows=random_reference_rows,
    )
    write_json(
        {
            "input_path": str(args.input_path),
            "random_reference_path": str(args.random_reference_path)
            if args.random_reference_path is not None
            else None,
            **summary,
        },
        args.output_path,
    )


if __name__ == "__main__":
    main()
