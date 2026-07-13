from __future__ import annotations

import argparse
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mias_dcms.preference_selector_audit import audit_preference_selector_scores
from mias_dcms.utils import read_jsonl, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit selector score sanity checks for preference baselines."
    )
    parser.add_argument("--input_path", type=Path, required=True)
    parser.add_argument("--output_path", type=Path, required=True)
    parser.add_argument("--method", required=True)
    parser.add_argument("--budget", type=int, required=True)
    parser.add_argument("--id_field", default="sample_id")
    parser.add_argument("--score_field")
    parser.add_argument("--length_field", default="length_gap")
    parser.add_argument("--swap_pair_field", default="swap_pair_id")
    parser.add_argument("--position_field", default="ab_position")
    parser.add_argument("--selector_compute_seconds", type=float, default=0.0)
    parser.add_argument("--allow_degenerate_scores", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = read_jsonl(args.input_path)
    summary = audit_preference_selector_scores(
        rows,
        method=str(args.method),
        budget=int(args.budget),
        id_field=str(args.id_field),
        score_field=str(args.score_field) if args.score_field else None,
        length_field=str(args.length_field),
        swap_pair_field=str(args.swap_pair_field),
        position_field=str(args.position_field),
        selector_compute_seconds=float(args.selector_compute_seconds),
        require_non_degenerate=not bool(args.allow_degenerate_scores),
    )
    write_json({"input_path": str(args.input_path), **summary}, args.output_path)


if __name__ == "__main__":
    main()
