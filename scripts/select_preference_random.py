from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mias_dcms.selectors import assert_selector_rows_are_label_safe, random_without_replacement
from mias_dcms.utils import read_jsonl, write_json, write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Select a random top-budget preference batch from a selector-safe active pool."
    )
    parser.add_argument("--input_path", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--budget", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--id_field", default="sample_id")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = read_jsonl(args.input_path)
    assert_selector_rows_are_label_safe(rows)
    sample_ids = [str(row[args.id_field]) for row in rows]
    selected_ids = random_without_replacement(
        sample_ids,
        budget=int(args.budget),
        seed=int(args.seed),
    )
    selected_id_set = set(selected_ids)
    membership_rows = [
        {
            str(args.id_field): sample_id,
            "sample_id": sample_id,
            "method": "random",
            "selected": int(sample_id in selected_id_set),
        }
        for sample_id in sample_ids
    ]
    summary = {
        "input_path": str(args.input_path),
        "method": "random",
        "budget": int(args.budget),
        "seed": int(args.seed),
        "pool_size": len(rows),
        "selected_count": len(selected_ids),
    }
    selected_payload = {
        "selected_ids": selected_ids,
        "budget": int(args.budget),
        "selected_count": len(selected_ids),
        "method": "random",
        "seed": int(args.seed),
    }

    output_dir = args.output_dir
    write_json(selected_payload, output_dir / "selected_ids.json")
    write_jsonl(membership_rows, output_dir / "membership.jsonl")
    write_json(summary, output_dir / "selection_summary.json")
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
