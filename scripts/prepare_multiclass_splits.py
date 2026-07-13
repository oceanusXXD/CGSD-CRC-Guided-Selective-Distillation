from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mias_dcms.multiclass_protocol import build_fixed_multiclass_splits, pool_class_prior
from mias_dcms.utils import read_jsonl, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create fixed multiclass split ids and pool prior artifacts."
    )
    parser.add_argument("--input_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--seed_size", type=int, required=True)
    parser.add_argument("--active_size", type=int, required=True)
    parser.add_argument("--test_size", type=int, required=True)
    parser.add_argument("--id_field", default="id")
    parser.add_argument("--label_field", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = read_jsonl(Path(args.input_path))
    splits = build_fixed_multiclass_splits(
        rows,
        seed=int(args.seed),
        seed_size=int(args.seed_size),
        active_size=int(args.active_size),
        test_size=int(args.test_size),
        id_field=str(args.id_field),
    )
    prior = pool_class_prior(rows, label_field=str(args.label_field))
    output_dir = Path(args.output_dir)
    write_json(splits, output_dir / "split_ids.json")
    write_json(prior.as_dict(), output_dir / "pool_prior.json")
    summary = {
        "input_path": str(args.input_path),
        "row_count": len(rows),
        "seed": int(args.seed),
        "seed_size": int(args.seed_size),
        "active_size": int(args.active_size),
        "test_size": int(args.test_size),
        "id_field": str(args.id_field),
        "label_field": str(args.label_field),
        "artifacts": {
            "split_ids": str(output_dir / "split_ids.json"),
            "pool_prior": str(output_dir / "pool_prior.json"),
        },
    }
    write_json(summary, output_dir / "summary.json")
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
