from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mias_dcms.preference_split_manifest import build_preference_split_manifest
from mias_dcms.utils import read_jsonl, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create fixed seed/active/heldout/test split manifest for preference pools."
    )
    parser.add_argument("--input_path", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--seed_size", type=int, required=True)
    parser.add_argument("--active_size", type=int, required=True)
    parser.add_argument("--heldout_size", type=int, required=True)
    parser.add_argument("--test_size", type=int, required=True)
    parser.add_argument("--id_field", default="sample_id")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = read_jsonl(args.input_path)
    manifest = build_preference_split_manifest(
        rows,
        seed=int(args.seed),
        seed_size=int(args.seed_size),
        active_size=int(args.active_size),
        heldout_size=int(args.heldout_size),
        test_size=int(args.test_size),
        id_field=str(args.id_field),
    )
    output_dir = Path(args.output_dir)
    manifest_path = output_dir / "split_manifest.json"
    summary_path = output_dir / "split_summary.json"
    write_json(manifest, manifest_path)
    summary = {
        "input_path": str(args.input_path),
        "row_count": len(rows),
        "seed": int(args.seed),
        "seed_size": int(args.seed_size),
        "active_size": int(args.active_size),
        "heldout_size": int(args.heldout_size),
        "test_size": int(args.test_size),
        "id_field": str(args.id_field),
        "artifacts": {
            "split_manifest": str(manifest_path),
            "split_summary": str(summary_path),
        },
    }
    write_json(summary, summary_path)
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
