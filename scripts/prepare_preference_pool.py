from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mias_dcms.preference_pool import build_preference_fixed_pool
from mias_dcms.utils import read_jsonl, write_json, write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create selector-safe active preference pool artifacts."
    )
    parser.add_argument("--input_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument(
        "--force_swap",
        choices=("true", "false", "random"),
        default="random",
        help="Force all A/B pairs swapped, unswapped, or use seeded random swaps.",
    )
    parser.add_argument(
        "--paired_swap",
        action="store_true",
        help="Emit both original and swapped selector-safe rows for paired position audits.",
    )
    return parser.parse_args()


def _force_swap_value(value: str) -> bool | None:
    if value == "true":
        return True
    if value == "false":
        return False
    return None


def main() -> None:
    args = parse_args()
    rows = read_jsonl(Path(args.input_path))
    fixed_pool = build_preference_fixed_pool(
        rows,
        seed=int(args.seed),
        force_swap=_force_swap_value(str(args.force_swap)),
        include_both_positions=bool(args.paired_swap),
    )
    output_dir = Path(args.output_dir)
    write_jsonl(fixed_pool.active_pool, output_dir / "active_pool.jsonl")
    write_json(fixed_pool.oracle_store, output_dir / "oracle_store.json")
    write_json(fixed_pool.swap_manifest, output_dir / "swap_manifest.json")
    summary = {
        "input_path": str(args.input_path),
        "seed": int(args.seed),
        "force_swap": str(args.force_swap),
        "paired_swap": bool(args.paired_swap),
        "active_pool_size": len(fixed_pool.active_pool),
        "oracle_store_size": len(fixed_pool.oracle_store),
        "swap_manifest_size": len(fixed_pool.swap_manifest),
        "artifacts": {
            "active_pool": str(output_dir / "active_pool.jsonl"),
            "oracle_store": str(output_dir / "oracle_store.json"),
            "swap_manifest": str(output_dir / "swap_manifest.json"),
        },
    }
    summary_path = output_dir / "pool_summary.json"
    summary["artifacts"]["pool_summary"] = str(summary_path)
    write_json(summary, summary_path)
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
