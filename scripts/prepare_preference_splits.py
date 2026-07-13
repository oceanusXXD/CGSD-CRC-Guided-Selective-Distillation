from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mias_dcms.preference_split_manifest import (
    build_preference_split_manifest,
    materialize_preference_split_oracle_store,
    materialize_preference_split_rows,
)
from mias_dcms.utils import read_json, read_jsonl, write_json, write_jsonl


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
    parser.add_argument("--prompt_field", default="prompt")
    parser.add_argument("--group_field", default="swap_pair_id")
    parser.add_argument("--allow_prompt_overlap", action="store_true")
    parser.add_argument(
        "--oracle_store_path",
        type=Path,
        help="Optional oracle store JSON. When present, write split-specific oracle stores.",
    )
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
        prompt_field=str(args.prompt_field),
        enforce_prompt_disjoint=not bool(args.allow_prompt_overlap),
        group_field=str(args.group_field),
    )
    output_dir = Path(args.output_dir)
    manifest_path = output_dir / "split_manifest.json"
    summary_path = output_dir / "split_summary.json"
    write_json(manifest, manifest_path)
    split_pool_paths: dict[str, str] = {}
    for split_name in ("seed", "selection", "heldout", "test", "unused"):
        split_rows = materialize_preference_split_rows(
            rows,
            manifest,
            split=split_name,
            id_field=str(args.id_field),
        )
        split_path = output_dir / f"{split_name}_pool.jsonl"
        write_jsonl(split_rows, split_path)
        split_pool_paths[split_name] = str(split_path)

    seed_selected_ids_path = output_dir / "seed_selected_ids.json"
    write_json(
        {
            "selected_ids": list(manifest["seed_ids"]),
            "selected_count": len(manifest["seed_ids"]),
            "budget": len(manifest["seed_ids"]),
            "method": "initial_seed",
            "seed": int(args.seed),
        },
        seed_selected_ids_path,
    )

    split_oracle_paths: dict[str, str] = {}
    if args.oracle_store_path is not None:
        oracle_store = read_json(args.oracle_store_path)
        if not all(isinstance(value, dict) for value in oracle_store.values()):
            raise ValueError("oracle store JSON must be an object keyed by sample id")
        for split_name in ("seed", "selection", "heldout", "test", "unused"):
            split_oracle_store = materialize_preference_split_oracle_store(
                oracle_store,
                manifest,
                split=split_name,
            )
            split_oracle_path = output_dir / f"{split_name}_oracle_store.json"
            write_json(split_oracle_store, split_oracle_path)
            split_oracle_paths[split_name] = str(split_oracle_path)
    summary = {
        "input_path": str(args.input_path),
        "row_count": len(rows),
        "seed": int(args.seed),
        "seed_size": int(args.seed_size),
        "active_size": int(args.active_size),
        "heldout_size": int(args.heldout_size),
        "test_size": int(args.test_size),
        "id_field": str(args.id_field),
        "prompt_field": str(args.prompt_field),
        "group_field": str(args.group_field),
        "prompt_disjoint_enforced": not bool(args.allow_prompt_overlap),
        "artifacts": {
            "split_manifest": str(manifest_path),
            "split_summary": str(summary_path),
            "seed_selected_ids": str(seed_selected_ids_path),
            "split_pools": split_pool_paths,
            "split_oracle_stores": split_oracle_paths,
        },
    }
    write_json(summary, summary_path)
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
