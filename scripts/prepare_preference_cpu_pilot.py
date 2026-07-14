from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


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
        description="Build a small, label-isolated HelpSteer2 preference pilot suitable for CPU DPO."
    )
    parser.add_argument("--input_pool_path", type=Path, required=True)
    parser.add_argument("--oracle_store_path", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--seed_size", type=int, required=True)
    parser.add_argument("--selection_size", type=int, required=True)
    parser.add_argument("--heldout_size", type=int, required=True)
    parser.add_argument("--test_size", type=int, required=True)
    parser.add_argument("--max_response_word_count", type=int, required=True)
    parser.add_argument("--id_field", default="sample_id")
    parser.add_argument("--prompt_field", default="prompt")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_rows = read_jsonl(args.input_pool_path)
    filtered_rows = filter_rows_for_cpu_pilot(
        source_rows,
        max_response_word_count=int(args.max_response_word_count),
    )
    requested_count = sum(
        int(value)
        for value in (args.seed_size, args.selection_size, args.heldout_size, args.test_size)
    )
    if len(filtered_rows) < requested_count:
        raise ValueError(
            f"only {len(filtered_rows)} selector-safe rows meet the CPU length limit, "
            f"but {requested_count} are required"
        )

    manifest = build_preference_split_manifest(
        filtered_rows,
        seed=int(args.seed),
        seed_size=int(args.seed_size),
        active_size=int(args.selection_size),
        heldout_size=int(args.heldout_size),
        test_size=int(args.test_size),
        id_field=str(args.id_field),
        prompt_field=str(args.prompt_field),
        enforce_prompt_disjoint=True,
        group_field="",
    )
    oracle_store = read_json(args.oracle_store_path)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    split_pool_paths: dict[str, str] = {}
    split_oracle_paths: dict[str, str] = {}
    for split_name in ("seed", "selection", "heldout", "test"):
        pool_path = output_dir / f"{split_name}_pool.jsonl"
        oracle_path = output_dir / f"{split_name}_oracle_store.json"
        write_jsonl(
            materialize_preference_split_rows(
                filtered_rows,
                manifest,
                split=split_name,
                id_field=str(args.id_field),
            ),
            pool_path,
        )
        write_json(
            materialize_preference_split_oracle_store(
                oracle_store,
                manifest,
                split=split_name,
            ),
            oracle_path,
        )
        split_pool_paths[split_name] = str(pool_path)
        split_oracle_paths[split_name] = str(oracle_path)

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
    manifest_path = output_dir / "split_manifest.json"
    write_json(manifest, manifest_path)
    summary = {
        "input_pool_path": str(args.input_pool_path),
        "oracle_store_path": str(args.oracle_store_path),
        "source_row_count": len(source_rows),
        "candidate_row_count": len(filtered_rows),
        "max_response_word_count": int(args.max_response_word_count),
        "seed": int(args.seed),
        "split_sizes": {
            "seed": int(args.seed_size),
            "selection": int(args.selection_size),
            "heldout": int(args.heldout_size),
            "test": int(args.test_size),
        },
        "artifacts": {
            "split_manifest": str(manifest_path),
            "seed_selected_ids": str(seed_selected_ids_path),
            "split_pools": split_pool_paths,
            "split_oracle_stores": split_oracle_paths,
        },
    }
    write_json(summary, output_dir / "summary.json")
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))


def filter_rows_for_cpu_pilot(
    rows: list[dict[str, Any]],
    *,
    max_response_word_count: int,
) -> list[dict[str, Any]]:
    if max_response_word_count <= 0:
        raise ValueError("max_response_word_count must be positive")
    filtered: list[dict[str, Any]] = []
    for row in rows:
        response_a_words = _response_word_count(row, response="a")
        response_b_words = _response_word_count(row, response="b")
        if max(response_a_words, response_b_words) <= max_response_word_count:
            filtered.append(dict(row))
    return filtered


def _response_word_count(row: dict[str, Any], *, response: str) -> int:
    cached_field = f"response_{response}_word_count"
    cached_value = row.get(cached_field)
    if cached_value is not None:
        return int(cached_value)
    return len(str(row[f"response_{response}"]).split())


if __name__ == "__main__":
    main()
