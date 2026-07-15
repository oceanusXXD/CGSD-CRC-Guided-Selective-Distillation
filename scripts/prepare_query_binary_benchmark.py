from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mias_dcms.binary_benchmark_protocol import prepare_binary_benchmark_protocol  # noqa: E402
from mias_dcms.binary_reaudit import sha256_file  # noqa: E402
from mias_dcms.query_binary_benchmark import prepare_query_binary_source  # noqa: E402
from mias_dcms.utils import read_jsonl, write_json, write_jsonl  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare a leakage-safe MIAS binary protocol from a local single-query JSONL dataset."
    )
    parser.add_argument("--input_path", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--source_name", required=True)
    parser.add_argument("--source_query_id")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--source_test_size", type=int, required=True)
    parser.add_argument("--seed_label_count", type=int, required=True)
    parser.add_argument("--active_pool_size", type=int, required=True)
    parser.add_argument("--development_size", type=int, required=True)
    parser.add_argument("--train_row_limit", type=int)
    parser.add_argument("--test_row_limit", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    prepared = prepare_query_binary_source(
        read_jsonl(args.input_path),
        dataset=str(args.dataset),
        seed=int(args.seed),
        test_size=int(args.source_test_size),
        expected_query_id=args.source_query_id,
    )
    source_train_path = output_dir / "source_train.jsonl"
    source_test_path = output_dir / "source_test.jsonl"
    write_jsonl(prepared["source_train_rows"], source_train_path)
    write_jsonl(prepared["source_test_rows"], source_test_path)

    source_manifest = _source_manifest(args, prepared["source_summary"], source_train_path, source_test_path)
    write_json(source_manifest, output_dir / "source_manifest.json")

    artifacts = prepare_binary_benchmark_protocol(
        prepared["source_train_rows"],
        validation_rows=None,
        test_rows=prepared["source_test_rows"],
        dataset=str(args.dataset),
        seed_label_count=int(args.seed_label_count),
        active_pool_size=int(args.active_pool_size),
        development_size=int(args.development_size),
        train_row_limit=args.train_row_limit,
        test_row_limit=args.test_row_limit,
        seed=int(args.seed),
    )
    paths = {
        "seed_train_rows": output_dir / "seed_train_rows.jsonl",
        "selection_pool": output_dir / "selection_pool.jsonl",
        "selection_oracle_store": output_dir / "selection_oracle_store.json",
        "development_rows": output_dir / "development_rows.jsonl",
        "fixed_test_rows": output_dir / "fixed_test_rows.jsonl",
    }
    write_jsonl(artifacts["seed_train_rows"], paths["seed_train_rows"])
    write_jsonl(artifacts["selection_pool"], paths["selection_pool"])
    write_json(artifacts["selection_oracle_store"], paths["selection_oracle_store"])
    write_jsonl(artifacts["development_rows"], paths["development_rows"])
    write_jsonl(artifacts["official_test_rows"], paths["fixed_test_rows"])

    manifest = dict(artifacts["protocol_manifest"])
    manifest["source_manifest_path"] = str(output_dir / "source_manifest.json")
    manifest["test_split_origin"] = "deterministic_document_disjoint_source_holdout"
    manifest["test_split_policy"] = "fixed_before_selection_and_never_used_for_selection_or_checkpoint_choice"
    manifest.pop("official_test_policy", None)
    manifest["artifacts"] = {name: str(path) for name, path in paths.items()}
    write_json(manifest, output_dir / "protocol_manifest.json")
    compact_source_summary = dict(prepared["source_summary"])
    compact_source_summary.pop("dropped_exact_duplicate_ids", None)
    print(
        json.dumps(
            {
                "dataset": args.dataset,
                "output_dir": str(output_dir),
                "source_summary": compact_source_summary,
                "protocol_split_sizes": manifest["split_sizes"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )


def _source_manifest(
    args: argparse.Namespace,
    source_summary: dict[str, Any],
    source_train_path: Path,
    source_test_path: Path,
) -> dict[str, Any]:
    return {
        **source_summary,
        "source": {
            "name": str(args.source_name),
            "input_path": str(args.input_path),
            "input_sha256": sha256_file(args.input_path),
            "label_policy": "groundtruth_only",
            "not_a_recovery_claim": True,
        },
        "record_schema": {
            "id": "source-stable identifier",
            "query": "single source query",
            "document": "source document text",
            "groundtruth": "source binary label",
        },
        "splits": {
            "train": {"path": str(source_train_path), "sha256": sha256_file(source_train_path)},
            "test": {"path": str(source_test_path), "sha256": sha256_file(source_test_path)},
        },
    }


if __name__ == "__main__":
    main()
