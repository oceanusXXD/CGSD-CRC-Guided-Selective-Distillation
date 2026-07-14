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
from mias_dcms.utils import read_json, read_jsonl, write_json, write_jsonl  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Freeze a selector-safe binary benchmark protocol with an untouched official test split."
    )
    parser.add_argument("--train_path", type=Path, required=True)
    parser.add_argument("--validation_path", type=Path)
    parser.add_argument("--test_path", type=Path, required=True)
    parser.add_argument("--source_manifest_path", type=Path)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--seed_label_count", type=int, required=True)
    parser.add_argument("--active_pool_size", type=int, required=True)
    parser.add_argument("--development_size", type=int, default=0)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--train_row_limit", type=int)
    parser.add_argument("--validation_row_limit", type=int)
    parser.add_argument("--test_row_limit", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _validate_source_manifest(args)
    artifacts = prepare_binary_benchmark_protocol(
        read_jsonl(args.train_path),
        validation_rows=read_jsonl(args.validation_path) if args.validation_path else None,
        test_rows=read_jsonl(args.test_path),
        dataset=str(args.dataset),
        seed_label_count=int(args.seed_label_count),
        active_pool_size=int(args.active_pool_size),
        seed=int(args.seed),
        development_size=int(args.development_size),
        train_row_limit=args.train_row_limit,
        validation_row_limit=args.validation_row_limit,
        test_row_limit=args.test_row_limit,
    )
    output_dir = args.output_dir
    paths = {
        "seed_train_rows": output_dir / "seed_train_rows.jsonl",
        "selection_pool": output_dir / "selection_pool.jsonl",
        "selection_oracle_store": output_dir / "selection_oracle_store.json",
        "development_rows": output_dir / "development_rows.jsonl",
        "official_test_rows": output_dir / "official_test_rows.jsonl",
    }
    write_jsonl(artifacts["seed_train_rows"], paths["seed_train_rows"])
    write_jsonl(artifacts["selection_pool"], paths["selection_pool"])
    write_json(artifacts["selection_oracle_store"], paths["selection_oracle_store"])
    write_jsonl(artifacts["development_rows"], paths["development_rows"])
    write_jsonl(artifacts["official_test_rows"], paths["official_test_rows"])

    manifest = dict(artifacts["protocol_manifest"])
    manifest["source_inputs"] = _input_hashes(args)
    manifest["artifacts"] = {name: str(path) for name, path in paths.items()}
    write_json(manifest, output_dir / "protocol_manifest.json")
    print(
        json.dumps(
            {
                "dataset": args.dataset,
                "output_dir": str(output_dir),
                "split_sizes": manifest["split_sizes"],
                "development_source": manifest["development_source"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )


def _input_hashes(args: argparse.Namespace) -> dict[str, dict[str, str]]:
    inputs: dict[str, Path | None] = {
        "train": args.train_path,
        "validation": args.validation_path,
        "official_test": args.test_path,
        "source_manifest": args.source_manifest_path,
    }
    return {
        name: {"path": str(path), "sha256": sha256_file(path)}
        for name, path in inputs.items()
        if path is not None
    }


def _validate_source_manifest(args: argparse.Namespace) -> None:
    if args.source_manifest_path is None:
        return
    source_manifest = read_json(args.source_manifest_path)
    if str(source_manifest.get("dataset", "")) != str(args.dataset):
        raise ValueError("source manifest dataset does not match --dataset")
    splits = source_manifest.get("splits")
    if not isinstance(splits, dict):
        raise ValueError("source manifest has no split metadata")
    supplied = {"train": args.train_path, "test": args.test_path}
    if args.validation_path is not None:
        supplied["validation"] = args.validation_path
    for split, path in supplied.items():
        details = splits.get(split)
        if not isinstance(details, dict) or details.get("sha256") != sha256_file(path):
            raise ValueError(f"{split} input hash does not match source manifest")


if __name__ == "__main__":
    main()
