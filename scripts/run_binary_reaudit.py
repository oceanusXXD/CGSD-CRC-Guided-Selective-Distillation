from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mias_dcms.binary_reaudit import (
    materialize_binary_reaudit_selection,
    prepare_binary_reaudit_splits,
    sha256_file,
)
from mias_dcms.utils import read_json, read_jsonl, write_json, write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare and audit selector-safe binary MIAS re-audit artifacts."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--input_path", type=Path, required=True)
    prepare.add_argument("--output_dir", type=Path, required=True)
    prepare.add_argument("--dataset", required=True)
    prepare.add_argument("--seed_label_count", type=int, required=True)
    prepare.add_argument("--active_pool_size", type=int, required=True)
    prepare.add_argument("--test_size", type=int, required=True)
    prepare.add_argument("--seed", type=int, required=True)
    prepare.add_argument("--row_limit", type=int)
    prepare.add_argument("--id_field", default="id")
    prepare.add_argument("--query_field", default="query")
    prepare.add_argument("--document_field", default="document")
    prepare.add_argument("--label_field", default="groundtruth")

    select = subparsers.add_parser("select")
    select.add_argument("--scored_path", type=Path, required=True)
    select.add_argument("--oracle_store_path", type=Path, required=True)
    select.add_argument("--seed_train_rows_path", type=Path, required=True)
    select.add_argument("--output_dir", type=Path, required=True)
    select.add_argument("--dataset", required=True)
    select.add_argument("--model", required=True)
    select.add_argument("--methods", required=True)
    select.add_argument("--budget", type=int, required=True)
    select.add_argument("--seed", type=int, required=True)
    select.add_argument("--config_hash", required=True)
    select.add_argument("--evaluation_label_count", type=int, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "prepare":
        payload = _prepare(args)
    else:
        payload = _select(args)
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True))


def _prepare(args: argparse.Namespace) -> dict[str, Any]:
    rows = read_jsonl(args.input_path)
    artifacts = prepare_binary_reaudit_splits(
        rows,
        dataset=str(args.dataset),
        seed_label_count=int(args.seed_label_count),
        active_pool_size=int(args.active_pool_size),
        test_size=int(args.test_size),
        seed=int(args.seed),
        id_field=str(args.id_field),
        query_field=str(args.query_field),
        document_field=str(args.document_field),
        label_field=str(args.label_field),
        row_limit=int(args.row_limit) if args.row_limit is not None else None,
    )
    output_dir = args.output_dir
    write_jsonl(artifacts["seed_train_rows"], output_dir / "seed_train_rows.jsonl")
    write_jsonl(artifacts["selection_pool"], output_dir / "selection_pool.jsonl")
    write_json(artifacts["selection_oracle_store"], output_dir / "selection_oracle_store.json")
    write_jsonl(artifacts["test_rows"], output_dir / "test_rows.jsonl")
    manifest = {
        **artifacts["split_manifest"],
        "input_path": str(args.input_path),
        "input_sha256": sha256_file(args.input_path),
        "row_limit": int(args.row_limit) if args.row_limit is not None else None,
        "artifacts": {
            "seed_train_rows": str(output_dir / "seed_train_rows.jsonl"),
            "selection_pool": str(output_dir / "selection_pool.jsonl"),
            "selection_oracle_store": str(output_dir / "selection_oracle_store.json"),
            "test_rows": str(output_dir / "test_rows.jsonl"),
        },
    }
    write_json(manifest, output_dir / "split_manifest.json")
    return {
        "dataset": str(args.dataset),
        "seed": int(args.seed),
        "source_size": int(artifacts["source_size"]),
        "split_sizes": dict(manifest["split_sizes"]),
        "split_manifest_path": str(output_dir / "split_manifest.json"),
        "artifacts": dict(manifest["artifacts"]),
    }


def _select(args: argparse.Namespace) -> dict[str, Any]:
    methods = [part.strip() for part in str(args.methods).split(",") if part.strip()]
    results = materialize_binary_reaudit_selection(
        read_jsonl(args.scored_path),
        oracle_store=read_json(args.oracle_store_path),
        seed_train_rows=read_jsonl(args.seed_train_rows_path),
        dataset=str(args.dataset),
        model=str(args.model),
        methods=methods,
        budget=int(args.budget),
        seed=int(args.seed),
        config_hash=str(args.config_hash),
        evaluation_label_count=int(args.evaluation_label_count),
    )
    output_dir = args.output_dir
    summary: dict[str, Any] = {"methods": {}, "scored_path": str(args.scored_path)}
    for method, payload in results.items():
        method_dir = output_dir / _method_directory(method)
        write_json(payload["selected_ids"] and {"selected_ids": payload["selected_ids"]} or {}, method_dir / "selected_ids.json")
        write_jsonl(payload["membership"], method_dir / "membership.jsonl")
        write_jsonl(payload["revealed_rows"], method_dir / "revealed_rows.jsonl")
        write_jsonl(payload["train_rows"], method_dir / "train_rows.jsonl")
        selection_summary = {
            key: value
            for key, value in payload.items()
            if key not in {"membership", "revealed_rows", "train_rows"}
        }
        write_json(selection_summary, method_dir / "selection_summary.json")
        summary["methods"][method] = {
            "selected_count": len(payload["selected_ids"]),
            "selection_summary_path": str(method_dir / "selection_summary.json"),
            "train_rows_path": str(method_dir / "train_rows.jsonl"),
        }
    write_json(summary, output_dir / "selection_stage_summary.json")
    return summary


def _method_directory(method: str) -> str:
    return str(method).strip().lower().replace("+", "_").replace("-", "_").replace(" ", "_")


if __name__ == "__main__":
    main()
