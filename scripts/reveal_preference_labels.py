from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mias_dcms.preference_reveal import reveal_selected_preference_labels
from mias_dcms.utils import read_json, read_jsonl, write_json, write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reveal oracle preference labels only for selected active-pool pairs."
    )
    parser.add_argument("--active_pool_path", type=Path, required=True)
    parser.add_argument("--oracle_store_path", type=Path, required=True)
    parser.add_argument("--selected_ids_path", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--round_index", type=int, required=True)
    parser.add_argument("--method", required=True)
    parser.add_argument("--id_field", default="sample_id")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    active_pool = read_jsonl(args.active_pool_path)
    oracle_store = read_json(args.oracle_store_path)
    selected_payload = read_json(args.selected_ids_path)
    selected_ids = [str(sample_id) for sample_id in selected_payload["selected_ids"]]
    result = reveal_selected_preference_labels(
        active_pool,
        oracle_store=oracle_store,
        selected_ids=selected_ids,
        round_index=int(args.round_index),
        method=str(args.method),
        id_field=str(args.id_field),
    )

    output_dir = args.output_dir
    write_jsonl(result.revealed_rows, output_dir / "revealed_rows.jsonl")
    write_jsonl(result.training_rows, output_dir / "dpo_train_rows.jsonl")
    summary = {
        "active_pool_path": str(args.active_pool_path),
        "oracle_store_path": str(args.oracle_store_path),
        "selected_ids_path": str(args.selected_ids_path),
        "round": int(args.round_index),
        "method": str(args.method),
        "selected_count": len(selected_ids),
        "revealed_count": len(result.revealed_rows),
        "dpo_train_row_count": len(result.training_rows),
        "unrevealed_count": len(result.unrevealed_ids),
        "revealed_ids": result.revealed_ids,
        "unrevealed_id_preview": result.unrevealed_ids[:10],
        "artifacts": {
            "revealed_rows": str(output_dir / "revealed_rows.jsonl"),
            "dpo_train_rows": str(output_dir / "dpo_train_rows.jsonl"),
        },
    }
    write_json(summary, output_dir / "summary.json")
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
