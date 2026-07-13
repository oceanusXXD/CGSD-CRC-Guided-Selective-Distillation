from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mias_dcms.selectors import (
    assert_selector_rows_are_label_safe,
    random_group_without_replacement,
    random_without_replacement,
)
from mias_dcms.preference_selection_metrics import (
    build_preference_selection_metrics,
    materialize_preference_group_fields,
)
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
    parser.add_argument(
        "--selection_group_field",
        default="",
        help="Optional observable field that may contribute at most one selected row per group.",
    )
    parser.add_argument(
        "--metadata_path",
        action="append",
        default=[],
        help="Optional selector-safe JSONL metadata to merge by sample_id before selection.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = _merge_metadata(
        read_jsonl(args.input_path),
        metadata_paths=[Path(path) for path in args.metadata_path],
        id_field=str(args.id_field),
    )
    assert_selector_rows_are_label_safe(rows)
    sample_ids = [str(row[args.id_field]) for row in rows]
    selection_group_field = str(args.selection_group_field).strip()
    if selection_group_field:
        group_ids = [str(row.get(selection_group_field, "")) for row in rows]
        if any(not group_id for group_id in group_ids):
            raise ValueError(f"rows are missing selection group field {selection_group_field!r}")
        selected_ids = random_group_without_replacement(
            sample_ids,
            group_ids,
            budget=int(args.budget),
            seed=int(args.seed),
        )
    else:
        selected_ids = random_without_replacement(
            sample_ids,
            budget=int(args.budget),
            seed=int(args.seed),
        )
    selected_id_set = set(selected_ids)
    membership_rows = [
        {
            **materialize_preference_group_fields(row),
            str(args.id_field): sample_id,
            "sample_id": sample_id,
            "method": "random",
            "selected": int(sample_id in selected_id_set),
        }
        for row, sample_id in zip(rows, sample_ids, strict=True)
    ]
    summary = {
        "input_path": str(args.input_path),
        "method": "random",
        "budget": int(args.budget),
        "seed": int(args.seed),
        "selection_group_field": selection_group_field or None,
        "pool_size": len(rows),
        "selected_count": len(selected_ids),
        "selection_metrics": build_preference_selection_metrics(
            membership_rows,
            method="random",
        ),
    }
    selected_payload = {
        "selected_ids": selected_ids,
        "budget": int(args.budget),
        "selected_count": len(selected_ids),
        "method": "random",
        "seed": int(args.seed),
        "selection_group_field": selection_group_field or None,
    }

    output_dir = args.output_dir
    write_json(selected_payload, output_dir / "selected_ids.json")
    write_jsonl(membership_rows, output_dir / "membership.jsonl")
    write_json(summary, output_dir / "selection_summary.json")
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))


def _merge_metadata(
    rows: list[dict[str, object]],
    *,
    metadata_paths: list[Path],
    id_field: str,
) -> list[dict[str, object]]:
    merged = [dict(row) for row in rows]
    for path in metadata_paths:
        metadata_by_id: dict[str, dict[str, object]] = {}
        for row in read_jsonl(path):
            sample_id = str(row.get(id_field, row.get("id")))
            if not sample_id or sample_id == "None":
                raise ValueError(f"metadata row in {path} is missing id field {id_field!r}")
            if sample_id in metadata_by_id:
                raise ValueError(f"duplicate metadata row for sample id {sample_id!r} in {path}")
            metadata_by_id[sample_id] = dict(row)
        for row in merged:
            sample_id = str(row.get(id_field, row.get("id")))
            if sample_id in metadata_by_id:
                row.update(metadata_by_id[sample_id])
    return merged


if __name__ == "__main__":
    main()
