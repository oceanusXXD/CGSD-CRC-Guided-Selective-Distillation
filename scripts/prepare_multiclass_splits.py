from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mias_dcms.multiclass_protocol import build_fixed_multiclass_splits, pool_class_prior
from mias_dcms.selectors import FORBIDDEN_SELECTOR_INPUT_FIELDS
from mias_dcms.utils import read_jsonl, write_json, write_jsonl


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
        label_field=str(args.label_field),
    )
    prior = pool_class_prior(rows, label_field=str(args.label_field))
    output_dir = Path(args.output_dir)
    materialized = _materialize_split_rows(
        rows,
        splits=splits,
        id_field=str(args.id_field),
        label_field=str(args.label_field),
    )
    write_json(splits, output_dir / "split_ids.json")
    write_json(prior.as_dict(), output_dir / "pool_prior.json")
    write_jsonl(materialized["seed_rows"], output_dir / "seed_rows.jsonl")
    write_jsonl(materialized["active_pool"], output_dir / "active_pool.jsonl")
    write_json(materialized["active_oracle_store"], output_dir / "active_oracle_store.json")
    write_jsonl(materialized["test_rows"], output_dir / "test_rows.jsonl")
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
            "seed_rows": str(output_dir / "seed_rows.jsonl"),
            "active_pool": str(output_dir / "active_pool.jsonl"),
            "active_oracle_store": str(output_dir / "active_oracle_store.json"),
            "test_rows": str(output_dir / "test_rows.jsonl"),
        },
    }
    write_json(summary, output_dir / "summary.json")
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))


def _materialize_split_rows(
    rows: list[dict[str, object]],
    *,
    splits: dict[str, list[str]],
    id_field: str,
    label_field: str,
) -> dict[str, object]:
    """Materialize minimal train/select/eval files from fixed split ids.

    The active pool deliberately excludes every known label-bearing field. The
    matching labels remain in a separate oracle store and can only be joined
    after selection.
    """
    by_id = {str(row[id_field]): dict(row) for row in rows}
    expected_ids = set().union(*(set(values) for values in splits.values()))
    missing_ids = sorted(expected_ids - set(by_id))
    if missing_ids:
        raise ValueError(f"split ids are missing from input rows: {missing_ids[:5]}")

    def rows_for(split_name: str) -> list[dict[str, object]]:
        return [dict(by_id[sample_id]) for sample_id in splits[split_name]]

    active_rows = rows_for("active_pool_ids")
    active_pool = [
        {
            key: value
            for key, value in row.items()
            if key not in FORBIDDEN_SELECTOR_INPUT_FIELDS and key != label_field
        }
        for row in active_rows
    ]
    oracle_store = {
        str(row[id_field]): {
            str(id_field): str(row[id_field]),
            str(label_field): row[label_field],
        }
        for row in active_rows
    }
    return {
        "seed_rows": rows_for("seed_ids"),
        "active_pool": active_pool,
        "active_oracle_store": oracle_store,
        "test_rows": rows_for("test_ids"),
    }


if __name__ == "__main__":
    main()
