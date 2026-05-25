#!/usr/bin/env python

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.binary_protocol import normalize_binary_label
from src.utils import resolve_input_path, resolve_output_path, write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_path", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--id_field", default="id")
    parser.add_argument("--query_field", default="query")
    parser.add_argument("--document_field", default="document")
    parser.add_argument("--label_field", default="groundtruth")
    parser.add_argument("--fixed_query", default=None, help="Use this query when the source file has no query field.")
    parser.add_argument("--id_prefix", default="row")
    return parser.parse_args()


def _row_value(row: dict[str, Any], field_name: str, *, source: str) -> Any:
    if field_name not in row:
        raise KeyError(f"{source} missing field {field_name!r}")
    return row[field_name]


def convert_rows(
    rows: list[dict[str, Any]],
    *,
    id_field: str = "id",
    query_field: str = "query",
    document_field: str = "document",
    label_field: str = "groundtruth",
    fixed_query: str | None = None,
    id_prefix: str = "row",
) -> list[dict[str, Any]]:
    converted: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for index, row in enumerate(rows, start=1):
        source = f"row {index}"
        sample_id = row.get(id_field)
        if sample_id is None or str(sample_id) == "":
            sample_id = f"{id_prefix}_{index}"
        sample_id = str(sample_id)
        if sample_id in seen_ids:
            raise ValueError(f"duplicate sample id after conversion: {sample_id!r}")
        seen_ids.add(sample_id)

        query = fixed_query if fixed_query is not None else _row_value(row, query_field, source=source)
        document = _row_value(row, document_field, source=source)
        label = normalize_binary_label(
            _row_value(row, label_field, source=source),
            field_name=f"{source} {label_field}",
        )
        payload = dict(row)
        payload.update(
            {
                "id": sample_id,
                "query": str(query),
                "document": str(document),
                "groundtruth": int(label),
            }
        )
        converted.append(payload)
    return converted


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number} must be a JSON object")
            rows.append(row)
    return rows


def main() -> None:
    args = parse_args()
    input_path = resolve_input_path(args.input_path, PROJECT_ROOT)
    output_path = resolve_output_path(args.output_path, PROJECT_ROOT)
    rows = read_jsonl(input_path)
    converted = convert_rows(
        rows,
        id_field=str(args.id_field),
        query_field=str(args.query_field),
        document_field=str(args.document_field),
        label_field=str(args.label_field),
        fixed_query=args.fixed_query,
        id_prefix=str(args.id_prefix),
    )
    write_jsonl(converted, output_path)
    print(json.dumps({"converted_rows": len(converted), "output_path": str(output_path)}, sort_keys=True))


if __name__ == "__main__":
    main()
