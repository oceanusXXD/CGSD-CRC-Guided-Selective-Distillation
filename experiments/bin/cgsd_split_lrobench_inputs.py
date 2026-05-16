#!/usr/bin/env python
"""Split a merged LROBench CGSD input and embedding matrix into per-query inputs."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import numpy as np


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def sanitize_group_name(value: str) -> str:
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value).strip())
    text = text.strip("._-")
    return text or "unknown"


def row_group(row: dict[str, Any]) -> str:
    if row.get("query_id") not in {None, ""}:
        return sanitize_group_name(str(row["query_id"]))
    sample_id = str(row.get("id", row.get("sample_id", "")))
    if ":" in sample_id:
        return sanitize_group_name(sample_id.rsplit(":", 1)[1])
    return sanitize_group_name(str(row.get("query", "unknown"))[:48])


def read_embedding_ids(embeddings_path: Path) -> list[str]:
    ids_path = embeddings_path.with_suffix(".ids.jsonl")
    if not ids_path.exists():
        raise FileNotFoundError(f"missing embedding id sidecar: {ids_path}")
    ids: list[str] = []
    for row in read_jsonl(ids_path):
        sample_id = row.get("id", row.get("sample_id"))
        if sample_id is None:
            raise ValueError(f"{ids_path} rows must contain id or sample_id")
        ids.append(str(sample_id))
    return ids


def split_lrobench_inputs(
    *,
    data_path: str | Path,
    embeddings_path: str | Path,
    output_root: str | Path,
    prefix: str = "lrobench",
) -> dict[str, Any]:
    data_source = Path(data_path)
    embedding_source = Path(embeddings_path)
    output_base = Path(output_root)
    rows = read_jsonl(data_source)
    matrix = np.load(embedding_source, allow_pickle=False)
    embedding_ids = read_embedding_ids(embedding_source)
    if matrix.ndim != 2:
        raise ValueError(f"{embedding_source} must be a 2D matrix, got {matrix.shape}")
    if len(embedding_ids) != int(matrix.shape[0]):
        raise ValueError("embedding matrix row count does not match ids sidecar")
    vector_by_id = {sample_id: matrix[index] for index, sample_id in enumerate(embedding_ids)}

    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        sample_id = str(row.get("id", row.get("sample_id", "")))
        if sample_id not in vector_by_id:
            raise ValueError(f"missing embedding for data row {sample_id!r}")
        groups.setdefault(row_group(row), []).append(row)

    summary_rows: list[dict[str, Any]] = []
    for group_name, group_rows in sorted(groups.items()):
        target = output_base / f"{prefix}_{group_name}"
        target.mkdir(parents=True, exist_ok=True)
        ids = [str(row.get("id", row.get("sample_id"))) for row in group_rows]
        vectors = np.vstack([np.asarray(vector_by_id[sample_id], dtype=np.float32) for sample_id in ids]).astype(np.float32)
        write_jsonl(group_rows, target / "data.jsonl")
        np.save(target / "embeddings.npy", vectors)
        write_jsonl([{"id": sample_id} for sample_id in ids], target / "embeddings.ids.jsonl")
        (target / "embeddings.meta.json").write_text(
            json.dumps(
                {
                    "source_data_path": str(data_source),
                    "source_embeddings_path": str(embedding_source),
                    "group": group_name,
                    "row_count": len(group_rows),
                    "dimension": int(vectors.shape[1]),
                },
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        summary_rows.append({"group": group_name, "input_dir": str(target), "row_count": len(group_rows)})

    summary = {
        "source_data_path": str(data_source),
        "source_embeddings_path": str(embedding_source),
        "output_root": str(output_base),
        "groups": len(summary_rows),
        "rows": len(rows),
        "items": summary_rows,
    }
    (output_base / f"{prefix}_split_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_path", default="experiments/inputs/lrobench/data.jsonl")
    parser.add_argument("--embeddings_path", default="experiments/inputs/lrobench/embeddings.npy")
    parser.add_argument("--output_root", default="experiments/inputs")
    parser.add_argument("--prefix", default="lrobench")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = split_lrobench_inputs(
        data_path=args.data_path,
        embeddings_path=args.embeddings_path,
        output_root=args.output_root,
        prefix=args.prefix,
    )
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
