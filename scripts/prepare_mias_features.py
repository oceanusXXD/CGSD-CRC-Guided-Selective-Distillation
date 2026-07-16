from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mias_dcms.selectors import assert_selector_rows_are_label_safe
from mias_dcms.utils import read_jsonl, write_json, write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Package precomputed frozen embeddings into MIAS feature rows."
    )
    parser.add_argument("--task", choices=("classification", "preference"), required=True)
    parser.add_argument("--rows_path", type=Path, required=True)
    parser.add_argument("--output_path", type=Path, required=True)
    parser.add_argument("--embedding_path", type=Path)
    parser.add_argument("--embedding_ids_path", type=Path)
    parser.add_argument("--response_a_embedding_path", type=Path)
    parser.add_argument("--response_a_ids_path", type=Path)
    parser.add_argument("--response_b_embedding_path", type=Path)
    parser.add_argument("--response_b_ids_path", type=Path)
    parser.add_argument("--metadata_path", type=Path, action="append", default=[])
    parser.add_argument("--allow_seed_labels", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = read_jsonl(args.rows_path)
    if not args.allow_seed_labels:
        assert_selector_rows_are_label_safe(rows)
    metadata = [_rows_by_id(read_jsonl(path), name=str(path)) for path in args.metadata_path]
    if args.task == "classification":
        features = _classification_features(args, rows)
    else:
        features = _preference_features(args, rows, metadata)
    write_jsonl(features, args.output_path)
    write_json(
        {
            "task": str(args.task),
            "rows_path": str(args.rows_path),
            "output_path": str(args.output_path),
            "row_count": len(features),
            "feature_dimension": _feature_dimension(features, task=str(args.task)),
            "metadata_paths": [str(path) for path in args.metadata_path],
            "response_aware": args.task == "preference",
        },
        args.output_path.with_suffix(".summary.json"),
    )


def _classification_features(args: argparse.Namespace, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if args.embedding_path is None or args.embedding_ids_path is None:
        raise ValueError("classification features require --embedding_path and --embedding_ids_path")
    vectors = _embedding_map(args.embedding_path, args.embedding_ids_path)
    _require_exact_coverage(rows, vectors, name="classification embeddings")
    return [{"sample_id": _row_id(row), "representation_embedding": vectors[_row_id(row)].tolist()} for row in rows]


def _preference_features(
    args: argparse.Namespace,
    rows: list[dict[str, Any]],
    metadata: list[dict[str, dict[str, Any]]],
) -> list[dict[str, Any]]:
    required = (
        args.response_a_embedding_path,
        args.response_a_ids_path,
        args.response_b_embedding_path,
        args.response_b_ids_path,
    )
    if any(value is None for value in required):
        raise ValueError("preference features require both response A/B embedding matrices and id files")
    a_vectors = _embedding_map(args.response_a_embedding_path, args.response_a_ids_path)
    b_vectors = _embedding_map(args.response_b_embedding_path, args.response_b_ids_path)
    _require_exact_coverage(rows, a_vectors, name="response A embeddings")
    _require_exact_coverage(rows, b_vectors, name="response B embeddings")
    output = []
    for row in rows:
        sample_id = _row_id(row)
        merged = dict(row)
        for values in metadata:
            if sample_id not in values:
                raise ValueError(f"metadata is missing preference row {sample_id!r}")
            merged.update(values[sample_id])
        feature = {
            "sample_id": sample_id,
            "response_a_embedding": a_vectors[sample_id].tolist(),
            "response_b_embedding": b_vectors[sample_id].tolist(),
        }
        for field in (
            "response_a_token_count",
            "response_b_token_count",
            "response_1_token_count",
            "response_2_token_count",
            "response_a_word_count",
            "response_b_word_count",
            "response_1_word_count",
            "response_2_word_count",
            "completion_token_cost",
            "prompt_cluster",
            "prompt_cluster_id",
            "prompt_cluster_membership",
            "ab_position",
        ):
            if merged.get(field) is not None:
                feature[field] = merged[field]
        if not any(
            field in feature
            for field in (
                "response_a_token_count",
                "response_1_token_count",
                "response_a_word_count",
                "response_1_word_count",
            )
        ):
            response_a = merged.get("response_a", merged.get("response_1", ""))
            response_b = merged.get("response_b", merged.get("response_2", ""))
            feature["response_a_word_count"] = len(str(response_a).split())
            feature["response_b_word_count"] = len(str(response_b).split())
        output.append(feature)
    return output


def _embedding_map(matrix_path: Path, ids_path: Path) -> dict[str, np.ndarray]:
    matrix = np.asarray(np.load(matrix_path), dtype=np.float32)
    id_rows = read_jsonl(ids_path)
    ids = [_row_id(row) for row in id_rows]
    if matrix.ndim != 2 or matrix.shape[0] != len(ids):
        raise ValueError(f"embedding matrix {matrix_path} does not match {ids_path}")
    if len(set(ids)) != len(ids) or np.any(~np.isfinite(matrix)):
        raise ValueError("embedding artifacts must contain unique ids and finite vectors")
    return dict(zip(ids, matrix, strict=True))


def _rows_by_id(rows: list[dict[str, Any]], *, name: str) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for row in rows:
        sample_id = _row_id(row)
        if sample_id in output:
            raise ValueError(f"{name} contains duplicate id {sample_id!r}")
        output[sample_id] = {key: value for key, value in row.items() if key not in {"id", "sample_id"}}
    return output


def _require_exact_coverage(rows: list[dict[str, Any]], vectors: dict[str, np.ndarray], *, name: str) -> None:
    row_ids = {_row_id(row) for row in rows}
    if row_ids != set(vectors):
        raise ValueError(
            f"{name} must exactly cover rows: "
            f"missing={len(row_ids - set(vectors))}, extra={len(set(vectors) - row_ids)}"
        )


def _feature_dimension(rows: list[dict[str, Any]], *, task: str) -> int:
    if not rows:
        return 0
    field = "representation_embedding" if task == "classification" else "response_a_embedding"
    return len(rows[0][field])


def _row_id(row: dict[str, Any]) -> str:
    value = row.get("sample_id", row.get("id"))
    if value is None or not str(value):
        raise ValueError("row is missing a non-empty sample_id/id")
    return str(value)


if __name__ == "__main__":
    main()
