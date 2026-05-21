"""embedding 工件读取和覆盖校验。

本模块只处理已经落盘的 embedding 工件：读取 `.npy/.npz/.json/.jsonl`
向量文件，并用 `*.ids.jsonl` 这类 sidecar 建立样本 ID 到向量的映射。
需要依赖 embedding 的 stage 可以调用 `assert_embedding_coverage`，在运行前确认
每个样本 ID 都有向量且维度一致。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from .utils import read_json, read_jsonl


def _embedding_ids_from_sidecar(path: Path, row_count: int) -> list[str]:
    candidates = [
        path.with_suffix(".ids.jsonl"),
        path.with_suffix(".ids.json"),
        path.parent / "embedding_rows.jsonl",
    ]
    for candidate in candidates:
        if not candidate.exists():
            continue
        if candidate.suffix == ".jsonl":
            ids = [
                str(row.get("id", row.get("sample_id", "")))
                for row in read_jsonl(candidate)
            ]
        else:
            payload = read_json(candidate)
            raw_ids = payload.get("ids", payload.get("embedding_ids", payload)) if isinstance(payload, dict) else payload
            ids = [
                str(item.get("id", item.get("sample_id", "")) if isinstance(item, dict) else item)
                for item in raw_ids
            ]
        ids = [sample_id for sample_id in ids if sample_id]
        if len(ids) == int(row_count):
            if len(set(ids)) != len(ids):
                raise ValueError(f"{candidate} contains duplicate embedding ids")
            return ids
    raise ValueError(
        f"{path} is a .npy embedding matrix with {row_count} rows, but no matching id sidecar was found. "
        "Expected *.ids.jsonl, *.ids.json, or embedding_rows.jsonl."
    )


def load_embeddings(path: str | Path) -> dict[str, np.ndarray]:
    source = Path(path)
    if source.suffix == ".npy":
        matrix = np.load(source, allow_pickle=False)
        if matrix.ndim != 2:
            raise ValueError(f"{source} must contain a 2D embedding matrix, got shape {matrix.shape}")
        ids = _embedding_ids_from_sidecar(source, int(matrix.shape[0]))
        return {
            sample_id: np.asarray(matrix[index], dtype=np.float32)
            for index, sample_id in enumerate(ids)
        }
    if source.suffix == ".npz":
        payload = np.load(source, allow_pickle=False)
        return {str(key): np.asarray(payload[key], dtype=np.float32) for key in payload.files}
    if source.suffix == ".json":
        data = read_json(source)
        return {str(key): np.asarray(value, dtype=np.float32) for key, value in data.items()}

    embeddings: dict[str, np.ndarray] = {}
    with source.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            row = json.loads(stripped)
            sample_id = str(row.get("id", row.get("sample_id", "")))
            vector = row.get("embedding")
            if not sample_id or vector is None:
                raise ValueError(f"{source}:{line_number} must contain id/sample_id and embedding")
            if sample_id in embeddings:
                raise ValueError(f"{source}:{line_number} duplicate embedding id: {sample_id!r}")
            embeddings[sample_id] = np.asarray(vector, dtype=np.float32)
    return embeddings


def assert_embedding_coverage(
    embeddings_by_id: dict[str, np.ndarray],
    rows: list[dict[str, Any]],
    *,
    expected_dim: int,
) -> None:
    row_ids = [str(row["id"]) for row in rows]
    missing = [sample_id for sample_id in row_ids if sample_id not in embeddings_by_id]
    if missing:
        raise ValueError(f"embeddings_path is missing {len(missing)} sample ids, examples: {missing[:5]}")

    bad_shape: list[tuple[str, tuple[int, ...]]] = []
    for sample_id in row_ids:
        vector = np.asarray(embeddings_by_id[sample_id], dtype=np.float32)
        if vector.ndim != 1 or (expected_dim > 0 and vector.shape[0] != int(expected_dim)):
            bad_shape.append((sample_id, tuple(vector.shape)))
    if bad_shape:
        raise ValueError(
            f"embeddings_path has vectors with invalid shape; expected ({expected_dim},), "
            f"examples: {bad_shape[:5]}"
        )
