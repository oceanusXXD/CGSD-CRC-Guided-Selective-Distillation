from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from mias_dcms.selectors import assert_selector_rows_are_label_safe


@dataclass(frozen=True)
class PromptClusterResult:
    rows: list[dict[str, Any]]
    summary: dict[str, Any]


def build_prompt_cluster_assignments(
    *,
    rows: Iterable[Mapping[str, Any]],
    embeddings_by_id: Mapping[str, Sequence[float] | np.ndarray],
    cluster_count: int,
    id_field: str = "sample_id",
    max_iterations: int = 50,
    softmax_temperature: float = 1.0,
) -> PromptClusterResult:
    source_rows = [dict(row) for row in rows]
    assert_selector_rows_are_label_safe(source_rows)
    if not source_rows:
        raise ValueError("rows must not be empty")
    if cluster_count <= 0:
        raise ValueError("cluster_count must be positive")
    if cluster_count > len(source_rows):
        raise ValueError("cluster_count must not exceed row count")
    if max_iterations <= 0:
        raise ValueError("max_iterations must be positive")

    sample_ids = [_row_id(row, id_field=id_field) for row in source_rows]
    vectors = _aligned_normalized_vectors(sample_ids, embeddings_by_id)
    centroids = _initial_centroids(vectors, sample_ids, int(cluster_count))
    assignments = np.full((len(sample_ids),), -1, dtype=np.int64)
    converged = False
    iterations = 0

    for iteration in range(1, int(max_iterations) + 1):
        distances = _squared_distances(vectors, centroids)
        next_assignments = np.argmin(distances, axis=1).astype(np.int64)
        iterations = iteration
        if np.array_equal(assignments, next_assignments):
            converged = True
            break
        assignments = next_assignments
        centroids = _updated_centroids(vectors, assignments, centroids)

    distances = _squared_distances(vectors, centroids)
    probabilities = _soft_cluster_probabilities(distances, temperature=float(softmax_temperature))
    labels = [f"c{index}" for index in range(int(cluster_count))]
    assignment_rows: list[dict[str, Any]] = []
    for row_index, sample_id in enumerate(sample_ids):
        cluster_index = int(assignments[row_index])
        assignment_rows.append(
            {
                "sample_id": sample_id,
                "id": str(source_rows[row_index].get("id", sample_id)),
                "prompt_cluster": labels[cluster_index],
                "prompt_cluster_id": cluster_index,
                "prompt_cluster_distance": float(distances[row_index, cluster_index]),
                "prompt_cluster_probabilities": [float(value) for value in probabilities[row_index]],
                "prompt_cluster_membership": {
                    label: float(probabilities[row_index, index])
                    for index, label in enumerate(labels)
                },
            }
        )

    cluster_counts = {
        labels[index]: int(np.sum(assignments == index))
        for index in range(int(cluster_count))
    }
    summary = {
        "row_count": len(source_rows),
        "cluster_count": int(cluster_count),
        "cluster_counts": cluster_counts,
        "converged": bool(converged),
        "iterations": int(iterations),
        "inertia": float(sum(distances[index, int(assignments[index])] for index in range(len(sample_ids)))),
        "softmax_temperature": float(softmax_temperature),
        "id_field": str(id_field),
        "cluster_labels": labels,
        "centroids": centroids.astype(float).tolist(),
    }
    return PromptClusterResult(rows=assignment_rows, summary=summary)


def _row_id(row: Mapping[str, Any], *, id_field: str) -> str:
    value = row.get(id_field, row.get("id"))
    if value is None:
        raise ValueError(f"row is missing id field {id_field!r} and fallback 'id'")
    return str(value)


def _aligned_normalized_vectors(
    sample_ids: Sequence[str],
    embeddings_by_id: Mapping[str, Sequence[float] | np.ndarray],
) -> np.ndarray:
    vectors: list[np.ndarray] = []
    missing: list[str] = []
    expected_dim: int | None = None
    for sample_id in sample_ids:
        if sample_id not in embeddings_by_id:
            missing.append(sample_id)
            continue
        vector = np.asarray(embeddings_by_id[sample_id], dtype=np.float32)
        if vector.ndim != 1:
            raise ValueError(f"embedding for {sample_id!r} must be one-dimensional")
        if expected_dim is None:
            expected_dim = int(vector.shape[0])
        elif int(vector.shape[0]) != expected_dim:
            raise ValueError("all embeddings must have the same dimension")
        vectors.append(vector)
    if missing:
        raise ValueError(f"embeddings missing {len(missing)} sample ids, examples: {missing[:5]}")
    matrix = np.vstack(vectors).astype(np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms <= 1e-12] = 1.0
    return (matrix / norms).astype(np.float32)


def _initial_centroids(vectors: np.ndarray, sample_ids: Sequence[str], cluster_count: int) -> np.ndarray:
    selected = [0]
    while len(selected) < int(cluster_count):
        current = vectors[selected]
        distances = _squared_distances(vectors, current)
        nearest = np.min(distances, axis=1)
        for index in selected:
            nearest[index] = -1.0
        best_distance = float(np.max(nearest))
        candidate_indexes = [index for index, distance in enumerate(nearest) if float(distance) == best_distance]
        selected.append(min(candidate_indexes, key=lambda index: sample_ids[index]))
    return vectors[selected].copy()


def _updated_centroids(vectors: np.ndarray, assignments: np.ndarray, previous: np.ndarray) -> np.ndarray:
    next_centroids = previous.copy()
    for cluster_index in range(previous.shape[0]):
        members = vectors[assignments == cluster_index]
        if len(members):
            centroid = np.mean(members, axis=0, dtype=np.float32)
            norm = float(np.linalg.norm(centroid))
            next_centroids[cluster_index] = centroid / norm if norm > 1e-12 else centroid
    return next_centroids.astype(np.float32)


def _squared_distances(vectors: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    diff = vectors[:, None, :] - centroids[None, :, :]
    return np.sum(diff * diff, axis=2)


def _soft_cluster_probabilities(distances: np.ndarray, *, temperature: float) -> np.ndarray:
    if temperature <= 0.0:
        hard = np.zeros_like(distances, dtype=np.float32)
        hard[np.arange(distances.shape[0]), np.argmin(distances, axis=1)] = 1.0
        return hard
    logits = -distances / float(temperature)
    logits = logits - np.max(logits, axis=1, keepdims=True)
    exp_values = np.exp(logits)
    return (exp_values / np.sum(exp_values, axis=1, keepdims=True)).astype(np.float32)
