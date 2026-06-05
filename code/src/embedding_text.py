"""Text formatting, chunking, and pooling helpers for embedding jobs."""

from __future__ import annotations

import re
from typing import Any

import numpy as np


_SENTENCE_BOUNDARY_RE = re.compile(r"(?<=[.!?。！？])\s+")


def split_into_sentences(text: Any) -> list[str]:
    normalized = str(text or "").strip()
    if not normalized:
        return []
    try:
        from blingfire import text_to_sentences
    except ImportError:
        return [part.strip() for part in _SENTENCE_BOUNDARY_RE.split(normalized) if part.strip()]

    segmented = str(text_to_sentences(normalized) or "").strip()
    if not segmented:
        return []
    return [line.strip() for line in segmented.splitlines() if line.strip()]


def chunk_text_for_embedding(
    text: str,
    *,
    target_chars: int = 3000,
    overlap_chars: int = 300,
) -> list[str]:
    sentences = split_into_sentences(text)
    if not sentences:
        normalized = str(text or "").strip()
        return [normalized] if normalized else [""]

    chunks: list[str] = []
    current: list[str] = []
    current_len = 0
    for sentence in sentences:
        sentence_len = len(sentence)
        if current and current_len + sentence_len > target_chars:
            chunk = " ".join(current).strip()
            if chunk:
                chunks.append(chunk)
            overlap_text = chunk[-overlap_chars:].strip() if overlap_chars > 0 else ""
            current = [overlap_text] if overlap_text else []
            current_len = len(overlap_text)
        current.append(sentence)
        current_len += sentence_len + 1
    if current:
        chunk = " ".join(current).strip()
        if chunk:
            chunks.append(chunk)
    return chunks or [str(text or "").strip()]


def format_pair_embedding_text(
    text: str,
    query: str,
    *,
    task_label: str = "Query",
    evidence_label: str = "Document",
    instruction_text: str = "",
) -> str:
    query_text = str(query or "").strip()
    document_text = str(text or "").strip()
    instruction_block = ""
    if str(instruction_text or "").strip():
        instruction_block = f"Instruction:\n{str(instruction_text or '').strip()}\n\n"
    return f"{instruction_block}{task_label}:\n{query_text}\n\n{evidence_label}:\n{document_text}"


def mean_pool_vectors(vectors: np.ndarray) -> np.ndarray:
    if vectors.ndim != 2 or vectors.shape[0] == 0:
        return np.zeros((0,), dtype=np.float32)
    pooled = np.mean(vectors, axis=0, dtype=np.float32)
    norm = float(np.linalg.norm(pooled))
    if norm <= 1e-12:
        return pooled.astype(np.float32)
    return (pooled / norm).astype(np.float32)


def normalize_vectors(arr: np.ndarray) -> np.ndarray:
    if arr.size == 0:
        return arr.astype(np.float32)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms[norms <= 1e-12] = 1.0
    return (arr / norms).astype(np.float32)
