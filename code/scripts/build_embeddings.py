
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.embedding_text import (  # noqa: E402
    chunk_text_for_embedding,
    format_pair_embedding_text,
    mean_pool_vectors,
    normalize_vectors,
)
from src.utils import resolve_input_path, resolve_output_path, write_json  # noqa: E402


DEFAULT_EMBEDDING_MODEL = "models/embedding-model"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--ids_path", default=None)
    parser.add_argument("--meta_path", default=None)
    parser.add_argument("--model_path", default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument("--backend", choices=["transformers", "vllm"], default="transformers")
    parser.add_argument("--request_batch_size", type=int, default=16)
    parser.add_argument("--flush_rows", type=int, default=256)
    parser.add_argument("--max_length", type=int, default=4096)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--torch_dtype", choices=["auto", "bfloat16", "float16", "float32"], default="bfloat16")
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.92)
    parser.add_argument("--enforce_eager", action="store_true")
    parser.add_argument("--query_field", default="query")
    parser.add_argument("--document_field", default="document")
    parser.add_argument("--id_field", default="id")
    parser.add_argument("--mode", choices=["document", "chunk"], default="document")
    parser.add_argument("--target_chars", type=int, default=3000)
    parser.add_argument("--overlap_chars", type=int, default=300)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def resolve_device(device_name: str) -> torch.device:
    if str(device_name) == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(str(device_name))


def resolve_torch_dtype(dtype_name: str, device: torch.device) -> torch.dtype:
    if device.type != "cuda":
        return torch.float32
    if dtype_name == "auto":
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    if dtype_name == "bfloat16":
        return torch.bfloat16
    if dtype_name == "float16":
        return torch.float16
    if dtype_name == "float32":
        return torch.float32
    raise ValueError(f"unsupported torch dtype: {dtype_name}")


def resolve_vllm_dtype(dtype_name: str) -> str:
    if dtype_name == "auto":
        return "auto"
    if dtype_name == "bfloat16":
        return "bfloat16"
    if dtype_name == "float16":
        return "float16"
    if dtype_name == "float32":
        return "float32"
    raise ValueError(f"unsupported vLLM dtype: {dtype_name}")


def last_token_pool(hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    if hidden_states.ndim != 3:
        raise ValueError(f"hidden_states must be rank 3, got shape {tuple(hidden_states.shape)}")
    if attention_mask.ndim != 2:
        raise ValueError(f"attention_mask must be rank 2, got shape {tuple(attention_mask.shape)}")
    sequence_lengths = attention_mask.to(dtype=torch.long).sum(dim=1).clamp(min=1) - 1
    batch_indices = torch.arange(hidden_states.size(0), device=hidden_states.device)
    return hidden_states[batch_indices, sequence_lengths.to(device=hidden_states.device)]


def load_transformers_embedding_model(
    model_path: str | Path,
    *,
    device_name: str,
    dtype_name: str,
) -> tuple[Any, torch.nn.Module, torch.device, bool, torch.dtype]:
    from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer

    device = resolve_device(device_name)
    dtype = resolve_torch_dtype(dtype_name, device)
    tokenizer = AutoTokenizer.from_pretrained(
        str(model_path),
        trust_remote_code=True,
        local_files_only=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.bos_token
    tokenizer.padding_side = "right"

    model_kwargs = {
        "trust_remote_code": True,
        "local_files_only": True,
        "torch_dtype": dtype,
    }
    try:
        model = AutoModel.from_pretrained(str(model_path), **model_kwargs)
        uses_lm_hidden_states = False
    except Exception:
        model = AutoModelForCausalLM.from_pretrained(str(model_path), **model_kwargs)
        uses_lm_hidden_states = True
    model.to(device)
    model.eval()
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False
    return tokenizer, model, device, uses_lm_hidden_states, dtype


def load_vllm_embedding_model(
    model_path: str | Path,
    *,
    dtype_name: str,
    max_length: int,
    tensor_parallel_size: int,
    gpu_memory_utilization: float,
    enforce_eager: bool,
) -> Any:
    from vllm import LLM

    return LLM(
        model=str(model_path),
        runner="pooling",
        convert="embed",
        trust_remote_code=True,
        dtype=resolve_vllm_dtype(dtype_name),
        tensor_parallel_size=int(tensor_parallel_size),
        gpu_memory_utilization=float(gpu_memory_utilization),
        enforce_eager=bool(enforce_eager),
        max_model_len=int(max_length),
    )


def embed_texts_transformers(
    *,
    model: torch.nn.Module,
    tokenizer: Any,
    device: torch.device,
    texts: list[str],
    batch_size: int,
    max_length: int,
    uses_lm_hidden_states: bool,
) -> np.ndarray:
    vectors: list[np.ndarray] = []
    safe_batch_size = max(1, int(batch_size))
    for start in range(0, len(texts), safe_batch_size):
        batch_texts = texts[start : start + safe_batch_size]
        encoded = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=int(max_length),
            return_tensors="pt",
        )
        encoded = {key: value.to(device) for key, value in encoded.items()}
        with torch.inference_mode():
            outputs = model(
                **encoded,
                output_hidden_states=uses_lm_hidden_states,
                use_cache=False,
                return_dict=True,
            )
            hidden_states = getattr(outputs, "last_hidden_state", None)
            if hidden_states is None:
                hidden_states = outputs.hidden_states[-1]
            pooled = last_token_pool(hidden_states, encoded["attention_mask"])
            pooled = torch.nn.functional.normalize(pooled.float(), p=2, dim=1)
        vectors.append(pooled.cpu().numpy().astype(np.float32))
        del encoded, outputs, hidden_states, pooled
    if not vectors:
        return np.empty((0, 0), dtype=np.float32)
    return np.vstack(vectors).astype(np.float32)


def embed_texts_vllm(
    *,
    model: Any,
    texts: list[str],
    batch_size: int,
) -> np.ndarray:
    vectors: list[np.ndarray] = []
    safe_batch_size = max(1, int(batch_size))
    for start in range(0, len(texts), safe_batch_size):
        batch_texts = texts[start : start + safe_batch_size]
        outputs = model.embed(batch_texts, use_tqdm=False)
        if len(outputs) != len(batch_texts):
            raise RuntimeError(f"embedding response mismatch: expected {len(batch_texts)} got {len(outputs)}")
        batch_vectors = np.asarray([item.outputs.embedding for item in outputs], dtype=np.float32)
        if batch_vectors.ndim != 2 or batch_vectors.shape[0] != len(batch_texts):
            raise RuntimeError(f"embedding response shape mismatch: expected {len(batch_texts)} got {batch_vectors.shape}")
        norms = np.linalg.norm(batch_vectors, axis=1, keepdims=True)
        norms = np.where(norms > 0, norms, 1.0).astype(np.float32)
        vectors.append((batch_vectors / norms).astype(np.float32))
    if not vectors:
        return np.empty((0, 0), dtype=np.float32)
    return np.vstack(vectors).astype(np.float32)


def read_jsonl(path: Path, *, limit: int | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            rows.append(json.loads(line))
            if limit is not None and len(rows) >= int(limit):
                break
    return rows


def pair_texts_for_row(
    row: dict[str, Any],
    *,
    query_field: str,
    document_field: str,
    mode: str,
    target_chars: int,
    overlap_chars: int,
) -> list[str]:
    query = str(row[query_field])
    document = str(row[document_field])
    if str(mode) == "document":
        return [format_pair_embedding_text(document, query)]
    chunks = chunk_text_for_embedding(
        document,
        target_chars=int(target_chars),
        overlap_chars=int(overlap_chars),
    )
    return [format_pair_embedding_text(chunk, query) for chunk in chunks]


def _read_completed_embedding_ids(ids_path: Path) -> list[str]:
    if not ids_path.exists():
        return []
    completed: list[str] = []
    with ids_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            payload = json.loads(line)
            if "id" not in payload:
                raise ValueError(f"embedding id cache missing id at {ids_path}:{line_number}")
            completed.append(str(payload["id"]))
    return completed


def _verify_resume_prefix(rows: list[dict[str, Any]], completed_ids: list[str], *, id_field: str) -> None:
    if len(completed_ids) > len(rows):
        raise ValueError(f"embedding id cache has {len(completed_ids)} rows, but data only has {len(rows)} rows")
    expected_ids = [str(row[id_field]) for row in rows[: len(completed_ids)]]
    if completed_ids != expected_ids:
        raise ValueError("embedding id cache is not a prefix of the current data; pass --overwrite to rebuild")


def build_embeddings(args: argparse.Namespace) -> dict[str, Any]:
    data_path = resolve_input_path(args.data_path, PROJECT_ROOT)
    output_path = resolve_output_path(args.output_path, PROJECT_ROOT)
    ids_path = resolve_output_path(args.ids_path, PROJECT_ROOT) if args.ids_path else output_path.with_suffix(".ids.jsonl")
    meta_path = resolve_output_path(args.meta_path, PROJECT_ROOT) if args.meta_path else output_path.with_suffix(".meta.json")
    if bool(args.overwrite):
        for path in (output_path, ids_path, meta_path):
            if path.exists():
                path.unlink()

    rows = read_jsonl(data_path, limit=args.limit)
    if not rows:
        raise ValueError(f"no rows loaded from {data_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ids_path.parent.mkdir(parents=True, exist_ok=True)

    total_rows = len(rows)
    completed_ids = _read_completed_embedding_ids(ids_path)
    if output_path.exists() and not completed_ids and not bool(args.overwrite):
        raise FileExistsError(f"embedding matrix exists without usable id cache: {output_path}; pass --overwrite to rebuild")
    _verify_resume_prefix(rows, completed_ids, id_field=str(args.id_field))
    pending_texts: list[str] = []
    pending_spans: list[tuple[int, int]] = []
    pending_ids: list[str] = []
    cursor = 0
    row_cursor = len(completed_ids)
    memmap: np.memmap | None = None
    dimension: int | None = None
    request_count = 0
    input_count = 0
    started_at = time.time()
    backend = str(args.backend)
    tokenizer: Any | None = None
    uses_lm_hidden_states = False
    device: torch.device | None = None
    dtype_label = resolve_vllm_dtype(str(args.torch_dtype))
    if backend == "transformers":
        tokenizer, model, device, uses_lm_hidden_states, dtype = load_transformers_embedding_model(
            args.model_path,
            device_name=str(args.device),
            dtype_name=str(args.torch_dtype),
        )
        dtype_label = str(dtype).replace("torch.", "")
        device_label = str(device)
    elif backend == "vllm":
        model = load_vllm_embedding_model(
            args.model_path,
            dtype_name=str(args.torch_dtype),
            max_length=int(args.max_length),
            tensor_parallel_size=int(args.tensor_parallel_size),
            gpu_memory_utilization=float(args.gpu_memory_utilization),
            enforce_eager=bool(args.enforce_eager),
        )
        device_label = "vllm"
    else:
        raise ValueError(f"unsupported embedding backend: {backend}")
    if row_cursor:
        if not output_path.exists():
            raise FileExistsError(f"embedding id cache exists but matrix is missing: {output_path}")
        memmap = np.load(output_path, mmap_mode="r+")
        if memmap.ndim != 2:
            raise ValueError(f"embedding matrix must be rank 2, got shape {tuple(memmap.shape)}")
        if int(memmap.shape[0]) != int(total_rows):
            raise ValueError(f"embedding matrix row count mismatch: expected {total_rows} got {memmap.shape[0]}")
        dimension = int(memmap.shape[1])

    progress = tqdm(total=total_rows, desc="embeddings", unit="row")
    if row_cursor:
        progress.update(row_cursor)

    def flush_pending() -> None:
        nonlocal pending_texts, pending_spans, pending_ids, cursor, row_cursor, memmap, dimension
        nonlocal request_count, input_count
        if not pending_texts:
            return
        batch_texts = list(pending_texts)
        batch_spans = list(pending_spans)
        batch_ids = list(pending_ids)
        pending_texts = []
        pending_spans = []
        pending_ids = []
        cursor = 0

        request_count += int(math.ceil(len(batch_texts) / max(1, int(args.request_batch_size))))
        input_count += len(batch_texts)
        if backend == "transformers":
            assert tokenizer is not None
            assert device is not None
            flat_vectors = embed_texts_transformers(
                model=model,
                tokenizer=tokenizer,
                device=device,
                texts=batch_texts,
                batch_size=int(args.request_batch_size),
                max_length=int(args.max_length),
                uses_lm_hidden_states=uses_lm_hidden_states,
            )
        else:
            flat_vectors = embed_texts_vllm(
                model=model,
                texts=batch_texts,
                batch_size=int(args.request_batch_size),
            )
        flat_vectors = np.asarray(flat_vectors, dtype=np.float32)
        if flat_vectors.ndim != 2 or flat_vectors.shape[0] != len(batch_texts):
            raise RuntimeError(f"embedding response mismatch: expected {len(batch_texts)} got {flat_vectors.shape}")
        pooled_rows = [mean_pool_vectors(flat_vectors[start:end]) for start, end in batch_spans]
        batch_rows = normalize_vectors(np.vstack(pooled_rows).astype(np.float32))
        if dimension is None:
            dimension = int(batch_rows.shape[1])
            memmap = np.lib.format.open_memmap(
                output_path,
                mode="w+",
                dtype=np.float32,
                shape=(total_rows, dimension),
            )
        elif int(batch_rows.shape[1]) != int(dimension):
            raise RuntimeError(f"embedding dimension changed: expected {dimension} got {batch_rows.shape[1]}")
        assert memmap is not None
        end = row_cursor + len(batch_rows)
        memmap[row_cursor:end, :] = batch_rows
        with ids_path.open("a", encoding="utf-8") as handle:
            for sample_id in batch_ids:
                handle.write(json.dumps({"id": sample_id}, ensure_ascii=False) + "\n")
            handle.flush()
        memmap.flush()
        row_cursor = end
        progress.update(len(batch_rows))

    # `ids_path` stores the flushed prefix, so restarts continue after it.
    pending_text_limit = int(args.request_batch_size) if backend == "transformers" else int(args.flush_rows)
    for row in rows[row_cursor:]:
        sample_id = str(row[args.id_field])
        formatted = pair_texts_for_row(
            row,
            query_field=str(args.query_field),
            document_field=str(args.document_field),
            mode=str(args.mode),
            target_chars=int(args.target_chars),
            overlap_chars=int(args.overlap_chars),
        )
        if pending_ids and (len(pending_ids) >= int(args.flush_rows) or len(pending_texts) + len(formatted) > pending_text_limit):
            flush_pending()
        start = cursor
        pending_texts.extend(formatted)
        cursor += len(formatted)
        pending_spans.append((start, cursor))
        pending_ids.append(sample_id)
    flush_pending()
    progress.close()

    if memmap is not None:
        memmap.flush()
        del memmap
    del model
    if (device is not None and device.type == "cuda") or (backend == "vllm" and torch.cuda.is_available()):
        torch.cuda.empty_cache()
    if row_cursor != total_rows:
        raise RuntimeError(f"embedding row count mismatch: expected {total_rows} wrote {row_cursor}")
    meta = {
        "data_path": str(data_path),
        "embedding_model": str(args.model_path),
        "embedding_backend": backend,
        "mode": str(args.mode),
        "embedding_text_format": "Query:\\n{query}\\n\\nDocument:\\n{document}",
        "pair_embedding_version": "query_document",
        "row_count": int(total_rows),
        "dimension": int(dimension or 0),
        "max_length": int(args.max_length),
        "device": device_label,
        "torch_dtype": dtype_label,
        "query_field": str(args.query_field),
        "document_field": str(args.document_field),
        "id_field": str(args.id_field),
        "target_chars": int(args.target_chars),
        "overlap_chars": int(args.overlap_chars),
        "request_batch_size": int(args.request_batch_size),
        "flush_rows": int(args.flush_rows),
        "tensor_parallel_size": int(args.tensor_parallel_size),
        "gpu_memory_utilization": float(args.gpu_memory_utilization),
        "enforce_eager": bool(args.enforce_eager),
        "request_count": int(request_count),
        "input_count": int(input_count),
        "elapsed_seconds": float(time.time() - started_at),
    }
    write_json(meta, meta_path)
    return meta


def main() -> None:
    meta = build_embeddings(parse_args())
    print(json.dumps(meta, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
