"""Shared utilities for training, prediction, and evaluation.

This module contains basic functionality reused across scripts: random seeding,
device selection, JSON/JSONL I/O, torch dtype parsing, batch device transfer,
and output-directory resolution. Business workflow logic should live in the
stage-specific modules instead.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(device_name: str = "auto") -> torch.device:
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_name)


def configure_torch_performance(enable_tf32: bool = True) -> None:
    if not torch.cuda.is_available():
        return
    if enable_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        try:
            torch.set_float32_matmul_precision("high")
        except AttributeError:
            pass


def parse_torch_dtype(dtype_name: str) -> torch.dtype | str | None:
    if dtype_name == "auto":
        return "auto"
    if dtype_name == "none":
        return None
    if dtype_name == "float16":
        return torch.float16
    if dtype_name == "bfloat16":
        return torch.bfloat16
    if dtype_name == "float32":
        return torch.float32
    raise ValueError(f"Unsupported torch dtype: {dtype_name}")


def ensure_tokenizer_padding(tokenizer: Any) -> None:
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is None:
            raise ValueError("Tokenizer has neither pad_token nor eos_token.")
        tokenizer.pad_token = tokenizer.eos_token
    if hasattr(tokenizer, "padding_side"):
        tokenizer.padding_side = "left"


def disable_tokenizer_thinking(tokenizer: Any) -> None:
    """Remove a thinking chat template that may remain in a checkpoint tokenizer.

    Inference constructs the query-document prompt directly and does not depend
    on the tokenizer chat template. Clearing the template avoids accidentally
    enabling thinking mode when a checkpoint tokenizer is reused.
    """
    if hasattr(tokenizer, "chat_template"):
        tokenizer.chat_template = None
    init_kwargs = getattr(tokenizer, "init_kwargs", None)
    if isinstance(init_kwargs, dict):
        init_kwargs.pop("chat_template", None)


def move_batch_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    moved: dict[str, Any] = {}
    for key, value in batch.items():
        moved[key] = value.to(device, non_blocking=True) if torch.is_tensor(value) else value
    return moved


def read_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_json(data: dict[str, Any], path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def write_jsonl(rows: list[dict[str, Any]], path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False))
            handle.write("\n")


def resolve_input_path(path: str | Path, project_root: Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    if candidate.parts and candidate.parts[0] == project_root.name:
        return project_root.parent / candidate
    return project_root / candidate


def resolve_output_path(path: str | Path, project_root: Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    if candidate.parts and candidate.parts[0] == project_root.name:
        return project_root.parent / candidate
    return project_root / candidate


def count_parameters(module: torch.nn.Module) -> dict[str, int]:
    total = sum(param.numel() for param in module.parameters())
    trainable = sum(param.numel() for param in module.parameters() if param.requires_grad)
    return {"total": total, "trainable": trainable}
