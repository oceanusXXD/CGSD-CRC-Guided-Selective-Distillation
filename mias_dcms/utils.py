from __future__ import annotations
import json
import os
import random
from pathlib import Path
from typing import Any


def _require_numpy() -> Any:
    try:
        import numpy as np
    except ModuleNotFoundError as exc:
        raise RuntimeError('NumPy is required for reproducible model seeding') from exc
    return np


def _require_torch() -> Any:
    try:
        import torch
    except ModuleNotFoundError as exc:
        raise RuntimeError('PyTorch is required for model runtime utilities') from exc
    return torch

def set_seed(seed: int) -> None:
    random.seed(seed)
    try:
        _require_numpy().random.seed(seed)
    except RuntimeError:
        pass
    try:
        torch = _require_torch()
    except RuntimeError:
        return
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def get_device(device_name: str='auto') -> torch.device:
    torch = _require_torch()
    if device_name == 'auto':
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return torch.device(device_name)

def configure_torch_performance(enable_tf32: bool=True) -> None:
    torch = _require_torch()
    if not torch.cuda.is_available():
        return
    if enable_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        if hasattr(torch, 'set_float32_matmul_precision'):
            torch.set_float32_matmul_precision('high')

def parse_torch_dtype(dtype_name: str) -> torch.dtype | str | None:
    if dtype_name == 'auto':
        return 'auto'
    if dtype_name == 'none':
        return None
    torch = _require_torch()
    if dtype_name == 'float16':
        return torch.float16
    if dtype_name == 'bfloat16':
        return torch.bfloat16
    if dtype_name == 'float32':
        return torch.float32
    raise ValueError(f'Unsupported torch dtype: {dtype_name}')

def ensure_tokenizer_padding(tokenizer: Any) -> None:
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is None:
            raise ValueError('Tokenizer has neither pad_token nor eos_token.')
        tokenizer.pad_token = tokenizer.eos_token
    if hasattr(tokenizer, 'padding_side'):
        tokenizer.padding_side = 'left'

def disable_tokenizer_thinking(tokenizer: Any) -> None:
    if hasattr(tokenizer, 'chat_template'):
        tokenizer.chat_template = None
    init_kwargs = getattr(tokenizer, 'init_kwargs', None)
    if isinstance(init_kwargs, dict):
        init_kwargs.pop('chat_template', None)

def move_batch_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    torch = _require_torch()
    moved: dict[str, Any] = {}
    for key, value in batch.items():
        moved[key] = value.to(device, non_blocking=True) if torch.is_tensor(value) else value
    return moved

def read_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open('r', encoding='utf-8') as handle:
        return json.load(handle)

def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open('r', encoding='utf-8') as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows

def write_json(data: dict[str, Any], path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open('w', encoding='utf-8') as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write('\n')

def write_jsonl(rows: list[dict[str, Any]], path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open('w', encoding='utf-8') as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False))
            handle.write('\n')

def resolve_input_path(path: str | Path, project_root: Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute() or candidate.exists():
        return candidate
    project_candidate = project_root / candidate
    if project_candidate.exists():
        return project_candidate
    workspace_candidate = project_root.parent / candidate
    if workspace_candidate.exists():
        return workspace_candidate
    return project_candidate

def resolve_output_path(path: str | Path, project_root: Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    if candidate.parts and candidate.parts[0] == project_root.name:
        return project_root.parent / candidate
    return project_root / candidate


def resolve_model_reference(value: str | Path, project_root: Path) -> str:
    """Resolve portable model aliases without making machine paths part of configs."""
    raw = str(value)
    candidate = Path(raw)
    candidates: list[Path] = []
    if candidate.is_absolute():
        candidates.append(candidate)
    else:
        candidates.extend(
            [
                candidate,
                project_root / candidate,
                project_root.parent / candidate,
                project_root / "model" / candidate,
                project_root / "models" / candidate,
                project_root.parent / "models" / candidate,
            ]
        )
        model_root = os.environ.get("MIAS_DCMS_MODEL_ROOT")
        if model_root:
            candidates.append(Path(model_root) / candidate)
    for path in candidates:
        if path.exists():
            return str(path)
    return raw

def count_parameters(module: torch.nn.Module) -> dict[str, int]:
    total = sum((param.numel() for param in module.parameters()))
    trainable = sum((param.numel() for param in module.parameters() if param.requires_grad))
    return {'total': total, 'trainable': trainable}
