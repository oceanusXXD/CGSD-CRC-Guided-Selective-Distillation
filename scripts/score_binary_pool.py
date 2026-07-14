from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mias_dcms.binary_protocol import BINARY_NEGATIVE_TEXT, BINARY_POSITIVE_TEXT  # noqa: E402
from mias_dcms.data import format_binary_chat_prompt  # noqa: E402
from mias_dcms.model import QwenGenerativeModel  # noqa: E402
from mias_dcms.selectors import assert_selector_rows_are_label_safe  # noqa: E402
from mias_dcms.trainer import get_single_token_id, last_non_padding_logits  # noqa: E402
from mias_dcms.utils import (  # noqa: E402
    configure_torch_performance,
    disable_tokenizer_thinking,
    ensure_tokenizer_padding,
    get_device,
    parse_torch_dtype,
    read_jsonl,
    write_json,
    write_jsonl,
)


class SelectorPoolDataset(Dataset):
    def __init__(self, rows: list[dict[str, Any]], tokenizer: Any, max_length: int) -> None:
        self.rows = rows
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        encoded = self.tokenizer(
            format_binary_chat_prompt(str(row["query"]), str(row["document"])),
            truncation=True,
            max_length=self.max_length,
            padding=False,
            add_special_tokens=False,
        )
        return {"id": str(row["id"]), "input_ids": list(encoded["input_ids"]), "attention_mask": list(encoded["attention_mask"])}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Score a label-safe binary selection pool with a local LoRA checkpoint."
    )
    parser.add_argument("--checkpoint_dir", type=Path, required=True)
    parser.add_argument("--input_path", type=Path, required=True)
    parser.add_argument("--output_path", type=Path, required=True)
    parser.add_argument("--summary_path", type=Path)
    parser.add_argument("--model_path", type=Path)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--torch_dtype", choices=["auto", "none", "float16", "bfloat16", "float32"], default="auto")
    parser.add_argument("--no_tf32", dest="tf32", action="store_false", default=True)
    return parser.parse_args()


def _collate(tokenizer: Any, features: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "ids": [feature["id"] for feature in features],
        "tokens": tokenizer.pad(
            [{"input_ids": feature["input_ids"], "attention_mask": feature["attention_mask"]} for feature in features],
            padding=True,
            pad_to_multiple_of=8,
            return_tensors="pt",
        ),
    }


def _load_safe_rows(path: Path) -> list[dict[str, Any]]:
    rows = [dict(row) for row in read_jsonl(path)]
    if not rows:
        raise ValueError("selection pool is empty")
    ids: set[str] = set()
    for index, row in enumerate(rows):
        sample_id = str(row.get("id", ""))
        if not sample_id or sample_id in ids:
            raise ValueError(f"selection pool row {index} has a missing or duplicate id")
        if not str(row.get("query", "")).strip() or not str(row.get("document", "")).strip():
            raise ValueError(f"selection pool row {sample_id!r} has a missing query or document")
        ids.add(sample_id)
    assert_selector_rows_are_label_safe(rows)
    return rows


@torch.inference_mode()
def score_rows(
    *,
    rows: list[dict[str, Any]],
    checkpoint_dir: Path,
    model_path: Path | None,
    max_length: int,
    batch_size: int,
    num_workers: int,
    device: torch.device,
    torch_dtype: Any,
) -> list[dict[str, Any]]:
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir, trust_remote_code=True, local_files_only=True)
    ensure_tokenizer_padding(tokenizer)
    disable_tokenizer_thinking(tokenizer)
    loader = DataLoader(
        SelectorPoolDataset(rows, tokenizer, max_length=max_length),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=num_workers > 0,
        collate_fn=lambda features: _collate(tokenizer, features),
    )
    model = QwenGenerativeModel.load_from_checkpoint(
        checkpoint_dir,
        torch_dtype=torch_dtype,
        map_location="cpu",
        model_path=model_path,
    )
    model.to(device)
    negative_token_id = get_single_token_id(tokenizer, BINARY_NEGATIVE_TEXT)
    positive_token_id = get_single_token_id(tokenizer, BINARY_POSITIVE_TEXT)
    rows_by_id = {str(row["id"]): row for row in rows}
    scored: list[dict[str, Any]] = []
    for batch in loader:
        tokens = {key: value.to(device, non_blocking=True) for key, value in batch["tokens"].items()}
        outputs = model(input_ids=tokens["input_ids"], attention_mask=tokens["attention_mask"])
        next_logits = last_non_padding_logits(outputs.logits, tokens["attention_mask"])
        binary_logits = next_logits[:, [negative_token_id, positive_token_id]].float()
        probabilities = torch.softmax(binary_logits, dim=-1).cpu().tolist()
        logits = binary_logits.cpu().tolist()
        for sample_id, row_probabilities, row_logits in zip(batch["ids"], probabilities, logits):
            p0, p1 = (float(row_probabilities[0]), float(row_probabilities[1]))
            scored.append(
                {
                    **rows_by_id[str(sample_id)],
                    "id": str(sample_id),
                    "probabilities": [p0, p1],
                    "entropy": float(
                        -(p0 * math.log(max(p0, 1e-12)) + p1 * math.log(max(p1, 1e-12)))
                    ),
                    "margin": float(abs(p1 - p0)),
                    "score": float(row_logits[1] - row_logits[0]),
                    "zero_logit": float(row_logits[0]),
                    "one_logit": float(row_logits[1]),
                    "score_source": "binary_next_token_logit_margin",
                }
            )
    model.to("cpu")
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return scored


def main() -> None:
    args = parse_args()
    if args.max_length < 8 or args.batch_size < 1 or args.num_workers < 0:
        raise ValueError("max_length must be at least 8, batch_size positive, and num_workers non-negative")
    configure_torch_performance(enable_tf32=bool(args.tf32))
    rows = _load_safe_rows(args.input_path)
    device = get_device(args.device)
    scored = score_rows(
        rows=rows,
        checkpoint_dir=args.checkpoint_dir,
        model_path=args.model_path,
        max_length=int(args.max_length),
        batch_size=int(args.batch_size),
        num_workers=int(args.num_workers),
        device=device,
        torch_dtype=parse_torch_dtype(args.torch_dtype),
    )
    write_jsonl(scored, args.output_path)
    summary = {
        "input_path": str(args.input_path),
        "checkpoint_dir": str(args.checkpoint_dir),
        "output_path": str(args.output_path),
        "size": len(scored),
        "score_source": "binary_next_token_logit_margin",
    }
    write_json(summary, args.summary_path or args.output_path.with_suffix(".summary.json"))
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
