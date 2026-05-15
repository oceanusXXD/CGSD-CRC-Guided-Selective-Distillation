#!/usr/bin/env python
"""Evaluate a saved query-document relevance checkpoint."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from torch.utils.data import DataLoader
from transformers import AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data import (
    GenerationPairCollator,
    GenerationQueryDocumentDataset,
    TokenBudgetBatchSampler,
    filter_examples_by_ids,
    load_examples,
)
from src.model import QwenGenerativeModel
from src.trainer import evaluate_model
from src.utils import (
    configure_torch_performance,
    disable_tokenizer_thinking,
    ensure_tokenizer_padding,
    get_device,
    parse_torch_dtype,
    read_json,
    resolve_input_path,
    resolve_output_path,
)


INPUT_FORMAT = "query_document_marked_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint_dir", required=True)
    parser.add_argument("--data_path", default="datasets")
    parser.add_argument("--model_path", default=None)
    parser.add_argument("--query_field", default="query")
    parser.add_argument("--document_field", default="document")
    parser.add_argument("--label_field", default="groundtruth")
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--max_tokens_per_batch", type=int, default=8192)
    parser.add_argument("--threshold", type=float, default=0.0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--prefetch_factor", type=int, default=2)
    parser.add_argument("--pad_to_multiple_of", type=int, default=8)
    parser.add_argument("--cache_tokenization", action="store_true", default=True)
    parser.add_argument("--no_cache_tokenization", dest="cache_tokenization", action="store_false")
    parser.add_argument("--pin_memory", action="store_true", default=True)
    parser.add_argument("--no_pin_memory", dest="pin_memory", action="store_false")
    parser.add_argument("--tf32", action="store_true", default=True)
    parser.add_argument("--no_tf32", dest="tf32", action="store_false")
    parser.add_argument(
        "--torch_dtype",
        choices=["auto", "none", "float16", "bfloat16", "float32"],
        default="auto",
    )
    parser.add_argument("--predictions_path", default=None)
    parser.add_argument("--metrics_path", default=None)
    parser.add_argument("--split_ids_path", default=None)
    parser.add_argument("--split_name", choices=["all", "train", "val", "test"], default="all")
    return parser.parse_args()


def build_dataloader(
    dataset: GenerationQueryDocumentDataset,
    batch_size: int,
    collator: GenerationPairCollator,
    num_workers: int,
    prefetch_factor: int,
    pin_memory: bool,
    max_tokens_per_batch: int = 0,
) -> DataLoader:
    """Create a DataLoader with fast CUDA input settings when available."""
    kwargs = {
        "dataset": dataset,
        "collate_fn": collator,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
    }
    if max_tokens_per_batch > 0:
        kwargs["batch_sampler"] = TokenBudgetBatchSampler(
            lengths=dataset.sequence_lengths(),
            max_batch_size=batch_size,
            max_tokens_per_batch=max_tokens_per_batch,
            shuffle=False,
        )
    else:
        kwargs["batch_size"] = batch_size
        kwargs["shuffle"] = False
    if num_workers > 0:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = prefetch_factor
    return DataLoader(**kwargs)


def load_tokenizer(checkpoint_dir: Path, model_path: str | Path | None) -> AutoTokenizer:
    """Load tokenizer from checkpoint if available, otherwise from model path."""
    source = checkpoint_dir
    try:
        # Training saves tokenizer files beside the adapter checkpoint.
        tokenizer = AutoTokenizer.from_pretrained(
            source,
            trust_remote_code=True,
            local_files_only=True,
        )
    except OSError:
        if model_path is None:
            config = read_json(checkpoint_dir / "model_config.json")
            model_path = config["model_path"]
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
            local_files_only=True,
        )
    ensure_tokenizer_padding(tokenizer)
    disable_tokenizer_thinking(tokenizer)
    return tokenizer


def main() -> None:
    args = parse_args()
    configure_torch_performance(enable_tf32=args.tf32)
    checkpoint_dir = resolve_input_path(args.checkpoint_dir, PROJECT_ROOT)
    data_path = resolve_input_path(args.data_path, PROJECT_ROOT)
    model_path = (
        resolve_input_path(args.model_path, PROJECT_ROOT)
        if args.model_path is not None
        else None
    )
    predictions_path = (
        resolve_output_path(args.predictions_path, PROJECT_ROOT)
        if args.predictions_path is not None
        else None
    )
    metrics_path = (
        resolve_output_path(args.metrics_path, PROJECT_ROOT)
        if args.metrics_path is not None
        else None
    )
    split_ids_path = (
        resolve_input_path(args.split_ids_path, PROJECT_ROOT)
        if args.split_ids_path is not None
        else None
    )

    tokenizer = load_tokenizer(checkpoint_dir, model_path)
    checkpoint_config = read_json(checkpoint_dir / "model_config.json")
    input_format = str(checkpoint_config.get("input_format", INPUT_FORMAT))
    examples = load_examples(
        data_path,
        query_field=args.query_field,
        document_field=args.document_field,
        label_field=args.label_field,
    )
    if args.split_name != "all":
        if split_ids_path is None:
            split_ids_path = checkpoint_dir / "split_ids.json"
        split_data = read_json(split_ids_path)
        split_ids = set(str(sample_id) for sample_id in split_data[f"{args.split_name}_ids"])
        examples = filter_examples_by_ids(examples, split_ids)

    device = get_device(args.device)
    dataset = GenerationQueryDocumentDataset(
        examples,
        tokenizer=tokenizer,
        max_length=args.max_length,
        cache_tokenization=args.cache_tokenization,
        input_format=input_format,
    )
    dataloader = build_dataloader(
        dataset=dataset,
        batch_size=args.batch_size,
        collator=GenerationPairCollator(
            tokenizer,
            pad_to_multiple_of=args.pad_to_multiple_of if args.pad_to_multiple_of > 0 else None,
        ),
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor,
        pin_memory=args.pin_memory and device.type == "cuda",
        max_tokens_per_batch=args.max_tokens_per_batch,
    )

    model = QwenGenerativeModel.load_from_checkpoint(
        checkpoint_dir,
        torch_dtype=parse_torch_dtype(args.torch_dtype),
        map_location="cpu",
        model_path=model_path,
    )
    model.to(device)

    metrics = evaluate_model(
        model=model,
        dataloader=dataloader,
        device=device,
        threshold=args.threshold,
        predictions_path=predictions_path,
        tokenizer=tokenizer,
    )
    if metrics_path is not None:
        from src.utils import write_json

        write_json(metrics, metrics_path)
    print(json.dumps(metrics, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
