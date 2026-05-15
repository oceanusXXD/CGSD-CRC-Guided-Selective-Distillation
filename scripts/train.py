#!/usr/bin/env python
"""Train one query-document relevance experiment."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

from torch.utils.data import DataLoader
from transformers import AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data import (
    GenerationPairCollator,
    GenerationQueryDocumentDataset,
    PairExample,
    TokenBudgetBatchSampler,
    filter_examples_by_ids,
    load_examples,
    split_examples_three_way,
)
from src.model import ALL_MODES, LORA_LAYER_SCOPES, LORA_TARGET_GROUPS, QwenGenerativeModel
from src.trainer import fit, get_single_token_id
from src.utils import (
    configure_torch_performance,
    disable_tokenizer_thinking,
    ensure_tokenizer_padding,
    get_device,
    parse_torch_dtype,
    resolve_input_path,
    resolve_output_path,
    set_seed,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", required=True, choices=ALL_MODES)
    parser.add_argument("--model_path", default="model/qwen3-0.6b")
    parser.add_argument("--data_path", default="datasets")
    parser.add_argument("--eval_data_path", default=None)
    parser.add_argument("--split_ids_path", default=None)
    parser.add_argument("--output_dir", default="outputs/run")
    parser.add_argument("--query_field", default="query")
    parser.add_argument("--document_field", default="document")
    parser.add_argument("--label_field", default="groundtruth")
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--eval_batch_size", type=int, default=None)
    parser.add_argument("--max_tokens_per_batch", type=int, default=16384)
    parser.add_argument("--eval_max_tokens_per_batch", type=int, default=32768)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=None)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--early_stopping_patience", type=int, default=None)
    parser.add_argument("--early_stopping_min_delta", type=float, default=0.0)
    parser.add_argument("--train_ratio", type=float, default=0.1)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--group_duplicates", action="store_true", default=True)
    parser.add_argument("--no_group_duplicates", dest="group_duplicates", action="store_false")
    parser.add_argument("--balance_train_classes", action="store_true", default=True)
    parser.add_argument("--no_balance_train_classes", dest="balance_train_classes", action="store_false")
    parser.add_argument("--threshold", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
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
    parser.add_argument("--trust_remote_code", action="store_true", default=True)
    parser.add_argument("--no_trust_remote_code", dest="trust_remote_code", action="store_false")
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--lora_target_modules", choices=sorted(LORA_TARGET_GROUPS), default="attention_mlp")
    parser.add_argument("--lora_layer_scope", choices=sorted(LORA_LAYER_SCOPES), default="last1")
    return parser.parse_args()


def apply_mode_speed_defaults(args: argparse.Namespace) -> None:
    """Choose L4-friendly defaults while preserving explicit CLI overrides."""
    if args.batch_size is None:
        args.batch_size = 8
    if args.eval_batch_size is None:
        args.eval_batch_size = 32
    if args.gradient_accumulation_steps is None:
        args.gradient_accumulation_steps = 2


def count_labels(examples: list[PairExample]) -> dict[int, int]:
    """Count binary labels in examples."""
    return dict(sorted(Counter(example.label for example in examples).items()))


def compute_balanced_class_weights(examples: list[PairExample]) -> dict[int, float]:
    """Return sklearn-style balanced class weights for binary labels."""
    label_counts = Counter(example.label for example in examples)
    total = sum(label_counts.values())
    if total == 0:
        return {}
    return {
        label: total / max(2.0 * label_counts.get(label, 0), 1.0)
        for label in (0, 1)
    }


def build_class_token_weights(tokenizer: object, class_weights: dict[int, float]) -> dict[int, float]:
    """Map binary class weights onto the tokenizer's answer token ids."""
    return {
        get_single_token_id(tokenizer, str(label)): weight
        for label, weight in class_weights.items()
    }


def apply_precomputed_split_ids(
    examples: list[PairExample],
    split_payload: dict[str, object],
) -> tuple[list[PairExample], list[PairExample], list[PairExample]]:
    """Build train/val/test splits from an existing split_ids.json payload."""
    by_id = {example.sample_id: example for example in examples}

    def resolve_split(name: str) -> list[PairExample]:
        raw_ids = split_payload.get(f"{name}_ids", [])
        if not isinstance(raw_ids, list):
            raise ValueError(f"{name}_ids must be a list in split_ids_path")
        ids = [str(sample_id) for sample_id in raw_ids]
        missing = [sample_id for sample_id in ids if sample_id not in by_id]
        if missing:
            raise ValueError(f"split_ids_path references missing {name} ids: {missing[:5]}")
        return [by_id[sample_id] for sample_id in ids]

    train_examples = resolve_split("train")
    eval_examples = resolve_split("val")
    test_examples = resolve_split("test")
    split_ids = (
        [example.sample_id for example in train_examples]
        + [example.sample_id for example in eval_examples]
        + [example.sample_id for example in test_examples]
    )
    if len(split_ids) != len(set(split_ids)):
        raise ValueError("split_ids_path contains duplicate ids across splits")
    return train_examples, eval_examples, test_examples


def build_dataloader(
    dataset: GenerationQueryDocumentDataset,
    batch_size: int,
    shuffle: bool,
    collator: GenerationPairCollator,
    num_workers: int,
    prefetch_factor: int,
    pin_memory: bool,
    max_tokens_per_batch: int = 0,
    seed: int = 42,
) -> DataLoader:
    """Create a DataLoader with fast CUDA input settings when available."""
    kwargs = {"dataset": dataset, "collate_fn": collator, "num_workers": num_workers, "pin_memory": pin_memory}
    if max_tokens_per_batch > 0:
        kwargs["batch_sampler"] = TokenBudgetBatchSampler(
            lengths=dataset.sequence_lengths(),
            max_batch_size=batch_size,
            max_tokens_per_batch=max_tokens_per_batch,
            shuffle=shuffle,
            seed=seed,
        )
    else:
        kwargs["batch_size"] = batch_size
        kwargs["shuffle"] = shuffle
    if num_workers > 0:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = prefetch_factor
    return DataLoader(**kwargs)


def main() -> None:
    args = parse_args()
    apply_mode_speed_defaults(args)
    set_seed(args.seed)
    configure_torch_performance(enable_tf32=args.tf32)

    model_path = resolve_input_path(args.model_path, PROJECT_ROOT)
    data_path = resolve_input_path(args.data_path, PROJECT_ROOT)
    output_dir = resolve_output_path(args.output_dir, PROJECT_ROOT)
    eval_data_path = (
        resolve_input_path(args.eval_data_path, PROJECT_ROOT)
        if args.eval_data_path is not None
        else None
    )
    split_ids_path = (
        resolve_input_path(args.split_ids_path, PROJECT_ROOT)
        if args.split_ids_path is not None
        else None
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=args.trust_remote_code,
        local_files_only=True,
    )
    ensure_tokenizer_padding(tokenizer)
    disable_tokenizer_thinking(tokenizer)

    all_examples = load_examples(
        data_path,
        query_field=args.query_field,
        document_field=args.document_field,
        label_field=args.label_field,
    )
    split_payload: dict[str, object] | None = None
    if split_ids_path is not None:
        with split_ids_path.open("r", encoding="utf-8") as handle:
            split_payload = json.load(handle)
        train_examples, eval_examples, test_examples = apply_precomputed_split_ids(
            all_examples,
            split_payload,
        )
    elif eval_data_path is None:
        train_examples, eval_examples, test_examples = split_examples_three_way(
            all_examples,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            seed=args.seed,
            stratified=True,
            group_duplicates=args.group_duplicates,
        )
    else:
        train_examples = all_examples
        eval_examples = load_examples(
            eval_data_path,
            query_field=args.query_field,
            document_field=args.document_field,
            label_field=args.label_field,
        )
        test_examples = []

    train_label_counts = count_labels(train_examples)
    effective_train_examples = list(train_examples)
    effective_train_label_counts = count_labels(effective_train_examples)
    class_weights = compute_balanced_class_weights(train_examples) if args.balance_train_classes else {}
    class_token_weights = build_class_token_weights(tokenizer, class_weights) if class_weights else {}

    train_dataset = GenerationQueryDocumentDataset(
        effective_train_examples,
        tokenizer=tokenizer,
        max_length=args.max_length,
        cache_tokenization=args.cache_tokenization,
        input_format="query_document_marked_v1",
    )
    eval_dataset = (
        GenerationQueryDocumentDataset(
            eval_examples,
            tokenizer=tokenizer,
            max_length=args.max_length,
            cache_tokenization=args.cache_tokenization,
            input_format="query_document_marked_v1",
        )
        if eval_examples
        else None
    )

    device = get_device(args.device)
    pin_memory = args.pin_memory and device.type == "cuda"
    collator = GenerationPairCollator(
        tokenizer,
        pad_to_multiple_of=args.pad_to_multiple_of if args.pad_to_multiple_of > 0 else None,
    )
    train_loader = build_dataloader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collator=collator,
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor,
        pin_memory=pin_memory,
        max_tokens_per_batch=args.max_tokens_per_batch,
        seed=args.seed,
    )
    eval_loader = (
        build_dataloader(
            dataset=eval_dataset,
            batch_size=args.eval_batch_size,
            shuffle=False,
            collator=collator,
            num_workers=args.num_workers,
            prefetch_factor=args.prefetch_factor,
            pin_memory=pin_memory,
            max_tokens_per_batch=args.eval_max_tokens_per_batch,
            seed=args.seed,
        )
        if eval_dataset is not None
        else None
    )

    model = QwenGenerativeModel(
        model_path=str(model_path),
        mode=args.mode,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        lora_target_modules=args.lora_target_modules,
        lora_layer_scope=args.lora_layer_scope,
        torch_dtype=parse_torch_dtype(args.torch_dtype),
        trust_remote_code=args.trust_remote_code,
    )

    run_config = {
        "mode": args.mode,
        "data_path": str(data_path),
        "eval_data_path": str(eval_data_path) if eval_data_path is not None else None,
        "split_ids_path": str(split_ids_path) if split_ids_path is not None else None,
        "output_dir": str(output_dir),
        "max_length": args.max_length,
        "batch_size": args.batch_size,
        "eval_batch_size": args.eval_batch_size,
        "max_tokens_per_batch": args.max_tokens_per_batch,
        "eval_max_tokens_per_batch": args.eval_max_tokens_per_batch,
        "epochs": args.epochs,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "max_grad_norm": args.max_grad_norm,
        "warmup_ratio": args.warmup_ratio,
        "early_stopping_patience": args.early_stopping_patience,
        "early_stopping_min_delta": args.early_stopping_min_delta,
        "balance_train_classes": args.balance_train_classes,
        "class_balance_strategy": "balanced_class_weight" if args.balance_train_classes else "none",
        "class_weights": class_weights,
        "class_token_weights": class_token_weights,
        "train_label_counts": train_label_counts,
        "effective_train_label_counts": effective_train_label_counts,
        "num_workers": args.num_workers,
        "prefetch_factor": args.prefetch_factor,
        "pad_to_multiple_of": args.pad_to_multiple_of,
        "cache_tokenization": args.cache_tokenization,
        "pin_memory": pin_memory,
        "tf32": args.tf32,
        "disable_thinking": True,
        "input_format": "query_document_marked_v1",
        "split_algorithm": (
            "precomputed_split_ids_v1"
            if split_ids_path is not None
            else "stratified_query_document_grouped_v1"
            if eval_data_path is None
            else "external_eval_v1"
        ),
        "split_group_duplicates": (
            split_payload.get("group_duplicates")
            if split_payload is not None
            else args.group_duplicates if eval_data_path is None else False
        ),
        "train_ratio": args.train_ratio,
        "val_ratio": args.val_ratio,
        "threshold": args.threshold,
        "seed": args.seed,
        "total_size": len(all_examples),
        "train_size": len(train_examples),
        "effective_train_size": len(effective_train_examples),
        "eval_size": len(eval_examples),
        "test_size": len(test_examples),
    }
    print(json.dumps({"run_config": run_config}, ensure_ascii=False, sort_keys=True))

    split_payload = {
        "train_ids": [example.sample_id for example in train_examples],
        "val_ids": [example.sample_id for example in eval_examples],
        "test_ids": [example.sample_id for example in test_examples],
        "train_ratio": args.train_ratio if eval_data_path is None else None,
        "val_ratio": args.val_ratio if eval_data_path is None else None,
        "test_ratio": 1.0 - args.train_ratio - args.val_ratio if eval_data_path is None else None,
        "split_algorithm": run_config["split_algorithm"],
        "group_duplicates": run_config["split_group_duplicates"],
        "split_key": "query_document",
        "seed": args.seed,
    }
    write_json(split_payload, output_dir / "split_ids.json")

    fit(
        model=model,
        train_loader=train_loader,
        eval_loader=eval_loader,
        tokenizer=tokenizer,
        output_dir=output_dir,
        device=device,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_grad_norm=args.max_grad_norm,
        warmup_ratio=args.warmup_ratio,
        threshold=args.threshold,
        run_config=run_config,
        early_stopping_patience=args.early_stopping_patience,
        early_stopping_min_delta=args.early_stopping_min_delta,
        class_token_weights=class_token_weights if args.balance_train_classes else None,
    )


if __name__ == "__main__":
    main()
