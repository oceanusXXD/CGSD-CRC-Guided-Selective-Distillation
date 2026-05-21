#!/usr/bin/env python
"""用选出的训练样本训练单个 LoRA round。

输入是 `cgsd_train_rows.jsonl` 或显式 `--train_rows_path`，每行必须是统一
`id/query/document/label|groundtruth` 格式。脚本从基座模型加载 Qwen，
只训练 LoRA adapter，并把 checkpoint 写到 `round_<n>/model/`。round 0
表示未训练基座模型，所以本脚本只允许训练 round >= 1。
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.cgsd_cli_common import (
    add_runtime_overrides,
    add_stage_cache_args,
    estimate_query_document_prompt_tokens,
    input_artifact_path,
    load_selected_train_rows,
    load_stage_examples,
    output_dir_from_arg,
    output_artifact_path,
    print_existing_stage_result,
    read_jsonl,
    runtime_args_from_cli,
    stage_cache_decision,
    summarize_teacher_label_usage,
    train_label_snapshot,
    write_stage_usage,
)
from src.binary_protocol import BINARY_NEGATIVE_TEXT, BINARY_POSITIVE_TEXT
from src.data import (
    GenerationPairCollator,
    GenerationQueryDocumentDataset,
    PairExample,
    examples_from_rows,
    filter_examples_by_ids,
)
from src.model import QwenGenerativeModel
from src.trainer import fit
from src.utils import (
    configure_torch_performance,
    disable_tokenizer_thinking,
    ensure_tokenizer_padding,
    get_device,
    parse_torch_dtype,
    read_json,
    set_seed,
    write_json,
    write_jsonl,
)
from transformers import AutoTokenizer


def count_labels(examples: list[PairExample]) -> dict[int, int]:
    return dict(sorted(Counter(example.label for example in examples).items()))


def compute_balanced_class_weights(examples: list[PairExample]) -> dict[int, float]:
    label_counts = Counter(example.label for example in examples)
    total = sum(label_counts.values())
    if total == 0:
        return {}
    return {label: total / max(2.0 * label_counts.get(label, 0), 1.0) for label in (0, 1)}


def get_single_token_id(tokenizer: Any, text: str) -> int:
    token_ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    if len(token_ids) != 1:
        raise ValueError(f"Expected {text!r} to encode to one token, got {token_ids}")
    return int(token_ids[0])


def build_class_token_weights(tokenizer: Any, class_weights: dict[int, float]) -> dict[int, float]:
    label_tokens = {0: BINARY_NEGATIVE_TEXT, 1: BINARY_POSITIVE_TEXT}
    return {get_single_token_id(tokenizer, label_tokens[int(label)]): weight for label, weight in class_weights.items()}


def build_train_dataloader(
    examples: list[PairExample],
    tokenizer: Any,
    *,
    max_length: int,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    prefetch_factor: int,
    pin_memory: bool,
    pad_to_multiple_of: int,
    cache_tokenization: bool,
) -> Any:
    from torch.utils.data import DataLoader

    dataset = GenerationQueryDocumentDataset(
        examples,
        tokenizer=tokenizer,
        max_length=max_length,
        cache_tokenization=cache_tokenization,
        input_format="cgsd_chat_binary_v1",
    )
    kwargs: dict[str, Any] = {
        "dataset": dataset,
        "batch_size": batch_size,
        "shuffle": shuffle,
        "collate_fn": GenerationPairCollator(
            tokenizer,
            pad_to_multiple_of=pad_to_multiple_of if pad_to_multiple_of > 0 else None,
        ),
        "num_workers": num_workers,
        "pin_memory": pin_memory,
    }
    if num_workers > 0:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = prefetch_factor
    return DataLoader(**kwargs)


def train_round_model(
    *,
    train_examples: list[PairExample],
    eval_examples: list[PairExample],
    tokenizer: Any,
    model_path: Path,
    init_adapter_path: Path | None,
    output_dir: Path,
    device: Any,
    args: Any,
    round_index: int,
    run_config: dict[str, Any],
) -> QwenGenerativeModel:
    train_loader = build_train_dataloader(
        train_examples,
        tokenizer,
        max_length=args.max_length,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor,
        pin_memory=args.pin_memory and device.type == "cuda",
        pad_to_multiple_of=args.pad_to_multiple_of,
        cache_tokenization=args.cache_tokenization,
    )
    model = QwenGenerativeModel(
        model_path=str(model_path),
        mode="lora_attention_mlp",
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        lora_target_modules=args.lora_target_modules,
        lora_layer_scope=args.lora_layer_scope,
        adapter_path=init_adapter_path,
        adapters_trainable=True,
        torch_dtype=parse_torch_dtype(args.torch_dtype),
        trust_remote_code=args.trust_remote_code,
    )
    class_weights = compute_balanced_class_weights(train_examples) if args.balance_train_classes else {}
    class_token_weights = build_class_token_weights(tokenizer, class_weights) if class_weights else {}
    round_config = dict(run_config)
    round_config.update(
        {
            "round_index": int(round_index),
            "train_size": len(train_examples),
            "eval_size": 0,
            "guide_size_held_out": len(eval_examples),
            "guide_used_for_training_or_model_selection": False,
            "train_label_counts": count_labels(train_examples),
            "class_weights": class_weights,
            "class_token_weights": class_token_weights,
        }
    )
    fit(
        model=model,
        train_loader=train_loader,
        eval_loader=None,
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
        run_config=round_config,
        early_stopping_patience=args.early_stopping_patience,
        early_stopping_min_delta=args.early_stopping_min_delta,
        class_token_weights=class_token_weights if args.balance_train_classes else None,
        scheduler_type="cosine",
    )
    model.to("cpu")
    if device.type == "cuda":
        import torch

        torch.cuda.empty_cache()
    trained_model = QwenGenerativeModel.load_from_checkpoint(
        output_dir,
        torch_dtype=parse_torch_dtype(args.torch_dtype),
        model_path=model_path,
    )
    trained_model.to(device)
    return trained_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--round_index", type=int, required=True, help="要产出的 LoRA round，例如用选出的训练集训练 round1")
    parser.add_argument("--model_path", default="model/qwen3-0.6b")
    parser.add_argument("--data_path", default="datasets")
    parser.add_argument("--query_field", default="query")
    parser.add_argument("--document_field", default="document")
    parser.add_argument("--label_field", default="groundtruth")
    parser.add_argument("--split_ids_path", default=None)
    parser.add_argument("--train_rows_path", default=None)
    parser.add_argument("--checkpoint_dir", default=None)
    parser.add_argument("--init_adapter_path", default=None)
    parser.add_argument("--training_rows_used_path", default=None)
    parser.add_argument("--train_label_snapshot_path", default=None)
    parser.add_argument("--training_summary_path", default=None)
    parser.add_argument("--usage_path", default=None)
    add_runtime_overrides(parser)
    add_stage_cache_args(parser)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if int(args.round_index) <= 0:
        raise ValueError("round 0 is the base model and has no LoRA checkpoint; train round_index must be >= 1")
    output_dir = output_dir_from_arg(args.output_dir)
    round_dir = output_dir / f"round_{args.round_index}"
    train_rows_path = input_artifact_path(args.train_rows_path, output_dir / "cgsd_train_rows.jsonl")
    checkpoint_dir = output_artifact_path(args.checkpoint_dir, round_dir / "model")
    training_rows_used_path = output_artifact_path(
        args.training_rows_used_path,
        round_dir / "training_rows_used.jsonl",
    )
    train_label_snapshot_path = output_artifact_path(
        args.train_label_snapshot_path,
        round_dir / "train_label_snapshot.json",
    )
    training_summary_path = output_artifact_path(
        args.training_summary_path,
        round_dir / "training_round_summary.json",
    )
    usage_path = output_artifact_path(args.usage_path, round_dir / "train_usage.json")
    if args.show_result:
        print_existing_stage_result(stage_name="cgsd_train_round", summary_path=training_summary_path)
        return
    cache_decision = stage_cache_decision(
        stage_name="cgsd_train_round",
        required_outputs=[
            checkpoint_dir / "model_config.json",
            training_rows_used_path,
            train_label_snapshot_path,
            training_summary_path,
            usage_path,
        ],
        cache_policy=args.cache_policy,
    )
    if cache_decision.cache_hit:
        print_existing_stage_result(stage_name="cgsd_train_round", summary_path=training_summary_path)
        return

    round_dir.mkdir(parents=True, exist_ok=True)
    runtime_args = runtime_args_from_cli(args)
    set_seed(int(runtime_args.seed))
    configure_torch_performance(enable_tf32=runtime_args.tf32)
    selected_rows = read_jsonl(train_rows_path) if args.train_rows_path else load_selected_train_rows(output_dir)
    if not selected_rows:
        raise RuntimeError(f"{train_rows_path} is empty; pass --train_rows_path with canonical training JSONL rows")
    training_rows_snapshot = [
        {
            **dict(row),
            "trained_for_round": int(args.round_index),
            "training_artifact_role": "cumulative_lora_train_row",
        }
        for row in selected_rows
    ]
    write_jsonl(training_rows_snapshot, training_rows_used_path)

    model_path = input_artifact_path(args.model_path, PROJECT_ROOT / "model" / "qwen3-0.6b")
    init_adapter_path = input_artifact_path(args.init_adapter_path, output_dir) if args.init_adapter_path else None
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=runtime_args.trust_remote_code,
        local_files_only=True,
    )
    ensure_tokenizer_padding(tokenizer)
    disable_tokenizer_thinking(tokenizer)

    train_examples = examples_from_rows(selected_rows)
    all_examples = load_stage_examples(
        data_path=args.data_path,
        query_field=args.query_field,
        document_field=args.document_field,
        label_field=args.label_field,
    )
    split_payload = read_json(input_artifact_path(args.split_ids_path, output_dir / "cgsd_split_ids.json"))
    guide_ids = {str(sample_id) for sample_id in split_payload["guide_ids"]}
    guide_examples = filter_examples_by_ids(all_examples, guide_ids)
    device = get_device(args.device)
    run_config = vars(args)
    run_config["seed"] = int(runtime_args.seed)
    model = train_round_model(
        train_examples=train_examples,
        eval_examples=guide_examples,
        tokenizer=tokenizer,
        model_path=model_path,
        init_adapter_path=init_adapter_path,
        output_dir=checkpoint_dir,
        device=device,
        args=runtime_args,
        round_index=args.round_index,
        run_config=run_config,
    )
    model.to("cpu")
    label_snapshot = train_label_snapshot(selected_rows)
    selection_round_counts: dict[str, int] = {}
    for row in selected_rows:
        key = str(row.get("selection_round", "unknown"))
        selection_round_counts[key] = selection_round_counts.get(key, 0) + 1
    write_json(label_snapshot, train_label_snapshot_path)
    write_json(
        {
            "round_index": int(args.round_index),
            "model_round_index": int(args.round_index),
            "checkpoint_dir": str(checkpoint_dir),
            "training_rows_path": str(training_rows_used_path),
            "train_size": len(selected_rows),
            "train_label_snapshot_size": len(label_snapshot),
            "selection_round_counts": selection_round_counts,
            "source_selection_rounds": sorted(selection_round_counts),
            "training_mode": "lora_sft_from_base_model",
            "init_adapter_path": str(init_adapter_path) if init_adapter_path is not None else None,
        },
        training_summary_path,
    )
    teacher_usage = summarize_teacher_label_usage(selected_rows, purpose="training_label_reuse")
    train_steps = int(math.ceil(len(selected_rows) / max(int(runtime_args.batch_size), 1)) * int(runtime_args.epochs))
    write_stage_usage(
        usage_path,
        {
            "stage_name": "cgsd_train_round",
            "round_index": int(args.round_index),
            "cache": cache_decision.to_dict(),
            "student_model_calls": 0,
            "student_model_train_steps_estimated": train_steps,
            "student_model_train_examples": int(len(selected_rows) * int(runtime_args.epochs)),
            "estimated_student_train_tokens": int(
                estimate_query_document_prompt_tokens(selected_rows) * int(runtime_args.epochs)
            ),
            "teacher_label_usage": teacher_usage,
            "groundtruth_substitute_calls": teacher_usage["groundtruth_substitute_calls"],
            "teacher_api_file_calls": teacher_usage["teacher_api_file_calls"],
            "checkpoint_dir": str(checkpoint_dir),
        },
    )
    print(
        json.dumps(
            {
                "round_index": args.round_index,
                "checkpoint_dir": str(checkpoint_dir),
                "train_size": len(selected_rows),
                "selection_round_counts": selection_round_counts,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
