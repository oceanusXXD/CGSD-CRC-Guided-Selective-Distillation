#!/usr/bin/env python
"""Train prefix tuning for query-document relevance generation."""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path
from types import MethodType
from typing import Any

import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data import (  # noqa: E402
    GenerationPairCollator,
    GenerationQueryDocumentDataset,
    PairExample,
    TokenBudgetBatchSampler,
    filter_examples_by_ids,
    load_examples,
    split_examples_three_way,
)
from src.model import set_use_cache_false  # noqa: E402
from src.trainer import evaluate_model, fit  # noqa: E402
from src.utils import (  # noqa: E402
    configure_torch_performance,
    disable_tokenizer_thinking,
    ensure_tokenizer_padding,
    get_device,
    parse_torch_dtype,
    read_json,
    resolve_input_path,
    resolve_output_path,
    set_seed,
    write_json,
)


PREFIX_MODE = "prefix_tuning"
PREFIX_ADAPTER_DIRNAME = "prefix_adapter"
INPUT_FORMAT = "query_document_marked_v1"


def _find_text_model(module: torch.nn.Module) -> torch.nn.Module | None:
    """Return the Qwen3 text model when the base model exposes one."""
    candidate = module
    if hasattr(candidate, "get_base_model"):
        candidate = candidate.get_base_model()
    nested_model = getattr(candidate, "model", None)
    if nested_model is not None and hasattr(nested_model, "language_model"):
        return nested_model.language_model
    if nested_model is not None and hasattr(getattr(nested_model, "config", None), "layer_types"):
        return nested_model
    if hasattr(candidate, "language_model"):
        return candidate.language_model
    return None


def install_qwen3_prefix_cache_compatibility(
    peft_model: torch.nn.Module,
    num_virtual_tokens: int,
) -> None:
    """Make PEFT prefix tuning work with Qwen3 hybrid attention caches."""
    text_model = _find_text_model(peft_model)
    if text_model is None or not hasattr(getattr(text_model, "config", None), "layer_types"):
        return

    from transformers import DynamicCache

    text_model._prefix_num_virtual_tokens = int(num_virtual_tokens)

    def update_linear_attn_mask_for_prefix(self: torch.nn.Module, attention_mask: Any, past_key_values: Any) -> Any:
        linear_attn_mask = attention_mask
        prefix_length = int(getattr(self, "_prefix_num_virtual_tokens", 0))
        if linear_attn_mask is not None and prefix_length > 0 and linear_attn_mask.shape[1] > prefix_length:
            linear_attn_mask = linear_attn_mask[:, prefix_length:]
        if (past_key_values is not None and past_key_values.has_previous_state()) or (
            linear_attn_mask is not None and torch.all(linear_attn_mask == 1)
        ):
            linear_attn_mask = None
        return linear_attn_mask

    def get_prompt_with_hybrid_cache(
        self: torch.nn.Module,
        batch_size: int,
        task_ids: torch.Tensor | None = None,
        max_cache_len: int | None = None,
    ) -> Any:
        del task_ids, max_cache_len
        peft_config = self.active_peft_config
        prompt_encoder = self.prompt_encoder[self.active_adapter]
        prompt_tokens = (
            self.prompt_tokens[self.active_adapter]
            .unsqueeze(0)
            .expand(batch_size, -1)
            .to(prompt_encoder.embedding.weight.device)
        )
        prompt_tokens = prompt_tokens[:, : peft_config.num_virtual_tokens]
        if peft_config.inference_mode:
            past_key_values = prompt_encoder.embedding.weight.repeat(batch_size, 1, 1)
        else:
            past_key_values = prompt_encoder(prompt_tokens)
        if self.base_model_torch_dtype is not None:
            past_key_values = past_key_values.to(self.base_model_torch_dtype)

        past_key_values = past_key_values.view(
            batch_size,
            peft_config.num_virtual_tokens,
            peft_config.num_layers * 2,
            peft_config.num_attention_heads,
            peft_config.token_dim // peft_config.num_attention_heads,
        )
        if peft_config.num_transformer_submodules == 2:
            past_key_values = torch.cat([past_key_values, past_key_values], dim=2)
        past_key_values = past_key_values.permute([2, 0, 3, 1, 4]).split(
            peft_config.num_transformer_submodules * 2
        )

        base_model = self.get_base_model()
        base_config = getattr(base_model, "config", None)
        text_config = (
            base_config.get_text_config(decoder=True)
            if hasattr(base_config, "get_text_config")
            else getattr(base_config, "text_config", base_config)
        )
        cache = DynamicCache(config=text_config)
        cache_position = torch.arange(peft_config.num_virtual_tokens, device=past_key_values[0].device)
        layer_types = list(getattr(text_config, "layer_types", ["full_attention"] * peft_config.num_layers))
        for layer_index, layer_type in enumerate(layer_types[: peft_config.num_layers]):
            if layer_type != "full_attention":
                continue
            key_states = past_key_values[layer_index][0]
            value_states = past_key_values[layer_index][1]
            cache.update(
                key_states,
                value_states,
                layer_index,
                cache_kwargs={"cache_position": cache_position},
            )
        return cache

    text_model._update_linear_attn_mask = MethodType(update_linear_attn_mask_for_prefix, text_model)
    peft_model.get_prompt = MethodType(get_prompt_with_hybrid_cache, peft_model)


class PrefixTunedQwenGenerativeModel(torch.nn.Module):
    """Causal LM with a trainable prefix adapter for binary generation."""

    def __init__(
        self,
        model_path: str,
        prefix_num_virtual_tokens: int = 16,
        prefix_projection: bool = False,
        torch_dtype: torch.dtype | str | None = "auto",
        trust_remote_code: bool = True,
        adapter_path: str | Path | None = None,
        adapters_trainable: bool = True,
    ) -> None:
        super().__init__()
        if prefix_num_virtual_tokens < 1:
            raise ValueError("prefix_num_virtual_tokens must be at least 1")

        self.model_path = str(model_path)
        self.mode = PREFIX_MODE
        self.prefix_num_virtual_tokens = prefix_num_virtual_tokens
        self.prefix_projection = prefix_projection
        self.trust_remote_code = trust_remote_code

        loaded_model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch_dtype,
            trust_remote_code=trust_remote_code,
            local_files_only=True,
        )
        set_use_cache_false(loaded_model)
        self.backbone = self._build_prefix_backbone(
            base_model=loaded_model,
            adapter_path=adapter_path,
            adapters_trainable=adapters_trainable,
        )

    def _build_prefix_backbone(
        self,
        base_model: torch.nn.Module,
        adapter_path: str | Path | None,
        adapters_trainable: bool,
    ) -> torch.nn.Module:
        try:
            from peft import PeftModel, PrefixTuningConfig, TaskType, get_peft_model
        except ImportError as exc:
            raise ImportError("Prefix tuning requires the peft package.") from exc

        if adapter_path is not None:
            peft_model = PeftModel.from_pretrained(
                base_model,
                adapter_path,
                is_trainable=adapters_trainable,
            )
            install_qwen3_prefix_cache_compatibility(peft_model, self.prefix_num_virtual_tokens)
            return peft_model

        config = PrefixTuningConfig(
            task_type=TaskType.CAUSAL_LM,
            num_virtual_tokens=self.prefix_num_virtual_tokens,
            prefix_projection=self.prefix_projection,
        )
        peft_model = get_peft_model(base_model, config)
        install_qwen3_prefix_cache_compatibility(peft_model, self.prefix_num_virtual_tokens)
        return peft_model

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor | None = None,
        **_: Any,
    ) -> Any:
        return self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            return_dict=True,
        )

    def generate(self, *args: Any, **kwargs: Any) -> Any:
        return self.backbone.generate(*args, **kwargs)

    def checkpoint_config(self) -> dict[str, Any]:
        """Return the config needed to reload this prefix tuning checkpoint."""
        return {
            "model_path": self.model_path,
            "mode": self.mode,
            "prefix_num_virtual_tokens": self.prefix_num_virtual_tokens,
            "prefix_projection": self.prefix_projection,
            "adapter_dirname": PREFIX_ADAPTER_DIRNAME,
            "adapter_type": "peft_prefix_tuning",
            "trust_remote_code": self.trust_remote_code,
        }

    def save_checkpoint(
        self,
        output_dir: str | Path,
        tokenizer: Any | None = None,
        extra_config: dict[str, Any] | None = None,
    ) -> None:
        """Save config, prefix adapter, and tokenizer."""
        target_dir = Path(output_dir)
        target_dir.mkdir(parents=True, exist_ok=True)

        config = self.checkpoint_config()
        if extra_config:
            config.update(extra_config)
        write_json(config, target_dir / "model_config.json")
        self.backbone.save_pretrained(target_dir / PREFIX_ADAPTER_DIRNAME)

        if tokenizer is not None:
            disable_tokenizer_thinking(tokenizer)
            tokenizer.save_pretrained(target_dir)
            chat_template_path = target_dir / "chat_template.jinja"
            if chat_template_path.exists():
                chat_template_path.unlink()

    @classmethod
    def load_from_checkpoint(
        cls,
        checkpoint_dir: str | Path,
        torch_dtype: torch.dtype | str | None = "auto",
        map_location: str | torch.device = "cpu",
        model_path: str | Path | None = None,
    ) -> "PrefixTunedQwenGenerativeModel":
        """Load a saved prefix tuning checkpoint for evaluation."""
        checkpoint_path = Path(checkpoint_dir)
        config = read_json(checkpoint_path / "model_config.json")
        adapter_dirname = str(config.get("adapter_dirname", PREFIX_ADAPTER_DIRNAME))
        base_model_path = str(model_path) if model_path is not None else str(config["model_path"])

        return cls(
            model_path=base_model_path,
            prefix_num_virtual_tokens=int(config.get("prefix_num_virtual_tokens", 16)),
            prefix_projection=bool(config.get("prefix_projection", False)),
            torch_dtype=torch_dtype,
            trust_remote_code=bool(config.get("trust_remote_code", True)),
            adapter_path=checkpoint_path / adapter_dirname,
            adapters_trainable=False,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_path", default="model/qwen3-0.6b")
    parser.add_argument("--data_path", default="datasets")
    parser.add_argument("--eval_data_path", default=None)
    parser.add_argument("--output_dir", default="outputs/prefix_tuning")
    parser.add_argument("--checkpoint_dir", default=None)
    parser.add_argument("--eval_only", action="store_true")
    parser.add_argument("--query_field", default="query")
    parser.add_argument("--document_field", default="document")
    parser.add_argument("--label_field", default="groundtruth")
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--eval_batch_size", type=int, default=None)
    parser.add_argument("--max_tokens_per_batch", type=int, default=16384)
    parser.add_argument("--eval_max_tokens_per_batch", type=int, default=32768)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-4)
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
    parser.add_argument("--prefix_num_virtual_tokens", type=int, default=16)
    parser.add_argument("--prefix_projection", action="store_true")
    parser.add_argument("--test_eval", dest="test_eval", action="store_true", default=True)
    parser.add_argument("--no_test_eval", dest="test_eval", action="store_false")
    parser.add_argument("--predictions_path", default=None)
    parser.add_argument("--metrics_path", default=None)
    parser.add_argument("--split_ids_path", default=None)
    parser.add_argument("--split_name", choices=["all", "train", "val", "test"], default="all")
    return parser.parse_args()


def apply_prefix_speed_defaults(args: argparse.Namespace) -> None:
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


def balance_train_examples(examples: list[PairExample], seed: int) -> list[PairExample]:
    """Oversample minority labels in the training split to match the majority label."""
    by_label: dict[int, list[PairExample]] = {}
    for example in examples:
        by_label.setdefault(example.label, []).append(example)

    if len(by_label) < 2:
        return list(examples)

    target_size = max(len(label_examples) for label_examples in by_label.values())
    rng = random.Random(seed)
    balanced: list[PairExample] = []
    for label in sorted(by_label):
        label_examples = list(by_label[label])
        balanced.extend(label_examples)
        missing = target_size - len(label_examples)
        if missing > 0:
            balanced.extend(rng.choices(label_examples, k=missing))

    rng.shuffle(balanced)
    return balanced


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


def load_tokenizer(model_path: Path, checkpoint_dir: Path | None = None) -> AutoTokenizer:
    """Load tokenizer from checkpoint when available, otherwise from the base model."""
    source = checkpoint_dir if checkpoint_dir is not None and checkpoint_dir.exists() else model_path
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            source,
            trust_remote_code=True,
            local_files_only=True,
        )
    except OSError:
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
            local_files_only=True,
        )
    ensure_tokenizer_padding(tokenizer)
    disable_tokenizer_thinking(tokenizer)
    return tokenizer


def make_dataset(
    examples: list[Any],
    tokenizer: Any,
    max_length: int,
    cache_tokenization: bool,
) -> GenerationQueryDocumentDataset:
    return GenerationQueryDocumentDataset(
        examples,
        tokenizer=tokenizer,
        max_length=max_length,
        cache_tokenization=cache_tokenization,
        input_format=INPUT_FORMAT,
    )


def evaluate_checkpoint(args: argparse.Namespace) -> dict[str, Any]:
    apply_prefix_speed_defaults(args)
    configure_torch_performance(enable_tf32=args.tf32)

    checkpoint_dir = resolve_input_path(args.checkpoint_dir or args.output_dir, PROJECT_ROOT)
    checkpoint_config = read_json(checkpoint_dir / "model_config.json")
    model_path = (
        resolve_input_path(args.model_path, PROJECT_ROOT)
        if args.model_path is not None
        else resolve_input_path(checkpoint_config["model_path"], PROJECT_ROOT)
    )
    data_path = resolve_input_path(args.data_path, PROJECT_ROOT)
    predictions_path = (
        resolve_output_path(args.predictions_path, PROJECT_ROOT)
        if args.predictions_path is not None
        else None
    )
    split_ids_path = (
        resolve_input_path(args.split_ids_path, PROJECT_ROOT)
        if args.split_ids_path is not None
        else checkpoint_dir / "split_ids.json"
    )

    tokenizer = load_tokenizer(model_path=model_path, checkpoint_dir=checkpoint_dir)
    examples = load_examples(
        data_path,
        query_field=args.query_field,
        document_field=args.document_field,
        label_field=args.label_field,
    )
    if args.split_name != "all":
        split_data = read_json(split_ids_path)
        split_ids = set(str(sample_id) for sample_id in split_data[f"{args.split_name}_ids"])
        examples = filter_examples_by_ids(examples, split_ids)

    device = get_device(args.device)
    dataset = make_dataset(
        examples,
        tokenizer=tokenizer,
        max_length=args.max_length,
        cache_tokenization=args.cache_tokenization,
    )
    dataloader = build_dataloader(
        dataset=dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        collator=GenerationPairCollator(
            tokenizer,
            pad_to_multiple_of=args.pad_to_multiple_of if args.pad_to_multiple_of > 0 else None,
        ),
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor,
        pin_memory=args.pin_memory and device.type == "cuda",
        max_tokens_per_batch=args.eval_max_tokens_per_batch,
        seed=args.seed,
    )
    model = PrefixTunedQwenGenerativeModel.load_from_checkpoint(
        checkpoint_dir,
        torch_dtype=parse_torch_dtype(args.torch_dtype),
        map_location="cpu",
        model_path=model_path,
    )
    model.to(device)
    return evaluate_model(
        model=model,
        dataloader=dataloader,
        device=device,
        threshold=args.threshold,
        predictions_path=predictions_path,
        tokenizer=tokenizer,
    )


def train(args: argparse.Namespace) -> None:
    apply_prefix_speed_defaults(args)
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

    tokenizer = load_tokenizer(model_path=model_path)
    all_examples = load_examples(
        data_path,
        query_field=args.query_field,
        document_field=args.document_field,
        label_field=args.label_field,
    )
    if eval_data_path is None:
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
    effective_train_examples = (
        balance_train_examples(train_examples, seed=args.seed)
        if args.balance_train_classes
        else list(train_examples)
    )
    effective_train_label_counts = count_labels(effective_train_examples)

    train_dataset = make_dataset(
        effective_train_examples,
        tokenizer=tokenizer,
        max_length=args.max_length,
        cache_tokenization=args.cache_tokenization,
    )
    eval_dataset = (
        make_dataset(
            eval_examples,
            tokenizer=tokenizer,
            max_length=args.max_length,
            cache_tokenization=args.cache_tokenization,
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

    model = PrefixTunedQwenGenerativeModel(
        model_path=str(model_path),
        prefix_num_virtual_tokens=args.prefix_num_virtual_tokens,
        prefix_projection=args.prefix_projection,
        torch_dtype=parse_torch_dtype(args.torch_dtype),
        trust_remote_code=args.trust_remote_code,
    )

    run_config = {
        "mode": PREFIX_MODE,
        "data_path": str(data_path),
        "eval_data_path": str(eval_data_path) if eval_data_path is not None else None,
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
        "class_balance_strategy": "oversample_minority_to_majority" if args.balance_train_classes else "none",
        "train_label_counts": train_label_counts,
        "effective_train_label_counts": effective_train_label_counts,
        "num_workers": args.num_workers,
        "prefetch_factor": args.prefetch_factor,
        "pad_to_multiple_of": args.pad_to_multiple_of,
        "cache_tokenization": args.cache_tokenization,
        "pin_memory": pin_memory,
        "tf32": args.tf32,
        "disable_thinking": True,
        "input_format": INPUT_FORMAT,
        "split_algorithm": "stratified_query_document_grouped_v1",
        "split_group_duplicates": args.group_duplicates if eval_data_path is None else False,
        "train_ratio": args.train_ratio,
        "val_ratio": args.val_ratio,
        "threshold": args.threshold,
        "seed": args.seed,
        "total_size": len(all_examples),
        "train_size": len(train_examples),
        "effective_train_size": len(effective_train_examples),
        "eval_size": len(eval_examples),
        "test_size": len(test_examples),
        "prefix_num_virtual_tokens": args.prefix_num_virtual_tokens,
        "prefix_projection": args.prefix_projection,
    }
    print(json.dumps({"run_config": run_config}, ensure_ascii=False, sort_keys=True))

    split_payload = {
        "train_ids": [example.sample_id for example in train_examples],
        "val_ids": [example.sample_id for example in eval_examples],
        "test_ids": [example.sample_id for example in test_examples],
        "train_ratio": args.train_ratio if eval_data_path is None else None,
        "val_ratio": args.val_ratio if eval_data_path is None else None,
        "test_ratio": 1.0 - args.train_ratio - args.val_ratio if eval_data_path is None else None,
        "split_algorithm": "stratified_query_document_grouped_v1" if eval_data_path is None else "external_eval_v1",
        "group_duplicates": args.group_duplicates if eval_data_path is None else False,
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
    )

    if args.test_eval and test_examples:
        test_dataset = make_dataset(
            test_examples,
            tokenizer=tokenizer,
            max_length=args.max_length,
            cache_tokenization=args.cache_tokenization,
        )
        test_loader = build_dataloader(
            dataset=test_dataset,
            batch_size=args.eval_batch_size,
            shuffle=False,
            collator=collator,
            num_workers=args.num_workers,
            prefetch_factor=args.prefetch_factor,
            pin_memory=pin_memory,
            max_tokens_per_batch=args.eval_max_tokens_per_batch,
            seed=args.seed,
        )
        best_model = PrefixTunedQwenGenerativeModel.load_from_checkpoint(
            output_dir,
            torch_dtype=parse_torch_dtype(args.torch_dtype),
            map_location="cpu",
            model_path=model_path,
        )
        best_model.to(device)
        test_metrics = evaluate_model(
            model=best_model,
            dataloader=test_loader,
            device=device,
            threshold=args.threshold,
            tokenizer=tokenizer,
        )
        write_json(test_metrics, output_dir / "test_metrics.json")
        print(json.dumps({"test": test_metrics}, ensure_ascii=False, sort_keys=True))


def main() -> None:
    args = parse_args()
    if args.eval_only:
        metrics = evaluate_checkpoint(args)
        if args.metrics_path is not None:
            write_json(metrics, resolve_output_path(args.metrics_path, PROJECT_ROOT))
        print(json.dumps(metrics, ensure_ascii=False, indent=2, sort_keys=True))
        return
    train(args)


if __name__ == "__main__":
    main()
