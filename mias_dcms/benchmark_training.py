from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from contextlib import nullcontext
import json
import math
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as functional
from torch.utils.data import DataLoader, Dataset

from mias_dcms.zero_shot_scoring import build_classification_messages, label_codes, sequence_log_likelihoods


def classification_sft_record(row: dict[str, Any], label_names: Sequence[str]) -> dict[str, Any]:
    codes = label_codes(len(label_names))
    label = int(row["label"])
    return {
        "id": str(row["id"]),
        "messages": build_classification_messages(str(row["text"]), label_names),
        "answer": codes[label],
    }


def preference_dpo_record(row: dict[str, Any]) -> dict[str, Any] | None:
    preferred_response = int(row.get("preferred_response", 0))
    if preferred_response not in (1, 2):
        return None
    response_1 = str(row["response_1"])
    response_2 = str(row["response_2"])
    return {
        "id": str(row["id"]),
        "messages": [{"role": "user", "content": str(row["prompt"])}],
        "chosen": response_1 if preferred_response == 1 else response_2,
        "rejected": response_2 if preferred_response == 1 else response_1,
    }


def dpo_loss(
    *,
    policy_chosen: torch.Tensor,
    policy_rejected: torch.Tensor,
    reference_chosen: torch.Tensor,
    reference_rejected: torch.Tensor,
    beta: float,
) -> torch.Tensor:
    if beta <= 0.0:
        raise ValueError("beta must be positive")
    policy_log_ratio = policy_chosen - policy_rejected
    reference_log_ratio = reference_chosen - reference_rejected
    return -functional.logsigmoid(beta * (policy_log_ratio - reference_log_ratio)).mean()


@dataclass(frozen=True)
class LoraTrainingConfig:
    model_name_or_path: str
    output_dir: Path
    epochs: int = 1
    learning_rate: float = 2e-4
    batch_size: int = 2
    gradient_accumulation_steps: int = 8
    max_length: int = 2048
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    target_modules: tuple[str, ...] = ("q_proj", "k_proj", "v_proj", "o_proj")
    weight_decay: float = 0.0
    warmup_ratio: float = 0.03
    max_grad_norm: float = 1.0
    mixed_precision: str = "no"
    dtype: str = "auto"
    gradient_checkpointing: bool = True
    seed: int = 42
    num_workers: int = 0
    beta: float = 0.1
    update_steps: int | None = None
    train_token_budget: int | None = None
    initial_policy_adapter_path: str | None = None
    reference_adapter_path: str | None = None


class ClassificationSFTDataset(Dataset[dict[str, list[int]]]):
    def __init__(
        self,
        rows: list[dict[str, Any]],
        *,
        label_names: Sequence[str],
        tokenizer: Any,
        max_length: int,
    ) -> None:
        self.examples = [
            _encode_sft_record(
                classification_sft_record(row, label_names),
                tokenizer=tokenizer,
                max_length=max_length,
            )
            for row in rows
        ]

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, list[int]]:
        return self.examples[index]


class PreferenceDPODataset(Dataset[dict[str, list[int] | int]]):
    def __init__(
        self,
        rows: list[dict[str, Any]],
        *,
        tokenizer: Any,
        max_length: int,
    ) -> None:
        records = [record for row in rows if (record := preference_dpo_record(row)) is not None]
        self.examples = [
            _encode_dpo_record(record, tokenizer=tokenizer, max_length=max_length) for record in records
        ]

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, list[int] | int]:
        return self.examples[index]


def train_classification_lora(
    rows: list[dict[str, Any]],
    *,
    label_names: Sequence[str],
    config: LoraTrainingConfig,
) -> dict[str, Any]:
    tokenizer, model = _load_lora_policy(config)
    dataset = ClassificationSFTDataset(
        rows,
        label_names=label_names,
        tokenizer=tokenizer,
        max_length=config.max_length,
    )
    if not dataset:
        raise ValueError("classification training dataset is empty")
    dataloader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=lambda batch: _collate_sft(batch, int(tokenizer.pad_token_id)),
        num_workers=config.num_workers,
    )
    summary = _run_sft_loop(model=model, tokenizer=tokenizer, dataloader=dataloader, config=config)
    summary.update({"task": "classification_lora_sft", "train_size": len(dataset)})
    _write_training_summary(config.output_dir, summary)
    return summary


def train_preference_dpo(
    rows: list[dict[str, Any]],
    *,
    config: LoraTrainingConfig,
) -> dict[str, Any]:
    tokenizer, policy = _load_lora_policy(config)
    reference = _load_reference_model(config)
    dataset = PreferenceDPODataset(rows, tokenizer=tokenizer, max_length=config.max_length)
    if not dataset:
        raise ValueError("preference training dataset has no non-tie rows")
    dataloader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=lambda batch: _collate_dpo(batch, int(tokenizer.pad_token_id)),
        num_workers=config.num_workers,
    )
    summary = _run_dpo_loop(
        policy=policy,
        reference=reference,
        tokenizer=tokenizer,
        dataloader=dataloader,
        config=config,
    )
    summary.update({"task": "preference_lora_dpo", "train_size": len(dataset), "beta": config.beta})
    _write_training_summary(config.output_dir, summary)
    return summary


def _encode_sft_record(record: dict[str, Any], *, tokenizer: Any, max_length: int) -> dict[str, list[int]]:
    prompt_ids = _prompt_token_ids(tokenizer, record["messages"])
    completion_ids = _completion_token_ids(tokenizer, str(record["answer"]))
    prompt_ids = _truncate_prompt(prompt_ids, completion_ids, max_length=max_length)
    input_ids = prompt_ids + completion_ids
    return {
        "input_ids": input_ids,
        "attention_mask": [1] * len(input_ids),
        "labels": [-100] * len(prompt_ids) + completion_ids,
    }


def _encode_dpo_record(
    record: dict[str, Any],
    *,
    tokenizer: Any,
    max_length: int,
) -> dict[str, list[int] | int]:
    prompt_ids = _prompt_token_ids(tokenizer, record["messages"])
    chosen_ids = _completion_token_ids(tokenizer, str(record["chosen"]))
    rejected_ids = _completion_token_ids(tokenizer, str(record["rejected"]))
    available_prompt_length = max_length - max(len(chosen_ids), len(rejected_ids))
    if available_prompt_length <= 0:
        raise ValueError("chosen or rejected response exceeds max_length")
    prompt_ids = prompt_ids[-available_prompt_length:]
    if not prompt_ids:
        raise ValueError("prompt must retain at least one token")
    return {
        "chosen_input_ids": prompt_ids + chosen_ids,
        "chosen_prompt_length": len(prompt_ids),
        "rejected_input_ids": prompt_ids + rejected_ids,
        "rejected_prompt_length": len(prompt_ids),
    }


def _prompt_token_ids(tokenizer: Any, messages: Sequence[dict[str, str]]) -> list[int]:
    try:
        rendered = tokenizer.apply_chat_template(
            list(messages),
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        rendered = tokenizer.apply_chat_template(
            list(messages), tokenize=False, add_generation_prompt=True
        )
    return list(tokenizer(rendered, add_special_tokens=False)["input_ids"])


def _completion_token_ids(tokenizer: Any, completion: str) -> list[int]:
    token_ids = list(tokenizer(completion, add_special_tokens=False)["input_ids"])
    if tokenizer.eos_token_id is not None:
        token_ids.append(int(tokenizer.eos_token_id))
    if not token_ids:
        raise ValueError("completion tokenized to an empty sequence")
    return token_ids


def _truncate_prompt(prompt_ids: list[int], completion_ids: list[int], *, max_length: int) -> list[int]:
    available_prompt_length = max_length - len(completion_ids)
    if available_prompt_length <= 0:
        raise ValueError("completion exceeds max_length")
    truncated = prompt_ids[-available_prompt_length:]
    if not truncated:
        raise ValueError("prompt must retain at least one token")
    return truncated


def _collate_sft(batch: Sequence[dict[str, list[int]]], pad_token_id: int) -> dict[str, torch.Tensor]:
    maximum_length = max(len(item["input_ids"]) for item in batch)
    input_ids: list[list[int]] = []
    attention_mask: list[list[int]] = []
    labels: list[list[int]] = []
    for item in batch:
        padding = maximum_length - len(item["input_ids"])
        input_ids.append(item["input_ids"] + [pad_token_id] * padding)
        attention_mask.append(item["attention_mask"] + [0] * padding)
        labels.append(item["labels"] + [-100] * padding)
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
    }


def _collate_dpo(
    batch: Sequence[dict[str, list[int] | int]], pad_token_id: int
) -> dict[str, torch.Tensor]:
    chosen = [list(item["chosen_input_ids"]) for item in batch]
    rejected = [list(item["rejected_input_ids"]) for item in batch]
    sequences = chosen + rejected
    prompt_lengths = [int(item["chosen_prompt_length"]) for item in batch] + [
        int(item["rejected_prompt_length"]) for item in batch
    ]
    maximum_length = max(len(sequence) for sequence in sequences)
    input_ids = [sequence + [pad_token_id] * (maximum_length - len(sequence)) for sequence in sequences]
    attention_mask = [
        [1] * len(sequence) + [0] * (maximum_length - len(sequence)) for sequence in sequences
    ]
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        "prompt_lengths": torch.tensor(prompt_lengths, dtype=torch.long),
        "pair_count": torch.tensor(len(batch), dtype=torch.long),
        "pair_input_token_counts": torch.tensor(
            [len(chosen_item) + len(rejected_item) for chosen_item, rejected_item in zip(chosen, rejected, strict=True)],
            dtype=torch.long,
        ),
    }


def _load_lora_policy(config: LoraTrainingConfig) -> tuple[Any, Any]:
    from peft import LoraConfig, PeftModel, TaskType, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    policy_path = Path(config.initial_policy_adapter_path) if config.initial_policy_adapter_path else None
    if policy_path is not None and not policy_path.is_dir():
        raise FileNotFoundError(
            f"initial policy adapter directory does not exist: {policy_path}"
        )
    tokenizer_source = str(policy_path) if policy_path is not None else config.model_name_or_path
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name_or_path,
        dtype=_resolve_dtype(config.dtype),
        low_cpu_mem_usage=True,
    )
    model.config.use_cache = False
    if config.gradient_checkpointing:
        model.gradient_checkpointing_enable()
    if policy_path is not None:
        return tokenizer, PeftModel.from_pretrained(
            model,
            policy_path,
            is_trainable=True,
        )
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=list(config.target_modules),
        bias="none",
    )
    return tokenizer, get_peft_model(model, lora_config)


def _load_reference_model(config: LoraTrainingConfig) -> Any:
    from peft import PeftModel
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        config.model_name_or_path,
        dtype=_resolve_dtype(config.dtype),
        low_cpu_mem_usage=True,
    )
    model.config.use_cache = False
    reference_adapter_path = config.reference_adapter_path or config.initial_policy_adapter_path
    if reference_adapter_path:
        adapter_path = Path(reference_adapter_path)
        if not adapter_path.is_dir():
            raise FileNotFoundError(
                f"reference adapter directory does not exist: {adapter_path}"
            )
        model = PeftModel.from_pretrained(model, adapter_path, is_trainable=False)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def _run_sft_loop(*, model: Any, tokenizer: Any, dataloader: DataLoader, config: LoraTrainingConfig) -> dict[str, Any]:
    from accelerate import Accelerator
    from transformers import get_scheduler, set_seed

    set_seed(config.seed)
    accelerator = Accelerator(
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        mixed_precision=config.mixed_precision,
    )
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    update_steps_per_epoch = math.ceil(len(dataloader) / config.gradient_accumulation_steps)
    total_steps = int(config.update_steps or max(1, update_steps_per_epoch * config.epochs))
    scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=int(total_steps * config.warmup_ratio),
        num_training_steps=total_steps,
    )
    model, optimizer, dataloader, scheduler = accelerator.prepare(
        model, optimizer, dataloader, scheduler
    )
    losses: list[float] = []
    model.train()
    optimizer.zero_grad()
    completed_steps = 0
    epoch = 0
    while completed_steps < total_steps:
        epoch += 1
        for batch in dataloader:
            with accelerator.accumulate(model):
                loss = model(**batch).loss
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), config.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
            losses.append(float(loss.detach().float().item()))
            if accelerator.sync_gradients:
                completed_steps += 1
                if completed_steps >= total_steps:
                    break
    _save_adapter(accelerator=accelerator, model=model, tokenizer=tokenizer, output_dir=config.output_dir)
    return {
        "epochs": epoch,
        "optimizer_steps": completed_steps,
        "mean_train_loss": sum(losses) / len(losses),
        "model_name_or_path": config.model_name_or_path,
    }


def _run_dpo_loop(
    *,
    policy: Any,
    reference: Any,
    tokenizer: Any,
    dataloader: DataLoader,
    config: LoraTrainingConfig,
) -> dict[str, Any]:
    from accelerate import Accelerator
    from transformers import get_scheduler, set_seed

    set_seed(config.seed)
    accelerator = Accelerator(mixed_precision=config.mixed_precision)
    optimizer = torch.optim.AdamW(
        (parameter for parameter in policy.parameters() if parameter.requires_grad),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    update_steps_per_epoch = math.ceil(len(dataloader) / config.gradient_accumulation_steps)
    token_budget = config.train_token_budget
    if token_budget is not None and token_budget <= 0:
        raise ValueError("train_token_budget must be positive when provided")
    total_steps = _dpo_scheduler_steps(
        dataloader=dataloader,
        config=config,
        update_steps_per_epoch=update_steps_per_epoch,
    )
    scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=int(total_steps * config.warmup_ratio),
        num_training_steps=total_steps,
    )
    policy, reference, optimizer, dataloader, scheduler = accelerator.prepare(
        policy, reference, optimizer, dataloader, scheduler
    )
    reference.eval()
    policy.train()
    optimizer.zero_grad()
    losses: list[float] = []
    accuracies: list[float] = []
    completed_steps = 0
    epoch = 0
    processed_pair_count = 0
    processed_input_tokens = 0
    accumulation_steps = max(1, int(config.gradient_accumulation_steps))
    while token_budget is None or processed_input_tokens < token_budget:
        epoch += 1
        made_progress = False
        accumulation_group: list[dict[str, torch.Tensor]] = []
        for batch in dataloader:
            if token_budget is not None:
                batch = _trim_dpo_batch_to_token_budget(
                    batch,
                    remaining_tokens=token_budget - processed_input_tokens,
                )
                if batch is None:
                    continue
            pair_count = int(batch.pop("pair_count").item())
            pair_input_token_counts = batch.pop("pair_input_token_counts")
            batch_token_count = int(pair_input_token_counts.sum().detach().item())
            processed_pair_count += pair_count
            processed_input_tokens += batch_token_count
            made_progress = True
            batch["pair_count"] = torch.tensor(pair_count, dtype=torch.long)
            batch["pair_input_token_counts"] = pair_input_token_counts
            accumulation_group.append(batch)
            budget_exhausted = token_budget is not None and processed_input_tokens >= token_budget
            if len(accumulation_group) >= accumulation_steps or budget_exhausted:
                group_losses, group_accuracies = _run_dpo_accumulation_group(
                    accumulation_group,
                    policy=policy,
                    reference=reference,
                    accelerator=accelerator,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    beta=config.beta,
                    max_grad_norm=config.max_grad_norm,
                )
                losses.extend(group_losses)
                accuracies.extend(group_accuracies)
                accumulation_group = []
                completed_steps += 1
                if token_budget is None and completed_steps >= total_steps:
                    break
                if budget_exhausted:
                    break
        if accumulation_group and (token_budget is not None or completed_steps < total_steps):
            group_losses, group_accuracies = _run_dpo_accumulation_group(
                accumulation_group,
                policy=policy,
                reference=reference,
                accelerator=accelerator,
                optimizer=optimizer,
                scheduler=scheduler,
                beta=config.beta,
                max_grad_norm=config.max_grad_norm,
            )
            losses.extend(group_losses)
            accuracies.extend(group_accuracies)
            completed_steps += 1
        if token_budget is None and completed_steps >= total_steps:
            break
        if token_budget is not None and not made_progress:
            break
    if not losses:
        raise ValueError(
            "train_token_budget is smaller than every complete DPO pair in the training dataset"
        )
    _save_adapter(
        accelerator=accelerator,
        model=policy,
        tokenizer=tokenizer,
        output_dir=config.output_dir,
    )
    return {
        "epochs": epoch,
        "optimizer_steps": completed_steps,
        "mean_train_loss": sum(losses) / len(losses),
        "mean_policy_preference_accuracy": sum(accuracies) / len(accuracies),
        "processed_pair_count": processed_pair_count,
        "processed_input_tokens": processed_input_tokens,
        "train_token_budget": token_budget,
        "unused_train_token_budget": (
            int(token_budget - processed_input_tokens) if token_budget is not None else None
        ),
        "token_budget_exhausted": (
            bool(processed_input_tokens == token_budget) if token_budget is not None else None
        ),
        "scheduler_steps": total_steps,
        "reference_adapter_path": config.reference_adapter_path or config.initial_policy_adapter_path,
        "model_name_or_path": config.model_name_or_path,
    }


def _run_dpo_accumulation_group(
    batches: Sequence[dict[str, torch.Tensor]],
    *,
    policy: Any,
    reference: Any,
    accelerator: Any,
    optimizer: Any,
    scheduler: Any,
    beta: float,
    max_grad_norm: float,
) -> tuple[list[float], list[float]]:
    if not batches:
        raise ValueError("DPO accumulation group must not be empty")
    pair_counts = [int(batch["pair_count"].item()) for batch in batches]
    total_pairs = sum(pair_counts)
    losses: list[float] = []
    accuracies: list[float] = []
    for index, (batch, pair_count) in enumerate(zip(batches, pair_counts, strict=True)):
        sync_context = (
            nullcontext()
            if index == len(batches) - 1
            else accelerator.no_sync(policy)
        )
        with sync_context:
            policy_output = policy(
                input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]
            )
            policy_scores = sequence_log_likelihoods(
                logits=policy_output.logits,
                input_ids=batch["input_ids"],
                prompt_lengths=batch["prompt_lengths"],
                attention_mask=batch["attention_mask"],
            )
            with torch.inference_mode():
                reference_output = reference(
                    input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]
                )
                reference_scores = sequence_log_likelihoods(
                    logits=reference_output.logits,
                    input_ids=batch["input_ids"],
                    prompt_lengths=batch["prompt_lengths"],
                    attention_mask=batch["attention_mask"],
                )
            loss = dpo_loss(
                policy_chosen=policy_scores[:pair_count],
                policy_rejected=policy_scores[pair_count:],
                reference_chosen=reference_scores[:pair_count],
                reference_rejected=reference_scores[pair_count:],
                beta=beta,
            )
            accelerator.backward(loss * (pair_count / total_pairs))
        losses.append(float(loss.detach().float().item()))
        accuracies.append(
            float(
                (policy_scores[:pair_count] > policy_scores[pair_count:])
                .float()
                .mean()
                .item()
            )
        )
    accelerator.clip_grad_norm_(policy.parameters(), max_grad_norm)
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad()
    return losses, accuracies


def _dpo_scheduler_steps(
    *,
    dataloader: DataLoader,
    config: LoraTrainingConfig,
    update_steps_per_epoch: int,
) -> int:
    """Estimate a scheduler horizon without using it as a token-budget stop condition."""
    if config.train_token_budget is None:
        return int(config.update_steps or max(1, update_steps_per_epoch * config.epochs))

    dataset = getattr(dataloader, "dataset", None)
    examples = getattr(dataset, "examples", ())
    token_counts = [
        len(example["chosen_input_ids"]) + len(example["rejected_input_ids"])
        for example in examples
    ]
    if not token_counts:
        return max(1, int(config.update_steps or 1))
    mean_pair_tokens = sum(token_counts) / len(token_counts)
    estimated_micro_batches = math.ceil(
        int(config.train_token_budget) / max(1.0, mean_pair_tokens * config.batch_size)
    )
    estimated_updates = math.ceil(estimated_micro_batches / config.gradient_accumulation_steps)
    return max(1, estimated_updates)


def _trim_dpo_batch_to_token_budget(
    batch: dict[str, torch.Tensor],
    *,
    remaining_tokens: int,
) -> dict[str, torch.Tensor] | None:
    """Keep a deterministic prefix of complete pairs that fits the remaining budget."""
    if remaining_tokens <= 0:
        return None
    pair_count = int(batch["pair_count"].item())
    pair_tokens = [int(value) for value in batch["pair_input_token_counts"].tolist()]
    retained_count = 0
    retained_tokens = 0
    for pair_tokens_count in pair_tokens:
        if retained_tokens + pair_tokens_count > remaining_tokens:
            break
        retained_count += 1
        retained_tokens += pair_tokens_count
    if retained_count == 0:
        return None
    if retained_count == pair_count:
        return batch

    sequence_indexes = [*range(retained_count), *range(pair_count, pair_count + retained_count)]
    trimmed = {
        "input_ids": batch["input_ids"][sequence_indexes],
        "attention_mask": batch["attention_mask"][sequence_indexes],
        "prompt_lengths": batch["prompt_lengths"][sequence_indexes],
        "pair_count": torch.tensor(retained_count, dtype=torch.long),
        "pair_input_token_counts": batch["pair_input_token_counts"][:retained_count],
    }
    return trimmed


def _save_adapter(*, accelerator: Any, model: Any, tokenizer: Any, output_dir: Path) -> None:
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        output_dir.mkdir(parents=True, exist_ok=True)
        unwrapped = accelerator.unwrap_model(model)
        unwrapped.save_pretrained(output_dir, save_function=accelerator.save)
        tokenizer.save_pretrained(output_dir)
    accelerator.wait_for_everyone()


def _resolve_dtype(value: str) -> str | torch.dtype:
    normalized = value.lower()
    if normalized == "auto":
        return "auto"
    mapping = {
        "float32": torch.float32,
        "fp32": torch.float32,
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
    }
    if normalized not in mapping:
        raise ValueError(f"unsupported dtype: {value}")
    return mapping[normalized]


def _write_training_summary(output_dir: Path, summary: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "training_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
