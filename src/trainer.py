"""Training and evaluation loops."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import get_cosine_schedule_with_warmup, get_linear_schedule_with_warmup

from .metrics import compute_binary_metrics
from .utils import count_parameters, move_batch_to_device, write_json, write_jsonl


def get_single_token_id(tokenizer: Any, text: str) -> int:
    """Return the token id for a single-token generation target."""
    token_ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    if len(token_ids) != 1:
        raise ValueError(f"Expected {text!r} to encode to one token, got {token_ids}")
    return int(token_ids[0])


def last_non_padding_logits(logits: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Select logits at the final non-padding position for each row."""
    positions = torch.arange(logits.size(1), device=logits.device)
    positions = positions.unsqueeze(0).expand_as(attention_mask)
    last_indices = positions.masked_fill(attention_mask == 0, 0).max(dim=1).values
    batch_indices = torch.arange(logits.size(0), device=logits.device)
    return logits[batch_indices, last_indices]


def build_optimizer(
    model: torch.nn.Module,
    lr: float,
    weight_decay: float,
) -> torch.optim.Optimizer:
    """Create AdamW over trainable parameters only."""
    parameters = [param for param in model.parameters() if param.requires_grad]
    if not parameters:
        raise ValueError("No trainable parameters found.")
    return torch.optim.AdamW(parameters, lr=lr, weight_decay=weight_decay)


def train_one_epoch(
    model: torch.nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    device: torch.device,
    gradient_accumulation_steps: int = 1,
    max_grad_norm: float = 1.0,
    class_token_weights: dict[int, float] | None = None,
) -> float:
    """Train for one epoch and return average loss."""
    if gradient_accumulation_steps < 1:
        raise ValueError("gradient_accumulation_steps must be at least 1")

    model.train()
    optimizer.zero_grad(set_to_none=True)
    trainable_parameters = [param for param in model.parameters() if param.requires_grad]
    total_loss = 0.0
    step_count = 0

    progress = tqdm(dataloader, desc="train", leave=False)
    for step, batch in enumerate(progress, start=1):
        batch = move_batch_to_device(batch, device)
        sample_weights = batch.pop("sample_weights", None)
        outputs = model(
            **batch,
            class_token_weights=class_token_weights,
            sample_weights=sample_weights,
        )
        accumulation_start = ((step - 1) // gradient_accumulation_steps) * gradient_accumulation_steps + 1
        accumulation_end = min(accumulation_start + gradient_accumulation_steps - 1, len(dataloader))
        current_accumulation_steps = accumulation_end - accumulation_start + 1
        loss = outputs["loss"] / current_accumulation_steps
        loss.backward()

        total_loss += float(outputs["loss"].detach().cpu())
        step_count += 1

        if step % gradient_accumulation_steps == 0 or step == len(dataloader):
            if max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(trainable_parameters, max_grad_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

        progress.set_postfix(loss=f"{total_loss / step_count:.4f}")

    return total_loss / max(step_count, 1)


@torch.no_grad()
def evaluate_model(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    threshold: float = 0.0,
    predictions_path: str | Path | None = None,
    tokenizer: Any | None = None,
    negative_token_text: str = "0",
    positive_token_text: str = "1",
) -> dict[str, Any]:
    """Evaluate a model and return metrics."""
    if tokenizer is None:
        raise ValueError("Generative evaluation requires a tokenizer.")

    model.eval()
    negative_token_id = get_single_token_id(tokenizer, negative_token_text)
    positive_token_id = get_single_token_id(tokenizer, positive_token_text)
    losses: list[float] = []
    labels: list[int] = []
    scores: list[float] = []
    rows: list[dict[str, Any]] = []
    score_source = f"{positive_token_text}_minus_{negative_token_text}_logit_margin"

    for batch in tqdm(dataloader, desc="eval", leave=False):
        sample_ids = list(batch.get("sample_ids", []))
        batch = move_batch_to_device(batch, device)
        outputs = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
        )
        prompt_outputs = model(
            input_ids=batch["prompt_input_ids"],
            attention_mask=batch["prompt_attention_mask"],
        )
        next_token_logits = last_non_padding_logits(
            prompt_outputs.logits,
            batch["prompt_attention_mask"],
        )
        binary_logits = next_token_logits[:, [negative_token_id, positive_token_id]]
        float_logits = binary_logits.float()
        batch_scores = (float_logits[:, 1] - float_logits[:, 0]).detach().cpu().tolist()
        probs = torch.softmax(float_logits, dim=-1)[:, 1].detach().cpu().tolist()
        batch_labels = batch["target_labels"].detach().cpu().int().tolist()

        if getattr(outputs, "loss", None) is not None:
            losses.append(float(outputs.loss.detach().cpu()))
        labels.extend(batch_labels)
        scores.extend(batch_scores)

        for sample_id, label, score, probability in zip(sample_ids, batch_labels, batch_scores, probs):
            prediction = 1 if float(score) > threshold else 0
            rows.append(
                {
                    "id": sample_id,
                    "label": int(label),
                    "score": float(score),
                    "probability": float(probability),
                    "prediction": prediction,
                    # 对外工件统一使用 1/0，避免模型协议里的 yes/no 泄漏到算法输入。
                    "generated_text": str(int(prediction)),
                    "score_source": score_source,
                }
            )

    metrics = compute_binary_metrics(labels, scores, threshold=threshold)
    if losses:
        metrics["loss"] = sum(losses) / len(losses)
    if predictions_path is not None:
        write_jsonl(rows, predictions_path)
    return metrics


@torch.no_grad()
def predict_model(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    tokenizer: Any,
    threshold: float = 0.0,
    negative_token_text: str = "0",
    positive_token_text: str = "1",
    predictions_path: str | Path | None = None,
) -> list[dict[str, Any]]:
    """Run prompt-only binary scoring and return per-sample prediction rows."""
    model.eval()
    negative_token_id = get_single_token_id(tokenizer, negative_token_text)
    positive_token_id = get_single_token_id(tokenizer, positive_token_text)
    rows: list[dict[str, Any]] = []
    score_source = f"{positive_token_text}_minus_{negative_token_text}_logit_margin"
    writer = None
    if predictions_path is not None:
        target = Path(predictions_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        writer = target.open("w", encoding="utf-8")

    try:
        for batch in tqdm(dataloader, desc="predict", leave=False):
            sample_ids = list(batch.get("sample_ids", []))
            batch = move_batch_to_device(batch, device)
            prompt_outputs = model(
                input_ids=batch["prompt_input_ids"],
                attention_mask=batch["prompt_attention_mask"],
            )
            next_token_logits = last_non_padding_logits(
                prompt_outputs.logits,
                batch["prompt_attention_mask"],
            )
            binary_logits = next_token_logits[:, [negative_token_id, positive_token_id]]
            float_logits = binary_logits.float()
            batch_scores = (float_logits[:, 1] - float_logits[:, 0]).detach().cpu().tolist()
            probs = torch.softmax(float_logits, dim=-1)[:, 1].detach().cpu().tolist()
            batch_labels = batch["target_labels"].detach().cpu().int().tolist()

            for sample_id, label, score, probability in zip(sample_ids, batch_labels, batch_scores, probs):
                prediction = 1 if float(score) > threshold else 0
                row = {
                    "id": str(sample_id),
                    "label": int(label),
                    "score": float(score),
                    "probability": float(probability),
                    "prediction": int(prediction),
                    # 对外工件统一使用 1/0，避免模型协议里的 yes/no 泄漏到算法输入。
                    "generated_text": str(int(prediction)),
                    "score_source": score_source,
                }
                rows.append(row)
                if writer is not None:
                    writer.write(json.dumps(row, ensure_ascii=False))
                    writer.write("\n")
            if writer is not None:
                writer.flush()
    finally:
        if writer is not None:
            writer.close()
    return rows


def fit(
    model: torch.nn.Module,
    train_loader: DataLoader,
    eval_loader: DataLoader | None,
    tokenizer: Any,
    output_dir: str | Path,
    device: torch.device,
    epochs: int,
    lr: float,
    weight_decay: float,
    gradient_accumulation_steps: int,
    max_grad_norm: float,
    warmup_ratio: float,
    threshold: float,
    run_config: dict[str, Any],
    early_stopping_patience: int | None = None,
    early_stopping_min_delta: float = 0.0,
    class_token_weights: dict[int, float] | None = None,
    scheduler_type: str = "linear",
) -> dict[str, Any]:
    """Train a model and save a final checkpoint."""
    model.to(device)
    optimizer = build_optimizer(model, lr=lr, weight_decay=weight_decay)

    updates_per_epoch = (len(train_loader) + gradient_accumulation_steps - 1) // gradient_accumulation_steps
    total_training_steps = max(1, updates_per_epoch * epochs)
    warmup_steps = int(total_training_steps * warmup_ratio)
    # Scheduler steps once per optimizer update, not once per dataloader batch.
    if scheduler_type == "linear":
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_training_steps,
        )
    elif scheduler_type == "cosine":
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_training_steps,
        )
    else:
        raise ValueError(f"Unsupported scheduler_type: {scheduler_type}")

    history: list[dict[str, Any]] = []
    best_metric = float("-inf")
    best_epoch: int | None = None
    epochs_without_improvement = 0
    early_stopped = False
    print(json.dumps({"parameters": count_parameters(model)}, ensure_ascii=False))

    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            gradient_accumulation_steps=gradient_accumulation_steps,
            max_grad_norm=max_grad_norm,
            class_token_weights=class_token_weights,
        )
        epoch_record: dict[str, Any] = {"epoch": epoch, "train_loss": train_loss}

        if eval_loader is not None:
            eval_metrics = evaluate_model(
                model=model,
                dataloader=eval_loader,
                device=device,
                threshold=threshold,
                tokenizer=tokenizer,
            )
            epoch_record["eval"] = eval_metrics

        history.append(epoch_record)
        print(json.dumps(epoch_record, ensure_ascii=False, sort_keys=True))
        # 每个 epoch 完成后立即落盘，长时间 GPU 训练中断时仍可复查已完成进度。
        output_path = Path(output_dir)
        write_jsonl(history, output_path / "training_history.jsonl")
        write_json(
            {
                "history": history,
                "requested_epochs": epochs,
                "completed_epochs": len(history),
                "early_stopped": False,
                "run_complete": False,
            },
            output_path / "training_state.json",
        )

        if eval_loader is not None:
            metric = float(epoch_record["eval"]["f1"])
            if metric > best_metric + early_stopping_min_delta:
                best_metric = metric
                best_epoch = epoch
                epochs_without_improvement = 0
                checkpoint_config = dict(run_config)
                checkpoint_config["history"] = list(history)
                checkpoint_config["best_epoch"] = best_epoch
                checkpoint_config["best_metric"] = "f1"
                checkpoint_config["best_eval"] = epoch_record["eval"]
                checkpoint_config["requested_epochs"] = epochs
                checkpoint_config["completed_epochs"] = epoch
                checkpoint_config["early_stopped"] = False
                checkpoint_config["run_complete"] = False
                model.save_checkpoint(
                    output_dir,
                    tokenizer=tokenizer,
                    extra_config=checkpoint_config,
                )
            else:
                epochs_without_improvement += 1
                if (
                    early_stopping_patience is not None
                    and early_stopping_patience >= 0
                    and epochs_without_improvement >= early_stopping_patience
                ):
                    print(
                        json.dumps(
                            {
                                "early_stopping": {
                                    "best_epoch": best_epoch,
                                    "best_metric": best_metric,
                                    "epoch": epoch,
                                    "patience": early_stopping_patience,
                                }
                            },
                            ensure_ascii=False,
                            sort_keys=True,
                        )
                    )
                    early_stopped = True
                    break

    final_config = dict(run_config)
    final_config["history"] = history
    final_config["requested_epochs"] = epochs
    final_config["completed_epochs"] = len(history)
    final_config["early_stopped"] = early_stopped
    final_config["run_complete"] = True
    if eval_loader is None:
        model.save_checkpoint(output_dir, tokenizer=tokenizer, extra_config=final_config)
    else:
        final_config["best_epoch"] = best_epoch
        final_config["best_metric"] = "f1"
        final_config["best_eval"] = history[best_epoch - 1]["eval"] if best_epoch is not None else None
        checkpoint_config = model.checkpoint_config()
        checkpoint_config.update(final_config)
        write_json(checkpoint_config, Path(output_dir) / "model_config.json")
    write_json(final_config, Path(output_dir) / "training_state.json")
    return {"history": history}
