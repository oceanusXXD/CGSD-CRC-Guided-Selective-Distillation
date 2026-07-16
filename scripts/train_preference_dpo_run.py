from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mias_dcms.benchmark_training import LoraTrainingConfig, train_preference_dpo
from mias_dcms.preference_run_summary import estimate_preference_train_tokens
from mias_dcms.utils import read_json, read_jsonl, resolve_model_reference, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train one DPO preference run from revealed rows.")
    parser.add_argument("--dpo_train_rows_path", type=Path, required=True)
    parser.add_argument("--selection_summary_path", type=Path)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--training_summary_path", type=Path, required=True)
    parser.add_argument("--cost_report_path", type=Path, required=True)
    parser.add_argument("--model_name_or_path", required=True)
    parser.add_argument("--training_config_json", default="{}")
    parser.add_argument("--training_config_path", type=Path)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--method", required=True)
    parser.add_argument("--budget", type=int, required=True)
    parser.add_argument("--evaluation_label_count", type=int, default=0)
    parser.add_argument("--seed_label_count", type=int, default=0)
    parser.add_argument("--judge_calls", type=int, default=0)
    parser.add_argument("--selector_compute_seconds", type=float, default=0.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = read_jsonl(args.dpo_train_rows_path)
    training_config = _load_training_config(args)
    config = _lora_config_from_payload(
        training_config,
        model_name_or_path=resolve_model_reference(str(args.model_name_or_path), PROJECT_ROOT),
        output_dir=args.output_dir,
        seed=int(args.seed),
    )
    backend_summary = train_preference_dpo(rows, config=config)
    selection_summary = (
        read_json(args.selection_summary_path)
        if args.selection_summary_path is not None
        else {}
    )
    training_metrics = _training_metrics(
        backend_summary,
        rows=rows,
        training_config=training_config,
    )
    training_summary = {
        "task": "preference_lora_dpo",
        "method": str(args.method),
        "budget": int(args.budget),
        "seed": int(args.seed),
        "model_name_or_path": str(args.model_name_or_path),
        "dpo_train_rows_path": str(args.dpo_train_rows_path),
        "output_dir": str(args.output_dir),
        "training_config": training_config,
        "input_sha256": _sha256_file(args.dpo_train_rows_path),
        "runtime_environment": _runtime_environment(),
        "training_metrics": training_metrics,
        "backend_summary": backend_summary,
    }
    write_json(training_summary, args.training_summary_path)
    cost_report = {
        "seed_label_count": int(args.seed_label_count),
        "evaluation_label_count": int(args.evaluation_label_count),
        "judge_calls": int(args.judge_calls),
        "selector_compute_seconds": float(
            selection_summary.get(
                "selector_compute_seconds",
                args.selector_compute_seconds,
            )
        ),
        "train_tokens": int(
            backend_summary.get("processed_input_tokens", estimate_preference_train_tokens(rows))
        ),
        "one_pass_train_token_estimate": estimate_preference_train_tokens(rows),
        "oracle_label_calls": int(args.budget),
    }
    write_json(cost_report, args.cost_report_path)
    print(json.dumps(training_summary, ensure_ascii=False, sort_keys=True))


def _load_training_config(args: argparse.Namespace) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    if args.training_config_path is not None:
        payload.update(read_json(args.training_config_path))
    inline_payload = json.loads(str(args.training_config_json))
    if not isinstance(inline_payload, dict):
        raise ValueError("--training_config_json must decode to an object")
    payload.update(inline_payload)
    return payload


def _lora_config_from_payload(
    payload: dict[str, Any],
    *,
    model_name_or_path: str,
    output_dir: Path,
    seed: int,
) -> LoraTrainingConfig:
    return LoraTrainingConfig(
        model_name_or_path=model_name_or_path,
        output_dir=output_dir,
        epochs=int(payload.get("epochs", 1)),
        learning_rate=float(payload.get("learning_rate", 2e-4)),
        batch_size=int(payload.get("batch_size", 2)),
        gradient_accumulation_steps=int(payload.get("gradient_accumulation_steps", 8)),
        max_length=int(payload.get("max_length", 2048)),
        lora_r=int(payload.get("lora_r", 8)),
        lora_alpha=int(payload.get("lora_alpha", 16)),
        lora_dropout=float(payload.get("lora_dropout", 0.05)),
        target_modules=tuple(str(value) for value in payload.get("target_modules", ("q_proj", "k_proj", "v_proj", "o_proj"))),
        weight_decay=float(payload.get("weight_decay", 0.0)),
        warmup_ratio=float(payload.get("warmup_ratio", 0.03)),
        max_grad_norm=float(payload.get("max_grad_norm", 1.0)),
        mixed_precision=str(payload.get("mixed_precision", "no")),
        dtype=str(payload.get("dtype", "auto")),
        gradient_checkpointing=bool(payload.get("gradient_checkpointing", True)),
        seed=seed,
        num_workers=int(payload.get("num_workers", 0)),
        beta=float(payload.get("beta", 0.1)),
        update_steps=(
            int(payload["update_steps"])
            if payload.get("update_steps") is not None
            else None
        ),
        train_token_budget=(
            int(payload["train_token_budget"])
            if payload.get("train_token_budget") is not None
            else None
        ),
        initial_policy_adapter_path=(
            str(payload["initial_policy_adapter_path"])
            if payload.get("initial_policy_adapter_path")
            else None
        ),
        reference_adapter_path=(
            str(payload["reference_adapter_path"])
            if payload.get("reference_adapter_path")
            else None
        ),
    )


def _training_metrics(
    backend_summary: dict[str, Any],
    *,
    rows: list[dict[str, Any]],
    training_config: dict[str, Any],
) -> dict[str, Any]:
    configured_update_steps = training_config.get("update_steps")
    observed_steps = backend_summary.get("optimizer_steps")
    reported_token_budget = backend_summary.get("train_token_budget")
    configured_token_budget = training_config.get("train_token_budget")
    training_token_budget = (
        reported_token_budget
        if reported_token_budget is not None
        else configured_token_budget
    )
    if training_token_budget is None:
        training_token_budget = estimate_preference_train_tokens(rows)
    return {
        "dpo_train_row_count": len(rows),
        "update_steps": (
            int(observed_steps)
            if observed_steps is not None
            else int(configured_update_steps)
            if configured_update_steps is not None
            else None
        ),
        "planned_update_steps": int(configured_update_steps) if configured_update_steps is not None else None,
        "training_token_budget": int(training_token_budget),
        "mean_train_loss": backend_summary.get("mean_train_loss"),
        "mean_policy_preference_accuracy": backend_summary.get("mean_policy_preference_accuracy"),
        "processed_pair_count": backend_summary.get("processed_pair_count"),
        "processed_input_tokens": backend_summary.get("processed_input_tokens"),
        "unused_train_token_budget": backend_summary.get("unused_train_token_budget"),
        "token_budget_exhausted": backend_summary.get("token_budget_exhausted"),
        "reference_adapter_path": backend_summary.get("reference_adapter_path"),
    }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _runtime_environment() -> dict[str, object]:
    import torch

    return {
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "pythonhashseed": os.environ.get("PYTHONHASHSEED"),
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
    }


if __name__ == "__main__":
    main()
