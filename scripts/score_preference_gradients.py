from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mias_dcms.preference_dcms_inputs import build_preference_dcms_candidate_rows
from mias_dcms.preference_gradient_utility import (
    add_direct_gradient_utilities,
    select_gradient_dpo_candidates,
)
from mias_dcms.preference_logprob_generation import load_tokenizer_for_logprobs
from mias_dcms.utils import get_device, read_jsonl, resolve_model_reference, write_json, write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute direct LoRA-gradient GradientDPO utilities after a top-k label-safe filter."
    )
    parser.add_argument("--input_path", type=Path, required=True)
    parser.add_argument("--output_path", type=Path, required=True)
    parser.add_argument("--target_moments_path", type=Path, required=True)
    parser.add_argument("--model_name_or_path", required=True)
    parser.add_argument("--policy_adapter_path", type=Path, required=True)
    parser.add_argument("--budget", type=int, required=True)
    parser.add_argument("--candidate_multiplier", type=int, default=4)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--prompt_format", default="chatml_pairwise_v1")
    parser.add_argument("--torch_dtype", default="auto")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--gradient_checkpointing", action="store_true", default=True)
    parser.add_argument("--no_gradient_checkpointing", dest="gradient_checkpointing", action="store_false")
    parser.add_argument(
        "--group_fields",
        default="prompt_cluster,length_gap_bin",
        help="Observable fields used to calculate full-pool DCMS target moments.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = read_jsonl(args.input_path)
    group_fields = tuple(field.strip() for field in str(args.group_fields).split(",") if field.strip())
    candidates = select_gradient_dpo_candidates(
        rows,
        budget=int(args.budget),
        candidate_multiplier=int(args.candidate_multiplier),
        coverage_fields=group_fields,
    )
    model_name_or_path = resolve_model_reference(str(args.model_name_or_path), PROJECT_ROOT)
    device = get_device(str(args.device))
    tokenizer, policy = _load_trainable_policy(
        model_name_or_path=model_name_or_path,
        adapter_path=args.policy_adapter_path,
        device=device,
        torch_dtype=_torch_dtype(str(args.torch_dtype)),
        gradient_checkpointing=bool(args.gradient_checkpointing),
    )
    raw_rows_by_id = {str(row.get("sample_id", row.get("id"))): row for row in rows}
    scored = add_direct_gradient_utilities(
        candidates,
        raw_rows_by_id=raw_rows_by_id,
        policy_model=policy,
        tokenizer=tokenizer,
        device=device,
        beta=float(args.beta),
        max_length=int(args.max_length),
        prompt_format=str(args.prompt_format),
    )
    full_pool_candidates = build_preference_dcms_candidate_rows(
        rows,
        method="gradient_dpo",
        group_fields=group_fields,
    )
    target_moments = _pool_target_moments(full_pool_candidates)
    write_jsonl(scored, args.output_path)
    write_json(target_moments, args.target_moments_path)
    summary = {
        "input_path": str(args.input_path),
        "output_path": str(args.output_path),
        "target_moments_path": str(args.target_moments_path),
        "pool_size": len(rows),
        "stage1_candidate_count": len(candidates),
        "candidate_multiplier": int(args.candidate_multiplier),
        "gradient_scored_count": len(scored),
        "group_fields": list(group_fields),
        "target_moment_count": len(target_moments),
    }
    write_json(summary, args.output_path.with_suffix(".summary.json"))
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))


def _load_trainable_policy(
    *,
    model_name_or_path: str,
    adapter_path: Path,
    device: torch.device,
    torch_dtype: Any,
    gradient_checkpointing: bool,
) -> tuple[Any, Any]:
    from peft import PeftModel
    from transformers import AutoModelForCausalLM

    tokenizer = load_tokenizer_for_logprobs(model_name_or_path, local_files_only=True)
    base = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=torch_dtype,
        local_files_only=True,
    )
    policy = PeftModel.from_pretrained(base, str(adapter_path), is_trainable=True)
    policy.config.use_cache = False
    if gradient_checkpointing:
        policy.gradient_checkpointing_enable()
        if hasattr(policy, "enable_input_require_grads"):
            policy.enable_input_require_grads()
    policy.to(device)
    return tokenizer, policy


def _pool_target_moments(rows: list[dict[str, Any]]) -> dict[str, float]:
    if not rows:
        raise ValueError("cannot compute target moments from an empty pool")
    groups = sorted({group for row in rows for group in row["groups"]})
    return {
        group: sum(float(row["groups"].get(group, 0.0)) for row in rows) / len(rows)
        for group in groups
    }


def _torch_dtype(value: str) -> Any:
    normalized = value.strip().lower()
    if normalized in {"auto", ""}:
        return "auto"
    aliases = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
    if normalized not in aliases:
        raise ValueError(f"unsupported torch dtype: {value!r}")
    return aliases[normalized]


if __name__ == "__main__":
    main()
