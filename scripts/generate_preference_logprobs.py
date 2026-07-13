from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mias_dcms.preference_logprob_audit import audit_preference_logprobs
from mias_dcms.preference_logprob_generation import (
    SUPPORTED_PROMPT_FORMATS,
    SUPPORTED_TRUNCATION_STRATEGIES,
    build_preference_logprob_rows,
    load_causal_lm_for_logprobs,
    load_tokenizer_for_logprobs,
)
from mias_dcms.utils import (
    configure_torch_performance,
    get_device,
    parse_torch_dtype,
    read_jsonl,
    resolve_input_path,
    resolve_output_path,
    write_json,
    write_jsonl,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate policy/reference response logprobs for a selector-safe preference active pool."
    )
    parser.add_argument("--input_path", type=Path, required=True)
    parser.add_argument("--output_path", type=Path, required=True)
    parser.add_argument("--summary_path", type=Path)
    parser.add_argument("--policy_model_path", required=True)
    parser.add_argument("--reference_model_path", required=True)
    parser.add_argument("--tokenizer_path")
    parser.add_argument("--policy_adapter_path")
    parser.add_argument("--reference_adapter_path")
    parser.add_argument("--prompt_field", default="prompt")
    parser.add_argument("--response_1_field", default="response_a")
    parser.add_argument("--response_2_field", default="response_b")
    parser.add_argument("--id_field", default="sample_id")
    parser.add_argument("--prompt_format", choices=SUPPORTED_PROMPT_FORMATS, default="chatml_pairwise_v1")
    parser.add_argument("--response_suffix", default="")
    parser.add_argument("--truncation_strategy", choices=SUPPORTED_TRUNCATION_STRATEGIES, default="truncate_prompt_left")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--max_length", type=int, default=4096)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--torch_dtype", choices=["auto", "none", "float16", "bfloat16", "float32"], default="auto")
    parser.add_argument("--tf32", action="store_true", default=True)
    parser.add_argument("--no_tf32", dest="tf32", action="store_false")
    parser.add_argument("--trust_remote_code", action="store_true", default=True)
    parser.add_argument("--no_trust_remote_code", dest="trust_remote_code", action="store_false")
    parser.add_argument("--local_files_only", action="store_true", default=True)
    parser.add_argument("--allow_remote_files", dest="local_files_only", action="store_false")
    parser.add_argument(
        "--allow_zero_implicit_margin",
        action="store_true",
        help="Permit identical policy/reference implicit margins; useful only for smoke checks.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_torch_performance(enable_tf32=bool(args.tf32))

    input_path = resolve_input_path(args.input_path, PROJECT_ROOT)
    output_path = resolve_output_path(args.output_path, PROJECT_ROOT)
    summary_path = (
        resolve_output_path(args.summary_path, PROJECT_ROOT)
        if args.summary_path is not None
        else output_path.with_suffix(".summary.json")
    )
    policy_model_path = _resolve_model_source(
        str(args.policy_model_path),
        local_files_only=bool(args.local_files_only),
    )
    reference_model_path = _resolve_model_source(
        str(args.reference_model_path),
        local_files_only=bool(args.local_files_only),
    )
    tokenizer_path = (
        _resolve_model_source(str(args.tokenizer_path), local_files_only=bool(args.local_files_only))
        if args.tokenizer_path is not None
        else policy_model_path
    )
    policy_adapter_path = (
        str(resolve_input_path(args.policy_adapter_path, PROJECT_ROOT))
        if args.policy_adapter_path is not None
        else None
    )
    reference_adapter_path = (
        str(resolve_input_path(args.reference_adapter_path, PROJECT_ROOT))
        if args.reference_adapter_path is not None
        else None
    )

    device = get_device(str(args.device))
    torch_dtype = parse_torch_dtype(str(args.torch_dtype))
    tokenizer = load_tokenizer_for_logprobs(
        tokenizer_path,
        trust_remote_code=bool(args.trust_remote_code),
        local_files_only=bool(args.local_files_only),
    )
    policy_model = load_causal_lm_for_logprobs(
        policy_model_path,
        device=device,
        torch_dtype=torch_dtype,
        trust_remote_code=bool(args.trust_remote_code),
        local_files_only=bool(args.local_files_only),
        adapter_path=policy_adapter_path,
    )
    reference_model = load_causal_lm_for_logprobs(
        reference_model_path,
        device=device,
        torch_dtype=torch_dtype,
        trust_remote_code=bool(args.trust_remote_code),
        local_files_only=bool(args.local_files_only),
        adapter_path=reference_adapter_path,
    )

    rows, generation_summary = build_preference_logprob_rows(
        read_jsonl(input_path),
        tokenizer=tokenizer,
        policy_model=policy_model,
        reference_model=reference_model,
        device=device,
        batch_size=int(args.batch_size),
        max_length=int(args.max_length),
        prompt_field=str(args.prompt_field),
        response_1_field=str(args.response_1_field),
        response_2_field=str(args.response_2_field),
        id_field=str(args.id_field),
        prompt_format=str(args.prompt_format),
        response_suffix=str(args.response_suffix),
        truncation_strategy=str(args.truncation_strategy),
    )
    audited_rows, audit_summary = audit_preference_logprobs(
        rows,
        id_field=str(args.id_field),
        require_nonzero_implicit_margin=not bool(args.allow_zero_implicit_margin),
    )
    write_jsonl(audited_rows, output_path)

    summary = {
        **generation_summary,
        "audit": audit_summary,
        "input_path": str(input_path),
        "output_path": str(output_path),
        "policy_model_path": policy_model_path,
        "reference_model_path": reference_model_path,
        "tokenizer_path": tokenizer_path,
        "policy_adapter_path": policy_adapter_path,
        "reference_adapter_path": reference_adapter_path,
        "device": str(device),
        "torch_dtype": str(args.torch_dtype),
        "local_files_only": bool(args.local_files_only),
    }
    write_json(summary, summary_path)
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))


def _resolve_model_source(value: str, *, local_files_only: bool) -> str:
    raw = str(value)
    candidate = Path(raw)
    if candidate.is_absolute() or candidate.exists() or (PROJECT_ROOT / candidate).exists():
        return str(resolve_input_path(candidate, PROJECT_ROOT))
    if local_files_only:
        return str(resolve_input_path(candidate, PROJECT_ROOT))
    return raw


if __name__ == "__main__":
    main()
