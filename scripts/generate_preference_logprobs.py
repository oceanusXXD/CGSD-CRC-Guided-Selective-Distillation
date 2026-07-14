from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


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
    resolve_model_reference,
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
    parser.add_argument(
        "--row_batch_size",
        type=int,
        default=64,
        help="Rows per durable checkpoint batch; each row still uses --batch_size sequence scoring.",
    )
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
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from the durable .partial.jsonl checkpoint beside --output_path.",
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

    if int(args.row_batch_size) <= 0:
        raise ValueError("row_batch_size must be positive")
    source_rows = read_jsonl(input_path)
    partial_path = _partial_path(output_path)
    completed_by_id, resume_source = _load_completed_rows(
        source_rows,
        output_path=output_path,
        partial_path=partial_path,
        id_field=str(args.id_field),
        resume=bool(args.resume),
    )
    pending_rows = [
        row for row in source_rows if _row_id(row, id_field=str(args.id_field)) not in completed_by_id
    ]
    batch_summaries: list[dict[str, Any]] = []
    device = get_device(str(args.device))
    if pending_rows:
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
        for start in range(0, len(pending_rows), int(args.row_batch_size)):
            batch = pending_rows[start : start + int(args.row_batch_size)]
            generated_rows, batch_summary = build_preference_logprob_rows(
                batch,
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
            _append_jsonl(partial_path, generated_rows)
            batch_summaries.append(batch_summary)
            print(
                json.dumps(
                    {
                        "event": "logprob_checkpoint",
                        "completed_rows": len(completed_by_id) + start + len(batch),
                        "total_rows": len(source_rows),
                        "partial_path": str(partial_path),
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                ),
                flush=True,
            )

    rows = _ordered_logprob_rows(
        source_rows,
        read_jsonl(partial_path if partial_path.exists() else output_path),
        id_field=str(args.id_field),
    )
    generation_summary = _generation_summary(
        batch_summaries,
        row_count=len(rows),
        args=args,
    )
    audited_rows, audit_summary = audit_preference_logprobs(
        rows,
        id_field=str(args.id_field),
        require_nonzero_implicit_margin=not bool(args.allow_zero_implicit_margin),
    )
    write_jsonl(audited_rows, output_path)
    if partial_path.exists():
        partial_path.unlink()

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
        "row_batch_size": int(args.row_batch_size),
        "resumed": bool(args.resume),
        "resume_source": resume_source,
    }
    write_json(summary, summary_path)
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))


def _resolve_model_source(value: str, *, local_files_only: bool) -> str:
    raw = str(value)
    resolved = resolve_model_reference(raw, PROJECT_ROOT)
    if resolved != raw:
        return resolved
    candidate = Path(raw)
    if candidate.is_absolute() or candidate.exists() or (PROJECT_ROOT / candidate).exists():
        return str(resolve_input_path(candidate, PROJECT_ROOT))
    return raw


def _partial_path(output_path: Path) -> Path:
    return output_path.with_name(f"{output_path.stem}.partial{output_path.suffix}")


def _load_completed_rows(
    source_rows: list[dict[str, Any]],
    *,
    output_path: Path,
    partial_path: Path,
    id_field: str,
    resume: bool,
) -> tuple[dict[str, dict[str, Any]], str]:
    if not resume and partial_path.exists():
        partial_path.unlink()
    checkpoint_path: Path | None = partial_path if partial_path.exists() else None
    resume_source = "partial" if checkpoint_path is not None else "none"
    if resume and checkpoint_path is None and output_path.exists():
        checkpoint_path = output_path
        resume_source = "complete_output"
    completed_rows = read_jsonl(checkpoint_path) if checkpoint_path is not None else []
    source_ids = {_row_id(row, id_field=id_field) for row in source_rows}
    completed_by_id: dict[str, dict[str, Any]] = {}
    for row in completed_rows:
        sample_id = _row_id(row, id_field=id_field)
        if sample_id not in source_ids:
            raise ValueError(f"resume logprob row {sample_id!r} is not in the input pool")
        if sample_id in completed_by_id:
            raise ValueError(f"resume logprob checkpoint contains duplicate sample id {sample_id!r}")
        completed_by_id[sample_id] = dict(row)
    return completed_by_id, resume_source


def _ordered_logprob_rows(
    source_rows: list[dict[str, Any]],
    logprob_rows: list[dict[str, Any]],
    *,
    id_field: str,
) -> list[dict[str, Any]]:
    by_id: dict[str, dict[str, Any]] = {}
    for row in logprob_rows:
        sample_id = _row_id(row, id_field=id_field)
        if sample_id in by_id:
            raise ValueError(f"logprob rows contain duplicate sample id {sample_id!r}")
        by_id[sample_id] = dict(row)
    ordered_ids = [_row_id(row, id_field=id_field) for row in source_rows]
    missing = [sample_id for sample_id in ordered_ids if sample_id not in by_id]
    extra = sorted(set(by_id) - set(ordered_ids))
    if missing or extra:
        message = []
        if missing:
            message.append(f"missing={len(missing)} example={missing[0]!r}")
        if extra:
            message.append(f"extra={len(extra)} example={extra[0]!r}")
        raise ValueError("logprob checkpoint does not cover the input pool: " + "; ".join(message))
    return [by_id[sample_id] for sample_id in ordered_ids]


def _generation_summary(
    batch_summaries: list[dict[str, Any]],
    *,
    row_count: int,
    args: argparse.Namespace,
) -> dict[str, Any]:
    base = dict(batch_summaries[-1]) if batch_summaries else {}
    base.update(
        {
            "row_count": int(row_count),
            "sequence_count": 2 * int(row_count),
            "batch_size": int(args.batch_size),
            "max_length": int(args.max_length),
            "prompt_field": str(args.prompt_field),
            "response_1_field": str(args.response_1_field),
            "response_2_field": str(args.response_2_field),
            "id_field": str(args.id_field),
            "prompt_format": str(args.prompt_format),
            "response_suffix": str(args.response_suffix),
            "truncation_strategy": str(args.truncation_strategy),
            "truncated_sequence_count": sum(
                int(summary.get("truncated_sequence_count", 0)) for summary in batch_summaries
            ),
        }
    )
    return base


def _append_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False))
            handle.write("\n")


def _row_id(row: dict[str, Any], *, id_field: str) -> str:
    value = row.get(id_field, row.get("id"))
    if value is None:
        raise ValueError(f"row is missing id field {id_field!r} and fallback 'id'")
    return str(value)


if __name__ == "__main__":
    main()
