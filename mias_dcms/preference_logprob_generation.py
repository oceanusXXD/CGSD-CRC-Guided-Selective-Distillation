from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
import math
from typing import Any


DEFAULT_POLICY_RESPONSE_1_FIELD = "policy_logprob_response_1"
DEFAULT_POLICY_RESPONSE_2_FIELD = "policy_logprob_response_2"
DEFAULT_REFERENCE_RESPONSE_1_FIELD = "reference_logprob_response_1"
DEFAULT_REFERENCE_RESPONSE_2_FIELD = "reference_logprob_response_2"

SUPPORTED_PROMPT_FORMATS = (
    "plain",
    "chatml_pairwise_v1",
    "chatml_pairwise_thinking_disabled_v1",
)
SUPPORTED_TRUNCATION_STRATEGIES = ("error", "truncate_prompt_left")

EMPTY_THINKING_BLOCK = "\n\n"


@dataclass(frozen=True)
class PreferenceLogprobSequence:
    sample_id: str
    response_key: str
    input_ids: list[int]
    labels: list[int]
    prompt_token_count: int
    response_token_count: int
    was_truncated: bool = False


def build_preference_logprob_rows(
    rows: Iterable[Mapping[str, Any]],
    *,
    tokenizer: Any,
    policy_model: Any,
    reference_model: Any | None,
    reference_rows_by_id: Mapping[str, Mapping[str, Any]] | None = None,
    device: Any,
    batch_size: int = 4,
    max_length: int = 4096,
    prompt_field: str = "prompt",
    response_1_field: str = "response_a",
    response_2_field: str = "response_b",
    id_field: str = "sample_id",
    prompt_format: str = "chatml_pairwise_v1",
    response_suffix: str = "",
    truncation_strategy: str = "truncate_prompt_left",
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    source_rows = [dict(row) for row in rows]
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if reference_model is None and reference_rows_by_id is None:
        raise ValueError("reference_model or reference_rows_by_id is required")

    sequences = build_preference_logprob_sequences(
        source_rows,
        tokenizer=tokenizer,
        max_length=max_length,
        prompt_field=prompt_field,
        response_1_field=response_1_field,
        response_2_field=response_2_field,
        id_field=id_field,
        prompt_format=prompt_format,
        response_suffix=response_suffix,
        truncation_strategy=truncation_strategy,
    )
    policy_scores = score_preference_logprob_sequences(
        policy_model,
        sequences,
        pad_token_id=_pad_token_id(tokenizer),
        device=device,
        batch_size=batch_size,
    )
    if reference_rows_by_id is None:
        reference_scores = score_preference_logprob_sequences(
            reference_model,
            sequences,
            pad_token_id=_pad_token_id(tokenizer),
            device=device,
            batch_size=batch_size,
        )
    else:
        reference_scores = {
            (sequence.sample_id, sequence.response_key): _cached_reference_score(
                reference_rows_by_id,
                sample_id=sequence.sample_id,
                response_key=sequence.response_key,
            )
            for sequence in sequences
        }

    metadata_by_key = {
        (sequence.sample_id, sequence.response_key): sequence for sequence in sequences
    }
    output_rows: list[dict[str, Any]] = []
    for row in source_rows:
        sample_id = _row_id(row, id_field=id_field)
        sequence_1 = metadata_by_key[(sample_id, "response_1")]
        sequence_2 = metadata_by_key[(sample_id, "response_2")]
        policy_1 = policy_scores[(sample_id, "response_1")]
        policy_2 = policy_scores[(sample_id, "response_2")]
        reference_1 = reference_scores[(sample_id, "response_1")]
        reference_2 = reference_scores[(sample_id, "response_2")]
        policy_gap = policy_1 - policy_2
        reference_gap = reference_1 - reference_2
        implicit_reward_gap = policy_gap - reference_gap
        probability_response_1 = _sigmoid(policy_gap)
        output_rows.append(
            {
                id_field: sample_id,
                "id": str(row.get("id", sample_id)),
                "probability_response_1": probability_response_1,
                "probability_response_2": 1.0 - probability_response_1,
                DEFAULT_POLICY_RESPONSE_1_FIELD: policy_1,
                DEFAULT_POLICY_RESPONSE_2_FIELD: policy_2,
                DEFAULT_REFERENCE_RESPONSE_1_FIELD: reference_1,
                DEFAULT_REFERENCE_RESPONSE_2_FIELD: reference_2,
                "policy_logprob_gap": policy_gap,
                "reference_logprob_gap": reference_gap,
                "implicit_reward_gap": implicit_reward_gap,
                "absolute_implicit_margin": abs(implicit_reward_gap),
                "prompt_token_count_response_1": sequence_1.prompt_token_count,
                "prompt_token_count_response_2": sequence_2.prompt_token_count,
                "response_1_token_count": sequence_1.response_token_count,
                "response_2_token_count": sequence_2.response_token_count,
                "response_1_prompt_truncated": sequence_1.was_truncated,
                "response_2_prompt_truncated": sequence_2.was_truncated,
            }
        )

    summary = {
        "row_count": len(output_rows),
        "sequence_count": len(sequences),
        "batch_size": int(batch_size),
        "max_length": int(max_length),
        "prompt_field": str(prompt_field),
        "response_1_field": str(response_1_field),
        "response_2_field": str(response_2_field),
        "id_field": str(id_field),
        "prompt_format": str(prompt_format),
        "response_suffix": str(response_suffix),
        "truncation_strategy": str(truncation_strategy),
        "truncated_sequence_count": sum(1 for sequence in sequences if sequence.was_truncated),
        "logprob_fields": {
            "policy_response_1": DEFAULT_POLICY_RESPONSE_1_FIELD,
            "policy_response_2": DEFAULT_POLICY_RESPONSE_2_FIELD,
            "reference_response_1": DEFAULT_REFERENCE_RESPONSE_1_FIELD,
            "reference_response_2": DEFAULT_REFERENCE_RESPONSE_2_FIELD,
        },
    }
    return output_rows, summary


def build_preference_logprob_sequences(
    rows: Sequence[Mapping[str, Any]],
    *,
    tokenizer: Any,
    max_length: int,
    prompt_field: str = "prompt",
    response_1_field: str = "response_a",
    response_2_field: str = "response_b",
    id_field: str = "sample_id",
    prompt_format: str = "chatml_pairwise_v1",
    response_suffix: str = "",
    truncation_strategy: str = "truncate_prompt_left",
) -> list[PreferenceLogprobSequence]:
    if max_length <= 1:
        raise ValueError("max_length must be greater than one")
    if prompt_format not in SUPPORTED_PROMPT_FORMATS:
        raise ValueError(f"unsupported prompt_format: {prompt_format!r}")
    if truncation_strategy not in SUPPORTED_TRUNCATION_STRATEGIES:
        raise ValueError(f"unsupported truncation_strategy: {truncation_strategy!r}")

    sequences: list[PreferenceLogprobSequence] = []
    for row in rows:
        sample_id = _row_id(row, id_field=id_field)
        prompt_text = format_preference_prompt(str(row[prompt_field]), prompt_format=prompt_format)
        for response_key, response_field in (
            ("response_1", response_1_field),
            ("response_2", response_2_field),
        ):
            if response_field not in row:
                raise ValueError(f"row {sample_id!r} is missing response field {response_field!r}")
            response_text = f"{row[response_field]}{response_suffix}"
            sequences.append(
                tokenize_preference_logprob_sequence(
                    sample_id=sample_id,
                    response_key=response_key,
                    prompt_text=prompt_text,
                    response_text=response_text,
                    tokenizer=tokenizer,
                    max_length=max_length,
                    truncation_strategy=truncation_strategy,
                )
            )
    return sequences


def tokenize_preference_logprob_sequence(
    *,
    sample_id: str,
    response_key: str,
    prompt_text: str,
    response_text: str,
    tokenizer: Any,
    max_length: int,
    truncation_strategy: str,
) -> PreferenceLogprobSequence:
    prompt_ids = _encode(tokenizer, prompt_text)
    response_ids = _encode(tokenizer, response_text)
    if not response_ids:
        raise ValueError(f"row {sample_id!r} {response_key} has zero response tokens")

    was_truncated = False
    total_length = len(prompt_ids) + len(response_ids)
    if total_length > max_length:
        if truncation_strategy == "error":
            raise ValueError(
                f"row {sample_id!r} {response_key} has {total_length} tokens, "
                f"which exceeds max_length={max_length}"
            )
        prompt_budget = int(max_length) - len(response_ids)
        if prompt_budget <= 0:
            raise ValueError(
                f"row {sample_id!r} {response_key} response has {len(response_ids)} tokens, "
                f"which leaves no prompt budget under max_length={max_length}"
            )
        prompt_ids = prompt_ids[-prompt_budget:]
        was_truncated = True

    input_ids = [*prompt_ids, *response_ids]
    labels = [-100] * len(prompt_ids) + list(response_ids)
    return PreferenceLogprobSequence(
        sample_id=sample_id,
        response_key=response_key,
        input_ids=input_ids,
        labels=labels,
        prompt_token_count=len(prompt_ids),
        response_token_count=len(response_ids),
        was_truncated=was_truncated,
    )


def score_preference_logprob_sequences(
    model: Any,
    sequences: Sequence[PreferenceLogprobSequence],
    *,
    pad_token_id: int,
    device: Any,
    batch_size: int,
) -> dict[tuple[str, str], float]:
    import torch

    model.eval()
    scores: dict[tuple[str, str], float] = {}
    with torch.no_grad():
        for start in range(0, len(sequences), batch_size):
            batch = list(sequences[start : start + batch_size])
            input_ids, attention_mask, labels = _collate_logprob_batch(
                batch,
                pad_token_id=pad_token_id,
                device=device,
            )
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
            batch_scores = _sequence_logprob_sums(outputs.logits, labels)
            for sequence, score in zip(batch, batch_scores, strict=True):
                scores[(sequence.sample_id, sequence.response_key)] = float(score)
    return scores


def format_preference_prompt(prompt: str, *, prompt_format: str) -> str:
    if prompt_format == "plain":
        return str(prompt)
    if prompt_format == "chatml_pairwise_v1":
        return f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
    if prompt_format == "chatml_pairwise_thinking_disabled_v1":
        return (
            f"<|im_start|>user\n{prompt}<|im_end|>\n"
            f"<|im_start|>assistant\n{EMPTY_THINKING_BLOCK}"
        )
    raise ValueError(f"unsupported prompt_format: {prompt_format!r}")


def load_causal_lm_for_logprobs(
    model_path: str,
    *,
    device: Any,
    torch_dtype: Any = "auto",
    trust_remote_code: bool = True,
    local_files_only: bool = True,
    adapter_path: str | None = None,
) -> Any:
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch_dtype,
        trust_remote_code=trust_remote_code,
        local_files_only=local_files_only,
    )
    if adapter_path is not None:
        try:
            from peft import PeftModel
        except ImportError as exc:
            raise ImportError("adapter_path requires the peft package") from exc
        model = PeftModel.from_pretrained(model, adapter_path, is_trainable=False)
    model.to(device)
    return model


def load_tokenizer_for_logprobs(
    tokenizer_path: str,
    *,
    trust_remote_code: bool = True,
    local_files_only: bool = True,
) -> Any:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path,
        trust_remote_code=trust_remote_code,
        local_files_only=local_files_only,
    )
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError("tokenizer has neither pad_token_id nor eos_token_id")
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def _sequence_logprob_sums(logits: Any, labels: Any) -> list[float]:
    import torch

    shift_logits = logits[:, :-1, :].float()
    shift_labels = labels[:, 1:]
    active = shift_labels.ne(-100)
    safe_labels = torch.where(active, shift_labels, torch.zeros_like(shift_labels))
    token_logprobs = torch.log_softmax(shift_logits, dim=-1).gather(
        dim=-1,
        index=safe_labels.unsqueeze(-1),
    ).squeeze(-1)
    token_logprobs = torch.where(active, token_logprobs, torch.zeros_like(token_logprobs))
    return token_logprobs.sum(dim=1).detach().cpu().tolist()


def _collate_logprob_batch(
    batch: Sequence[PreferenceLogprobSequence],
    *,
    pad_token_id: int,
    device: Any,
) -> tuple[Any, Any, Any]:
    import torch

    max_len = max(len(sequence.input_ids) for sequence in batch)
    input_rows: list[list[int]] = []
    label_rows: list[list[int]] = []
    mask_rows: list[list[int]] = []
    for sequence in batch:
        pad_count = max_len - len(sequence.input_ids)
        input_rows.append(sequence.input_ids + [pad_token_id] * pad_count)
        label_rows.append(sequence.labels + [-100] * pad_count)
        mask_rows.append([1] * len(sequence.input_ids) + [0] * pad_count)
    return (
        torch.tensor(input_rows, dtype=torch.long, device=device),
        torch.tensor(mask_rows, dtype=torch.long, device=device),
        torch.tensor(label_rows, dtype=torch.long, device=device),
    )


def _encode(tokenizer: Any, text: str) -> list[int]:
    encoded = tokenizer(str(text), add_special_tokens=False)
    return [int(token_id) for token_id in encoded["input_ids"]]


def _pad_token_id(tokenizer: Any) -> int:
    value = getattr(tokenizer, "pad_token_id", None)
    if value is None:
        value = getattr(tokenizer, "eos_token_id", None)
    if value is None:
        raise ValueError("tokenizer has no pad_token_id or eos_token_id")
    return int(value)


def _row_id(row: Mapping[str, Any], *, id_field: str) -> str:
    value = row.get(id_field, row.get("id"))
    if value is None:
        raise ValueError(f"row is missing id field {id_field!r} and fallback 'id'")
    return str(value)


def _cached_reference_score(
    rows_by_id: Mapping[str, Mapping[str, Any]],
    *,
    sample_id: str,
    response_key: str,
) -> float:
    row = rows_by_id.get(str(sample_id))
    if row is None:
        raise ValueError(f"reference cache is missing sample id {sample_id!r}")
    field = (
        DEFAULT_REFERENCE_RESPONSE_1_FIELD
        if response_key == "response_1"
        else DEFAULT_REFERENCE_RESPONSE_2_FIELD
    )
    if field not in row:
        raise ValueError(f"reference cache row {sample_id!r} is missing {field!r}")
    return float(row[field])


def _sigmoid(value: float) -> float:
    if value >= 0.0:
        z = math.exp(-float(value))
        return 1.0 / (1.0 + z)
    z = math.exp(float(value))
    return z / (1.0 + z)
