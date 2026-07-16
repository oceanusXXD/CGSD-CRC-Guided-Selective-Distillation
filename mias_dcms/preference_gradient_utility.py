from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
import math
from typing import Any

import torch

from mias_dcms.benchmark_training import dpo_loss
from mias_dcms.preference_logprob_generation import build_preference_logprob_sequences
from mias_dcms.selectors import assert_selector_rows_are_label_safe
from mias_dcms.zero_shot_scoring import sequence_log_likelihoods


def select_gradient_dpo_candidates(
    rows: Iterable[Mapping[str, Any]],
    *,
    budget: int,
    candidate_multiplier: int = 4,
    score_field: str = "gradient_dpo_cheap_score",
    id_field: str = "sample_id",
    coverage_fields: Sequence[str] = ("prompt_cluster", "length_gap_bin"),
) -> list[dict[str, Any]]:
    """Return a top-k candidate set while retaining observable coverage seeds."""
    source_rows = [dict(row) for row in rows]
    assert_selector_rows_are_label_safe(source_rows)
    if budget <= 0:
        raise ValueError("budget must be positive")
    if candidate_multiplier <= 0:
        raise ValueError("candidate_multiplier must be positive")
    candidate_count = min(len(source_rows), int(budget) * int(candidate_multiplier))
    ranked = sorted(
        source_rows,
        key=lambda row: (-float(row[score_field]), _row_id(row, id_field=id_field)),
    )
    coverage_representatives: dict[tuple[str, str], dict[str, Any]] = {}
    for row in ranked:
        for field in coverage_fields:
            value = row.get(field)
            if value is not None:
                coverage_representatives.setdefault((str(field), str(value)), row)
    coverage_seed_ids = {
        _row_id(row, id_field=id_field)
        for row in list(coverage_representatives.values())[:candidate_count]
    }
    selected_rows: list[dict[str, Any]] = []
    for row in ranked:
        if _row_id(row, id_field=id_field) in coverage_seed_ids:
            selected_rows.append(row)
    for row in ranked:
        if len(selected_rows) >= candidate_count:
            break
        if _row_id(row, id_field=id_field) not in coverage_seed_ids:
            selected_rows.append(row)

    selected: list[dict[str, Any]] = []
    for rank, row in enumerate(selected_rows[:candidate_count], start=1):
        payload = dict(row)
        payload["gradient_dpo_stage1_rank"] = rank
        payload["gradient_dpo_stage1_candidate_count"] = candidate_count
        payload["gradient_dpo_candidate_multiplier"] = int(candidate_multiplier)
        payload["gradient_dpo_stage1_coverage_seed"] = int(
            _row_id(row, id_field=id_field) in coverage_seed_ids
        )
        payload["gradient_dpo_stage1_coverage_fields"] = list(coverage_fields)
        selected.append(payload)
    return selected


def add_direct_gradient_utilities(
    rows: Iterable[Mapping[str, Any]],
    *,
    raw_rows_by_id: Mapping[str, Mapping[str, Any]],
    policy_model: Any,
    tokenizer: Any,
    device: torch.device,
    beta: float,
    max_length: int,
    prompt_format: str = "chatml_pairwise_v1",
    id_field: str = "sample_id",
) -> list[dict[str, Any]]:
    """Attach label-safe, direct LoRA-gradient utilities to stage-one rows.

    Pseudo orientation is inferred solely from the selection-time preference
    probability.  Oracle preference labels are intentionally neither accepted
    nor read here.
    """
    candidate_rows = [dict(row) for row in rows]
    assert_selector_rows_are_label_safe(candidate_rows)
    if beta <= 0.0:
        raise ValueError("beta must be positive")
    trainable = [parameter for parameter in policy_model.parameters() if parameter.requires_grad]
    if not trainable:
        raise ValueError("policy_model has no trainable parameters for gradient scoring")

    policy_model.eval()
    output_rows: list[dict[str, Any]] = []
    for row in candidate_rows:
        sample_id = _row_id(row, id_field=id_field)
        raw_row = raw_rows_by_id.get(sample_id)
        if raw_row is None:
            raise ValueError(f"raw preference row is missing sample id {sample_id!r}")
        _assert_raw_row_is_safe(raw_row, sample_id=sample_id)
        probability_response_1 = _probability_response_1(row, sample_id=sample_id)
        prefer_response_1 = probability_response_1 >= 0.5
        record = _pseudo_oriented_record(raw_row, prefer_response_1=prefer_response_1)
        sequences = build_preference_logprob_sequences(
            [record],
            tokenizer=tokenizer,
            max_length=max_length,
            prompt_field="prompt",
            response_1_field="response_a",
            response_2_field="response_b",
            id_field="sample_id",
            prompt_format=prompt_format,
        )
        input_ids, attention_mask, prompt_lengths = _single_pair_tensors(
            sequences,
            pad_token_id=_pad_token_id(tokenizer),
            device=device,
        )
        policy_model.zero_grad(set_to_none=True)
        output = policy_model(input_ids=input_ids, attention_mask=attention_mask)
        scores = sequence_log_likelihoods(
            logits=output.logits,
            input_ids=input_ids,
            prompt_lengths=prompt_lengths,
            attention_mask=attention_mask,
        )
        # The active policy is the shared initialization at selection time.
        # Its detached scores are therefore the frozen DPO reference used by
        # the subsequent training run.  This avoids silently mixing a base
        # model reference from an older log-prob artifact with an adapter
        # reference during training.
        reference_chosen = scores[:1].detach()
        reference_rejected = scores[1:].detach()
        loss = dpo_loss(
            policy_chosen=scores[:1],
            policy_rejected=scores[1:],
            reference_chosen=reference_chosen,
            reference_rejected=reference_rejected,
            beta=beta,
        )
        gradients = torch.autograd.grad(loss, trainable, allow_unused=True)
        gradient_norm = math.sqrt(
            sum(float(gradient.detach().float().pow(2).sum().item()) for gradient in gradients if gradient is not None)
        )
        pair_token_count = int(attention_mask.sum().detach().item())
        length_normalized_norm = gradient_norm / math.sqrt(max(1, pair_token_count))
        cheap_score = float(row["gradient_dpo_cheap_score"])
        utility = gradient_dpo_utility(
            cheap_score=cheap_score,
            gradient_norm=gradient_norm,
            pair_token_count=pair_token_count,
        )
        payload = dict(row)
        payload.update(
            {
                "gradient_dpo_pseudo_preferred_response": 1 if prefer_response_1 else 2,
                "gradient_dpo_reference": "frozen_selection_policy",
                "gradient_dpo_direct_loss": float(loss.detach().float().item()),
                "gradient_dpo_direct_gradient_norm": gradient_norm,
                "gradient_dpo_direct_length_normalized_gradient_norm": length_normalized_norm,
                "gradient_dpo_direct_pair_token_count": pair_token_count,
                "gradient_dpo_score": utility,
                "gradient_dpo_utility": utility,
            }
        )
        output_rows.append(payload)
    return output_rows


def gradient_dpo_utility(
    *,
    cheap_score: float,
    gradient_norm: float,
    pair_token_count: int,
) -> float:
    """Pure utility calculation used by the model scorer and tests."""
    if cheap_score < 0.0:
        raise ValueError("cheap_score must be non-negative")
    if gradient_norm < 0.0:
        raise ValueError("gradient_norm must be non-negative")
    if pair_token_count <= 0:
        raise ValueError("pair_token_count must be positive")
    return float(cheap_score) * float(gradient_norm) / math.sqrt(int(pair_token_count))


def _pseudo_oriented_record(row: Mapping[str, Any], *, prefer_response_1: bool) -> dict[str, str]:
    return {
        "sample_id": str(row.get("sample_id", row.get("id"))),
        "prompt": str(row["prompt"]),
        "response_a": str(row["response_a"] if prefer_response_1 else row["response_b"]),
        "response_b": str(row["response_b"] if prefer_response_1 else row["response_a"]),
    }


def _single_pair_tensors(
    sequences: Sequence[Any],
    *,
    pad_token_id: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if len(sequences) != 2:
        raise ValueError("expected exactly two sequences for a preference pair")
    maximum_length = max(len(sequence.input_ids) for sequence in sequences)
    input_ids = torch.tensor(
        [list(sequence.input_ids) + [pad_token_id] * (maximum_length - len(sequence.input_ids)) for sequence in sequences],
        dtype=torch.long,
        device=device,
    )
    attention_mask = torch.tensor(
        [[1] * len(sequence.input_ids) + [0] * (maximum_length - len(sequence.input_ids)) for sequence in sequences],
        dtype=torch.long,
        device=device,
    )
    prompt_lengths = torch.tensor(
        [int(sequence.prompt_token_count) for sequence in sequences],
        dtype=torch.long,
        device=device,
    )
    return input_ids, attention_mask, prompt_lengths


def _assert_raw_row_is_safe(row: Mapping[str, Any], *, sample_id: str) -> None:
    try:
        assert_selector_rows_are_label_safe([dict(row)])
    except ValueError as exc:
        raise ValueError(f"raw row {sample_id!r} is not selector-safe") from exc


def _probability_response_1(row: Mapping[str, Any], *, sample_id: str) -> float:
    probability = float(row.get("probability_response_1"))
    if not 0.0 <= probability <= 1.0:
        raise ValueError(f"row {sample_id!r} has invalid probability_response_1")
    return probability


def _pad_token_id(tokenizer: Any) -> int:
    value = getattr(tokenizer, "pad_token_id", None)
    if value is None:
        value = getattr(tokenizer, "eos_token_id", None)
    if value is None:
        raise ValueError("tokenizer must provide pad_token_id or eos_token_id")
    return int(value)


def _row_id(row: Mapping[str, Any], *, id_field: str) -> str:
    value = row.get(id_field, row.get("id"))
    if value is None:
        raise ValueError(f"row is missing id field {id_field!r}")
    return str(value)
