from __future__ import annotations

from collections.abc import Sequence
import math
from pathlib import Path
from typing import Any

import torch


def label_codes(count: int) -> list[str]:
    if not 2 <= count <= 26:
        raise ValueError("label count must be between 2 and 26")
    return [chr(ord("A") + index) for index in range(count)]


def build_classification_messages(text: str, label_names: Sequence[str]) -> list[dict[str, str]]:
    codes = label_codes(len(label_names))
    mapping = "\n".join(f"{code}: {name}" for code, name in zip(codes, label_names, strict=True))
    return [
        {
            "role": "system",
            "content": "You are a precise text classifier. Do not explain your answer.",
        },
        {
            "role": "user",
            "content": (
                f"Text:\n{text}\n\n"
                f"Categories:\n{mapping}\n\n"
                f"Classify the text and answer with exactly one letter from {codes[0]} to {codes[-1]}."
            ),
        },
    ]


def build_pairwise_messages(prompt: str, first_response: str, second_response: str) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": "You compare two assistant responses for overall human preference. Do not explain your answer.",
        },
        {
            "role": "user",
            "content": (
                f"User prompt:\n{prompt}\n\n"
                f"Response A:\n{first_response}\n\n"
                f"Response B:\n{second_response}\n\n"
                "Choose the better response. Answer with exactly one letter: A or B."
            ),
        },
    ]


def softmax_probabilities(log_scores: Sequence[float]) -> list[float]:
    if not log_scores:
        raise ValueError("log_scores cannot be empty")
    maximum = max(float(score) for score in log_scores)
    exponentials = [math.exp(float(score) - maximum) for score in log_scores]
    denominator = sum(exponentials)
    return [value / denominator for value in exponentials]


def sequence_log_likelihoods(
    *,
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    prompt_lengths: torch.Tensor,
    attention_mask: torch.Tensor,
    token_chunk_size: int = 128,
) -> torch.Tensor:
    if logits.ndim != 3 or input_ids.ndim != 2:
        raise ValueError("expected logits [batch, sequence, vocab] and input_ids [batch, sequence]")
    if token_chunk_size <= 0:
        raise ValueError("token_chunk_size must be positive")
    if logits.shape[:2] != input_ids.shape:
        raise ValueError("logits and input_ids must agree on batch and sequence dimensions")
    target_ids = input_ids[:, 1:]
    token_positions = torch.arange(1, input_ids.shape[1], device=input_ids.device).unsqueeze(0)
    candidate_mask = token_positions >= prompt_lengths.to(input_ids.device).unsqueeze(1)
    candidate_mask &= attention_mask[:, 1:].bool()
    if torch.any(candidate_mask.sum(dim=1) == 0):
        raise ValueError("each sequence must contain at least one candidate token")
    totals = torch.zeros(logits.shape[0], dtype=logits.dtype, device=logits.device)
    shifted_logits = logits[:, :-1, :]
    for start in range(0, shifted_logits.shape[1], int(token_chunk_size)):
        end = min(start + int(token_chunk_size), shifted_logits.shape[1])
        log_probs = shifted_logits[:, start:end, :].log_softmax(dim=-1)
        selected_log_probs = log_probs.gather(
            -1,
            target_ids[:, start:end].unsqueeze(-1),
        ).squeeze(-1)
        totals = totals + (selected_log_probs * candidate_mask[:, start:end]).sum(dim=1)
    return totals


class CausalCandidateScorer:
    def __init__(
        self,
        *,
        model: Any,
        tokenizer: Any,
        batch_size: int = 16,
        max_length: int = 4096,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if max_length <= 1:
            raise ValueError("max_length must be greater than one")
        self.model = model
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.max_length = max_length
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str,
        *,
        batch_size: int = 16,
        max_length: int = 4096,
        device_map: str | None = "auto",
        torch_dtype: str = "auto",
        trust_remote_code: bool = False,
    ) -> "CausalCandidateScorer":
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if is_local_peft_adapter(model_name_or_path):
            from peft import PeftConfig, PeftModel

            adapter_config = PeftConfig.from_pretrained(model_name_or_path)
            tokenizer = AutoTokenizer.from_pretrained(
                model_name_or_path, trust_remote_code=trust_remote_code
            )
            base_model = AutoModelForCausalLM.from_pretrained(
                adapter_config.base_model_name_or_path,
                device_map=device_map,
                dtype=torch_dtype,
                trust_remote_code=trust_remote_code,
            )
            model = PeftModel.from_pretrained(base_model, model_name_or_path)
        else:
            tokenizer = AutoTokenizer.from_pretrained(
                model_name_or_path, trust_remote_code=trust_remote_code
            )
            model = AutoModelForCausalLM.from_pretrained(
                model_name_or_path,
                device_map=device_map,
                dtype=torch_dtype,
                trust_remote_code=trust_remote_code,
            )
        model.eval()
        return cls(model=model, tokenizer=tokenizer, batch_size=batch_size, max_length=max_length)

    def score_messages(
        self,
        messages: Sequence[Sequence[dict[str, str]]],
        candidates: Sequence[str],
    ) -> list[list[float]]:
        probabilities, _ = self._score_messages_impl(
            messages,
            candidates,
            collect_representations=False,
        )
        return probabilities

    def score_messages_with_representations(
        self,
        messages: Sequence[Sequence[dict[str, str]]],
        candidates: Sequence[str],
    ) -> tuple[list[list[float]], list[list[float]]]:
        """Score candidates and return one frozen prompt representation per message."""
        return self._score_messages_impl(
            messages,
            candidates,
            collect_representations=True,
        )

    def _score_messages_impl(
        self,
        messages: Sequence[Sequence[dict[str, str]]],
        candidates: Sequence[str],
        *,
        collect_representations: bool,
    ) -> tuple[list[list[float]], list[list[float]]]:
        if not candidates:
            raise ValueError("candidates cannot be empty")
        encoded: list[tuple[int, list[int], int]] = []
        for message_index, message in enumerate(messages):
            prompt = self._render_messages(message)
            prompt_ids = self.tokenizer(prompt, add_special_tokens=False)["input_ids"]
            for candidate in candidates:
                candidate_ids = self.tokenizer(str(candidate), add_special_tokens=False)["input_ids"]
                if not candidate_ids:
                    raise ValueError(f"candidate tokenized to an empty sequence: {candidate!r}")
                available_prompt_length = self.max_length - len(candidate_ids)
                if available_prompt_length <= 0:
                    raise ValueError("candidate exceeds max_length")
                truncated_prompt_ids = prompt_ids[-available_prompt_length:]
                encoded.append(
                    (
                        message_index,
                        truncated_prompt_ids + candidate_ids,
                        len(truncated_prompt_ids),
                    )
                )
        grouped_scores = [[0.0 for _ in candidates] for _ in messages]
        grouped_representations: list[list[float] | None] = [None for _ in messages]
        candidate_count = len(candidates)
        for start in range(0, len(encoded), self.batch_size):
            batch = encoded[start : start + self.batch_size]
            scores, representations = self._score_encoded_batch(
                batch,
                collect_representation=collect_representations,
            )
            for offset, score in enumerate(scores):
                flat_index = start + offset
                message_index = flat_index // candidate_count
                candidate_index = flat_index % candidate_count
                grouped_scores[message_index][candidate_index] = float(score)
                if collect_representations and candidate_index == 0:
                    grouped_representations[message_index] = representations[offset]
        if collect_representations and any(value is None for value in grouped_representations):
            raise RuntimeError("representation collection returned an incomplete batch")
        return (
            [softmax_probabilities(scores) for scores in grouped_scores],
            [list(value or []) for value in grouped_representations],
        )

    def _render_messages(self, messages: Sequence[dict[str, str]]) -> str:
        try:
            return self.tokenizer.apply_chat_template(
                list(messages),
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            return self.tokenizer.apply_chat_template(
                list(messages),
                tokenize=False,
                add_generation_prompt=True,
            )

    def _score_encoded_batch(
        self,
        batch: Sequence[tuple[int, list[int], int]],
        *,
        collect_representation: bool = False,
    ) -> tuple[list[float], list[list[float]]]:
        device = _model_input_device(self.model)
        maximum_length = max(len(input_ids) for _, input_ids, _ in batch)
        pad_token_id = int(self.tokenizer.pad_token_id)
        padded_ids: list[list[int]] = []
        attention_masks: list[list[int]] = []
        prompt_lengths: list[int] = []
        for _, input_ids, prompt_length in batch:
            padding = maximum_length - len(input_ids)
            padded_ids.append(input_ids + [pad_token_id] * padding)
            attention_masks.append([1] * len(input_ids) + [0] * padding)
            prompt_lengths.append(prompt_length)
        input_tensor = torch.tensor(padded_ids, dtype=torch.long, device=device)
        attention_tensor = torch.tensor(attention_masks, dtype=torch.long, device=device)
        prompt_tensor = torch.tensor(prompt_lengths, dtype=torch.long, device=device)
        with torch.inference_mode():
            output = self.model(
                input_ids=input_tensor,
                attention_mask=attention_tensor,
                output_hidden_states=collect_representation,
                return_dict=True,
            )
        scores = sequence_log_likelihoods(
            logits=output.logits,
            input_ids=input_tensor,
            prompt_lengths=prompt_tensor,
            attention_mask=attention_tensor,
        )
        score_values = scores.detach().float().cpu().tolist()
        if not collect_representation:
            return score_values, [[] for _ in batch]
        hidden_states = getattr(output, "hidden_states", None)
        if not hidden_states:
            raise RuntimeError("model did not return hidden states for representation collection")
        final_hidden = hidden_states[-1].detach().float()
        representations = [
            final_hidden[index, prompt_length - 1].cpu().tolist()
            for index, (_, _, prompt_length) in enumerate(batch)
        ]
        return score_values, representations


def _model_input_device(model: Any) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def is_local_peft_adapter(model_name_or_path: str | Path) -> bool:
    path = Path(model_name_or_path)
    return path.is_dir() and (path / "adapter_config.json").is_file()
