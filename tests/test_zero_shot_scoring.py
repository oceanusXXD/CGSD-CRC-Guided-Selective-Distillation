from __future__ import annotations

from types import SimpleNamespace
import unittest

import torch

from mias_dcms.zero_shot_scoring import CausalCandidateScorer, sequence_log_likelihoods, softmax_probabilities


class SequenceLogLikelihoodTest(unittest.TestCase):
    def test_chunked_log_likelihood_matches_full_log_softmax(self) -> None:
        torch.manual_seed(7)
        logits = torch.randn(2, 7, 13)
        input_ids = torch.tensor(
            [
                [1, 2, 3, 4, 5, 0, 0],
                [2, 3, 4, 5, 6, 7, 8],
            ]
        )
        prompt_lengths = torch.tensor([3, 4])
        attention_mask = torch.tensor(
            [
                [1, 1, 1, 1, 1, 0, 0],
                [1, 1, 1, 1, 1, 1, 1],
            ]
        )

        actual = sequence_log_likelihoods(
            logits=logits,
            input_ids=input_ids,
            prompt_lengths=prompt_lengths,
            attention_mask=attention_mask,
            token_chunk_size=2,
        )
        full_log_probs = logits[:, :-1, :].log_softmax(dim=-1)
        target_ids = input_ids[:, 1:]
        token_log_probs = full_log_probs.gather(-1, target_ids.unsqueeze(-1)).squeeze(-1)
        positions = torch.arange(1, input_ids.shape[1]).unsqueeze(0)
        mask = positions >= prompt_lengths.unsqueeze(1)
        mask &= attention_mask[:, 1:].bool()
        expected = (token_log_probs * mask).sum(dim=1)

        torch.testing.assert_close(actual, expected)

    def test_rejects_non_positive_chunk_size(self) -> None:
        with self.assertRaisesRegex(ValueError, "chunk"):
            sequence_log_likelihoods(
                logits=torch.randn(1, 3, 5),
                input_ids=torch.tensor([[1, 2, 3]]),
                prompt_lengths=torch.tensor([1]),
                attention_mask=torch.ones(1, 3, dtype=torch.long),
                token_chunk_size=0,
            )


class CausalCandidateScorerTest(unittest.TestCase):
    def test_single_token_candidates_use_final_prompt_state_and_preserve_representation(self) -> None:
        tokenizer = _TinyTokenizer()
        model = _TinyCausalModel()
        scorer = CausalCandidateScorer(model=model, tokenizer=tokenizer, batch_size=2, max_length=16)
        messages = [[{"role": "user", "content": "a story"}], [{"role": "user", "content": "another story"}]]

        probabilities, representations = scorer.score_messages_with_representations(messages, ["A", "B"])

        prompt_ids = torch.tensor([[3, 4], [3, 5]], dtype=torch.long)
        hidden = model.model(input_ids=prompt_ids, attention_mask=torch.ones_like(prompt_ids), return_dict=True).last_hidden_state[:, -1]
        expected_scores = torch.nn.functional.linear(hidden, model.lm_head.weight[[1, 2]], model.lm_head.bias[[1, 2]])
        expected_probabilities = [softmax_probabilities(row) for row in expected_scores.tolist()]
        torch.testing.assert_close(torch.tensor(probabilities), torch.tensor(expected_probabilities))
        torch.testing.assert_close(torch.tensor(representations), hidden)


class _TinyTokenizer:
    pad_token_id = 0
    eos_token = "<eos>"
    pad_token = None

    def __call__(self, value: str, *, add_special_tokens: bool) -> dict[str, list[int]]:
        candidates = {"A": [1], "B": [2]}
        if value in candidates:
            return {"input_ids": candidates[value]}
        return {"input_ids": [3, 4 if value == "a story" else 5]}

    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
        enable_thinking: bool,
    ) -> str:
        return messages[-1]["content"]


class _TinyBackbone(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embedding = torch.nn.Embedding(8, 3)
        with torch.no_grad():
            self.embedding.weight.copy_(torch.arange(24, dtype=torch.float32).reshape(8, 3) / 10.0)

    def forward(self, *, input_ids: torch.Tensor, attention_mask: torch.Tensor, return_dict: bool) -> SimpleNamespace:
        del attention_mask, return_dict
        return SimpleNamespace(last_hidden_state=self.embedding(input_ids))


class _TinyCausalModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = _TinyBackbone()
        self.lm_head = torch.nn.Linear(3, 8)
        with torch.no_grad():
            self.lm_head.weight.copy_(torch.arange(24, dtype=torch.float32).reshape(8, 3) / 20.0)
            self.lm_head.bias.copy_(torch.arange(8, dtype=torch.float32) / 100.0)

    def get_output_embeddings(self) -> torch.nn.Linear:
        return self.lm_head

    def forward(self, *args: object, **kwargs: object) -> None:
        raise AssertionError("single-token scoring should call the backbone directly")
