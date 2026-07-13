from __future__ import annotations

import unittest

import torch

from mias_dcms.zero_shot_scoring import sequence_log_likelihoods


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
