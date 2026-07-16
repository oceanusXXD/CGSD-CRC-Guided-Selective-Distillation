from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch.utils.data import DataLoader

from mias_dcms.benchmark_training import (
    LoraTrainingConfig,
    _run_dpo_accumulation_group,
    _run_dpo_loop,
    _run_sft_loop,
    _trim_dpo_batch_to_token_budget,
)


class _ToySFTModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(0.5))

    def forward(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor,
    ) -> SimpleNamespace:
        del attention_mask, labels
        loss = (self.weight - input_ids.float().mean() / 10.0).pow(2)
        return SimpleNamespace(loss=loss)


class _ToyDPOModel(torch.nn.Module):
    def __init__(self, value: float) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(value))

    def forward(self, *, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> SimpleNamespace:
        del attention_mask
        logits = torch.zeros((*input_ids.shape, 3), dtype=torch.float32)
        logits[:, :, 1] = self.weight
        logits[:, :, 2] = -self.weight
        return SimpleNamespace(logits=logits)


class _ToyAccelerator:
    def no_sync(self, model: torch.nn.Module):
        del model
        from contextlib import nullcontext

        return nullcontext()

    def backward(self, loss: torch.Tensor) -> None:
        loss.backward()

    def clip_grad_norm_(self, parameters, max_norm: float) -> None:
        torch.nn.utils.clip_grad_norm_(parameters, max_norm)


class BenchmarkTrainingTest(unittest.TestCase):
    def test_sft_loop_honors_update_steps_even_when_more_than_one_epoch_is_needed(self) -> None:
        dataloader = DataLoader(
            [
                {
                    "input_ids": torch.tensor([1, 2]),
                    "attention_mask": torch.tensor([1, 1]),
                    "labels": torch.tensor([-100, 2]),
                },
                {
                    "input_ids": torch.tensor([2, 3]),
                    "attention_mask": torch.tensor([1, 1]),
                    "labels": torch.tensor([-100, 3]),
                },
            ],
            batch_size=1,
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            config = LoraTrainingConfig(
                model_name_or_path="toy",
                output_dir=Path(tmpdir),
                epochs=1,
                gradient_accumulation_steps=1,
                update_steps=3,
                mixed_precision="no",
            )
            with patch("mias_dcms.benchmark_training._save_adapter"):
                summary = _run_sft_loop(
                    model=_ToySFTModel(),
                    tokenizer=object(),
                    dataloader=dataloader,
                    config=config,
                )

        self.assertEqual(3, summary["optimizer_steps"])
        self.assertEqual(2, summary["epochs"])

    def test_dpo_batch_trimming_keeps_complete_pairs_within_token_budget(self) -> None:
        batch = {
            "input_ids": torch.tensor([[1, 2], [3, 4], [5, 6], [7, 8]]),
            "attention_mask": torch.ones((4, 2), dtype=torch.long),
            "prompt_lengths": torch.tensor([1, 1, 1, 1]),
            "pair_count": torch.tensor(2),
            "pair_input_token_counts": torch.tensor([4, 6]),
        }

        trimmed = _trim_dpo_batch_to_token_budget(batch, remaining_tokens=4)

        self.assertIsNotNone(trimmed)
        assert trimmed is not None
        self.assertEqual(1, int(trimmed["pair_count"].item()))
        self.assertEqual(4, int(trimmed["pair_input_token_counts"].sum().item()))
        self.assertEqual([[1, 2], [5, 6]], trimmed["input_ids"].tolist())
        self.assertIsNone(_trim_dpo_batch_to_token_budget(batch, remaining_tokens=3))

    def test_partial_dpo_accumulation_group_still_updates_policy(self) -> None:
        policy = _ToyDPOModel(0.0)
        reference = _ToyDPOModel(0.0)
        for parameter in reference.parameters():
            parameter.requires_grad_(False)
        optimizer = torch.optim.SGD(policy.parameters(), lr=0.1)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _step: 1.0)
        batch = {
            "input_ids": torch.tensor([[0, 1], [0, 2]], dtype=torch.long),
            "attention_mask": torch.ones((2, 2), dtype=torch.long),
            "prompt_lengths": torch.tensor([1, 1], dtype=torch.long),
            "pair_count": torch.tensor(1, dtype=torch.long),
            "pair_input_token_counts": torch.tensor([4], dtype=torch.long),
        }

        losses, _accuracies = _run_dpo_accumulation_group(
            [batch],
            policy=policy,
            reference=reference,
            accelerator=_ToyAccelerator(),
            optimizer=optimizer,
            scheduler=scheduler,
            beta=0.1,
            max_grad_norm=1.0,
        )

        self.assertEqual(1, len(losses))
        self.assertNotEqual(0.0, float(policy.weight.detach().item()))

    def test_dpo_token_budget_updates_a_partial_accumulation_window(self) -> None:
        policy = _ToyDPOModel(0.0)
        reference = _ToyDPOModel(0.0)
        for parameter in reference.parameters():
            parameter.requires_grad_(False)
        batch = {
            "input_ids": torch.tensor([[0, 1], [0, 2]], dtype=torch.long),
            "attention_mask": torch.ones((2, 2), dtype=torch.long),
            "prompt_lengths": torch.tensor([1, 1], dtype=torch.long),
            "pair_count": torch.tensor(1, dtype=torch.long),
            "pair_input_token_counts": torch.tensor([4], dtype=torch.long),
        }
        dataloader = DataLoader([batch], batch_size=None)

        with tempfile.TemporaryDirectory() as tmpdir:
            config = LoraTrainingConfig(
                model_name_or_path="toy",
                output_dir=Path(tmpdir),
                gradient_accumulation_steps=4,
                train_token_budget=4,
                mixed_precision="no",
            )
            with patch("mias_dcms.benchmark_training._save_adapter"):
                summary = _run_dpo_loop(
                    policy=policy,
                    reference=reference,
                    tokenizer=object(),
                    dataloader=dataloader,
                    config=config,
                )

        self.assertEqual(1, summary["optimizer_steps"])
        self.assertEqual(1, summary["processed_pair_count"])
        self.assertEqual(4, summary["processed_input_tokens"])
        self.assertTrue(summary["token_budget_exhausted"])
        self.assertNotEqual(0.0, float(policy.weight.detach().item()))


if __name__ == "__main__":
    unittest.main()
