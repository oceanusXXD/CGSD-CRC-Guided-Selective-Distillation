from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch.utils.data import DataLoader

from mias_dcms.benchmark_training import LoraTrainingConfig, _run_sft_loop


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


if __name__ == "__main__":
    unittest.main()
