from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from scripts.generate_preference_logprobs import (
    _load_completed_rows,
    _ordered_logprob_rows,
    _resolve_model_source,
)
from mias_dcms.utils import write_jsonl


class GeneratePreferenceLogprobsTest(unittest.TestCase):
    def test_local_only_preserves_huggingface_model_id(self) -> None:
        self.assertEqual(
            "Qwen/Qwen3-0.6B",
            _resolve_model_source("Qwen/Qwen3-0.6B", local_files_only=True),
        )

    def test_resume_uses_partial_checkpoint_and_preserves_input_order(self) -> None:
        source = [{"sample_id": "a"}, {"sample_id": "b"}, {"sample_id": "c"}]
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "logprobs.jsonl"
            partial_path = Path(tmpdir) / "logprobs.partial.jsonl"
            write_jsonl([{"sample_id": "b", "score": 2.0}], partial_path)

            completed, source_name = _load_completed_rows(
                source,
                output_path=output_path,
                partial_path=partial_path,
                id_field="sample_id",
                resume=True,
            )
            ordered = _ordered_logprob_rows(
                source,
                [
                    {"sample_id": "c", "score": 3.0},
                    {"sample_id": "a", "score": 1.0},
                    completed["b"],
                ],
                id_field="sample_id",
            )

        self.assertEqual("partial", source_name)
        self.assertEqual({"b"}, set(completed))
        self.assertEqual(["a", "b", "c"], [row["sample_id"] for row in ordered])

    def test_resume_rejects_extra_checkpoint_id(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "logprobs.jsonl"
            partial_path = Path(tmpdir) / "logprobs.partial.jsonl"
            write_jsonl([{"sample_id": "outside"}], partial_path)

            with self.assertRaisesRegex(ValueError, "not in the input pool"):
                _load_completed_rows(
                    [{"sample_id": "a"}],
                    output_path=output_path,
                    partial_path=partial_path,
                    id_field="sample_id",
                    resume=True,
                )
