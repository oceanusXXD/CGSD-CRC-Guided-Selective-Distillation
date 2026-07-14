from __future__ import annotations

import unittest

from mias_dcms.binary_benchmark_data import (
    EmptyBinaryBenchmarkTextError,
    normalize_binary_benchmark_row,
    validate_normalized_binary_rows,
)


class BinaryBenchmarkDataTest(unittest.TestCase):
    def test_imdb_normalization_preserves_native_binary_label(self) -> None:
        row = normalize_binary_benchmark_row(
            "imdb",
            {"text": "A tense and memorable film.", "label": 1},
            split="train",
            index=7,
        )

        self.assertEqual("imdb:train:7", row["id"])
        self.assertEqual(1, row["groundtruth"])
        self.assertEqual(1, row["native_label"])
        self.assertEqual("positive", row["label_name"])

    def test_paws_uses_the_two_native_sentences_without_label_rewriting(self) -> None:
        row = normalize_binary_benchmark_row(
            "paws_labeled_final",
            {"id": 23, "sentence1": "Birds fly.", "sentence2": "Birds can fly.", "label": 0},
            split="validation",
            index=2,
        )

        self.assertEqual("paws_labeled_final:validation:23", row["id"])
        self.assertIn("Birds fly.", str(row["query"]))
        self.assertIn("Birds can fly.", str(row["document"]))
        self.assertEqual(0, row["groundtruth"])

    def test_validator_rejects_changed_native_label(self) -> None:
        row = normalize_binary_benchmark_row(
            "tweeteval_hate",
            {"text": "example", "label": 1},
            split="test",
            index=0,
        )
        row["native_label"] = 0

        with self.assertRaisesRegex(ValueError, "changed its native label"):
            validate_normalized_binary_rows([row], dataset_name="tweeteval_hate", split="test")

    def test_normalizer_rejects_an_empty_native_text_instead_of_imputing(self) -> None:
        with self.assertRaises(EmptyBinaryBenchmarkTextError):
            normalize_binary_benchmark_row(
                "tweeteval_hate",
                {"text": "  ", "label": 0},
                split="train",
                index=3,
            )


if __name__ == "__main__":
    unittest.main()
