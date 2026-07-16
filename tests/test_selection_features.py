from __future__ import annotations

import unittest

from mias_dcms.selection.features import merge_feature_rows


class SelectionFeaturesTest(unittest.TestCase):
    def test_merges_exact_label_safe_feature_coverage(self) -> None:
        rows = [{"id": "a", "text": "A"}, {"id": "b", "text": "B"}]
        features = [
            {"id": "a", "representation_embedding": [1.0, 0.0]},
            {"id": "b", "representation_embedding": [0.0, 1.0]},
        ]

        merged = merge_feature_rows(rows, features, source_name="features.jsonl")

        self.assertEqual([1.0, 0.0], merged[0]["representation_embedding"])
        self.assertEqual([0.0, 1.0], merged[1]["representation_embedding"])

    def test_rejects_incomplete_coverage_and_hidden_labels(self) -> None:
        with self.assertRaisesRegex(ValueError, "exactly cover"):
            merge_feature_rows(
                [{"id": "a"}, {"id": "b"}],
                [{"id": "a", "representation_embedding": [1.0]}],
                source_name="features.jsonl",
            )
        with self.assertRaisesRegex(ValueError, "hidden fields"):
            merge_feature_rows(
                [{"id": "a"}],
                [{"id": "a", "label": 1, "representation_embedding": [1.0]}],
                source_name="features.jsonl",
            )


if __name__ == "__main__":
    unittest.main()
