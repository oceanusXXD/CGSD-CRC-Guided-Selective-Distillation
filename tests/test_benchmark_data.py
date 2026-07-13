from __future__ import annotations

import unittest

from mias_dcms.benchmark_data import normalize_trec_row, reservoir_sample_per_class


class BenchmarkDataTest(unittest.TestCase):
    def test_trec_normalizer_accepts_official_string_coarse_labels(self) -> None:
        row = normalize_trec_row(
            {"text": "Where is Paris?", "coarse_label": "LOC", "fine_label": "LOC:city"},
            split="train",
            index=7,
        )

        self.assertEqual(4, row["label"])
        self.assertEqual("LOCATION", row["label_name"])
        self.assertEqual("trec:train:7", row["id"])

    def test_reservoir_sampling_caps_each_trec_class_for_smoke_pools(self) -> None:
        rows = [
            {"id": f"row-{label}-{index}", "label": label}
            for label in range(3)
            for index in range(5)
        ]

        sampled = reservoir_sample_per_class(rows, per_class=2, seed=1000)

        self.assertEqual(6, len(sampled))
        self.assertEqual({0: 2, 1: 2, 2: 2}, {
            label: sum(int(row["label"]) == label for row in sampled)
            for label in range(3)
        })


if __name__ == "__main__":
    unittest.main()
