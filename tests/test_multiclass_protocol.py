from __future__ import annotations

import unittest

from mias_dcms.multiclass_protocol import (
    build_fixed_multiclass_splits,
    pool_class_prior,
    validate_disjoint_splits,
)


class MulticlassProtocolTest(unittest.TestCase):
    def test_pool_class_prior_reports_counts_and_shares(self) -> None:
        rows = [
            {"id": "a", "label": "World"},
            {"id": "b", "label": "Sports"},
            {"id": "c", "label": "Sports"},
            {"id": "d", "label": "Business"},
        ]

        prior = pool_class_prior(rows, label_field="label")

        self.assertEqual(4, prior.total_count)
        self.assertEqual({"Business": 1, "Sports": 2, "World": 1}, prior.class_counts)
        self.assertEqual(0.5, prior.class_shares["Sports"])

    def test_fixed_splits_are_seeded_disjoint_and_budget_exact(self) -> None:
        rows = [{"id": f"s{i}", "label": i % 3} for i in range(12)]

        first = build_fixed_multiclass_splits(
            rows,
            seed=42,
            seed_size=3,
            active_size=5,
            test_size=4,
        )
        second = build_fixed_multiclass_splits(
            rows,
            seed=42,
            seed_size=3,
            active_size=5,
            test_size=4,
        )

        self.assertEqual(first, second)
        self.assertEqual(3, len(first["seed_ids"]))
        self.assertEqual(5, len(first["active_pool_ids"]))
        self.assertEqual(4, len(first["test_ids"]))
        validate_disjoint_splits(first)

    def test_split_validation_rejects_overlap(self) -> None:
        with self.assertRaises(ValueError):
            validate_disjoint_splits(
                {
                    "seed_ids": ["a", "b"],
                    "active_pool_ids": ["b", "c"],
                    "test_ids": ["d"],
                }
            )


if __name__ == "__main__":
    unittest.main()
