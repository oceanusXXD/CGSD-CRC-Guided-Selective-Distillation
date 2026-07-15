from __future__ import annotations

import unittest

from mias_dcms.binary_reaudit import (
    materialize_binary_reaudit_selection,
    prepare_binary_reaudit_splits,
)


def _source_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for index in range(20):
        rows.append(
            {
                "id": f"row-{index}",
                "query": "Is this positive?",
                "document": f"document {index}",
                "groundtruth": index % 2,
            }
        )
    return rows


class BinaryReauditTest(unittest.TestCase):
    def test_prepare_hides_active_oracle_labels_and_keeps_splits_disjoint(self) -> None:
        artifacts = prepare_binary_reaudit_splits(
            _source_rows(),
            dataset="toy_binary",
            seed_label_count=4,
            active_pool_size=10,
            test_size=6,
            seed=17,
        )

        active = artifacts["selection_pool"]
        self.assertEqual(10, len(active))
        self.assertTrue(all("label" not in row for row in active))
        self.assertEqual(10, len(artifacts["selection_oracle_store"]))
        manifest = artifacts["split_manifest"]
        ids = set(manifest["seed_ids"])
        self.assertFalse(ids.intersection(manifest["active_pool_ids"]))
        self.assertFalse(ids.intersection(manifest["test_ids"]))

    def test_selection_uses_safe_scores_then_materializes_audit_and_train_rows(self) -> None:
        artifacts = prepare_binary_reaudit_splits(
            _source_rows(),
            dataset="toy_binary",
            seed_label_count=4,
            active_pool_size=10,
            test_size=6,
            seed=17,
        )
        scored_rows = [
            {
                **row,
                "probabilities": [0.5 + 0.01 * index, 0.5 - 0.01 * index],
                "entropy": 1.0 - 0.01 * index,
                "margin": 0.02 * index,
            }
            for index, row in enumerate(artifacts["selection_pool"])
        ]

        results = materialize_binary_reaudit_selection(
            scored_rows,
            oracle_store=artifacts["selection_oracle_store"],
            seed_train_rows=artifacts["seed_train_rows"],
            dataset="toy_binary",
            model="toy-model",
            methods=["random", "entropy"],
            budget=3,
            seed=5,
            config_hash="toy-config",
            evaluation_label_count=6,
        )

        self.assertEqual({"random", "entropy"}, set(results))
        entropy = results["entropy"]
        self.assertEqual(3, len(entropy["selected_ids"]))
        self.assertEqual(7, len(entropy["train_rows"]))
        self.assertTrue(all(row["query"] and row["document"] for row in entropy["revealed_rows"]))
        self.assertEqual(3, entropy["cost_metrics"]["active_label_count"])
        self.assertEqual(7, entropy["cost_metrics"]["supervision_budget_total"])
        self.assertEqual(0.0, entropy["selection_metrics"]["total_absolute_prediction_error"])


if __name__ == "__main__":
    unittest.main()
