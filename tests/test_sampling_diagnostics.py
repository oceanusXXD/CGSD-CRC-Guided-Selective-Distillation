from __future__ import annotations

import unittest

from mias_dcms.sampling_diagnostics import select_classification_rows, select_rows, selector_safe_view


class SamplingDiagnosticsTest(unittest.TestCase):
    def test_badge_and_galaxy_select_representation_diverse_rows(self) -> None:
        rows = [
            {"id": "a", "probabilities": [0.99, 0.01], "representation_embedding": [1.0, 0.0]},
            {"id": "b", "probabilities": [0.51, 0.49], "representation_embedding": [0.0, 1.0]},
            {"id": "c", "probabilities": [0.50, 0.50], "representation_embedding": [-1.0, 0.0]},
        ]

        badge = select_rows(rows, method="badge", budget=2, seed=7)
        galaxy = select_rows(rows, method="galaxy", budget=2, seed=7)

        self.assertEqual(2, len(badge))
        self.assertEqual(2, len(galaxy))
        self.assertEqual(2, len({row["id"] for row in badge}))
        self.assertEqual(2, len({row["id"] for row in galaxy}))

    def test_badge_requires_a_representation_embedding(self) -> None:
        with self.assertRaisesRegex(ValueError, "representation_embedding"):
            select_rows(
                [{"id": "a", "probabilities": [0.5, 0.5]}],
                method="badge",
                budget=1,
                seed=1,
            )

    def test_representation_selectors_honor_zero_budget(self) -> None:
        rows = [
            {"id": "a", "probabilities": [0.5, 0.5], "representation_embedding": [1.0, 0.0]},
            {"id": "b", "probabilities": [0.5, 0.5], "representation_embedding": [0.0, 1.0]},
        ]

        self.assertEqual([], select_rows(rows, method="badge", budget=0, seed=7))
        self.assertEqual([], select_rows(rows, method="galaxy", budget=0, seed=7))

    def test_entropy_dcms_uses_soft_class_posterior_without_true_labels(self) -> None:
        rows = [
            {"id": "a", "probabilities": [0.99, 0.01]},
            {"id": "b", "probabilities": [0.90, 0.10]},
            {"id": "c", "probabilities": [0.10, 0.90]},
            {"id": "d", "probabilities": [0.01, 0.99]},
        ]

        selected, metadata = select_classification_rows(
            rows,
            method="Entropy+DCMS",
            budget=2,
            seed=3,
            dcms_slack_grid=(0.0,),
            dcms_kappa=0.5,
        )

        self.assertIsNotNone(metadata)
        self.assertEqual(2, len(selected))
        self.assertEqual({"class=0": 0.5, "class=1": 0.5}, metadata["target_moments"])
        self.assertLessEqual(metadata["max_constraint_violation"], 1e-12)
        self.assertEqual(4, len(metadata["q_propensity"]))
        self.assertEqual(2, sum(metadata["selection_indicator"].values()))

    def test_classification_dcms_rejects_true_labels_in_solver_input(self) -> None:
        with self.assertRaisesRegex(ValueError, "hidden fields"):
            select_classification_rows(
                [{"id": "a", "label": 0, "probabilities": [0.5, 0.5]}],
                method="entropy_dcms",
                budget=1,
                seed=1,
            )

    def test_entropy_gradient_dcms_uses_top_four_budget_pool_and_semantic_coverage(self) -> None:
        rows = [
            {"id": "a", "probabilities": [0.5, 0.5], "representation_embedding": [3.0, 0.0]},
            {"id": "b", "probabilities": [0.6, 0.4], "representation_embedding": [0.0, 3.0]},
            {"id": "c", "probabilities": [0.4, 0.6], "representation_embedding": [-3.0, 0.0]},
            {"id": "d", "probabilities": [0.5, 0.5], "representation_embedding": [0.0, -3.0]},
        ]

        selected, metadata = select_classification_rows(
            rows,
            method="EntropyGradient+DCMS",
            budget=2,
            seed=5,
            candidate_multiplier=2,
            semantic_cluster_count=2,
            dcms_slack_grid=(0.0, 0.5),
            dcms_kappa=0.5,
        )

        self.assertEqual(2, len(selected))
        self.assertIsNotNone(metadata)
        assert metadata is not None
        self.assertEqual(4, metadata["stage1_candidate_count"])
        self.assertEqual(2, metadata["semantic_coverage"]["cluster_count"])
        self.assertTrue(any(key.startswith("semantic_cluster=") for key in metadata["target_moments"]))
        self.assertEqual(4, len(metadata["q_propensity"]))

    def test_selector_safe_view_removes_derived_classification_oracle_fields(self) -> None:
        safe = selector_safe_view(
            [
                {
                    "id": "a",
                    "label": 0,
                    "label_name": "World",
                    "prediction_correct": True,
                    "probabilities": [0.7, 0.3],
                }
            ]
        )[0]

        self.assertNotIn("label", safe)
        self.assertNotIn("label_name", safe)
        self.assertNotIn("prediction_correct", safe)
        self.assertIn("probabilities", safe)


if __name__ == "__main__":
    unittest.main()
