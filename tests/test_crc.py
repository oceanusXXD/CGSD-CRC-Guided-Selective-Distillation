from __future__ import annotations

import unittest

from src.crc import (
    apply_crc_defer_set,
    calibrate_crc,
    compute_crc_error_mass_plan,
    select_training_ids,
)


class CRCTest(unittest.TestCase):
    def test_calibrate_crc_and_apply_defer_set(self) -> None:
        guide = [
            {"id": "g1", "score": 10.0, "prediction": 1, "label": 1},
            {"id": "g2", "score": -10.0, "prediction": 0, "label": 0},
            {"id": "g3", "score": 0.1, "prediction": 1, "label": 0},
        ]

        crc = calibrate_crc(guide, alpha=0.5, temperature=1.0)
        pool = apply_crc_defer_set(
            [
                {"id": "p_accept", "score": 10.0, "prediction": 1, "label": 1},
                {"id": "p_defer", "score": 0.0, "prediction": 1, "label": 0},
            ],
            lambda_hat=max(crc.lambda_hat, 0.8),
            temperature=crc.temperature,
        )

        self.assertIn("routing_score", pool[0])
        self.assertFalse(pool[0]["defer"])
        self.assertTrue(pool[1]["defer"])

    def test_crc_error_mass_selection_splits_accept_and_defer_budget(self) -> None:
        guide_decisions = [
            {"id": "g1", "prediction": 1, "label": 1, "defer": False},
            {"id": "g2", "prediction": 0, "label": 0, "defer": False},
            {"id": "g3", "prediction": 1, "label": 0, "defer": True},
            {"id": "g4", "prediction": 1, "label": 1, "defer": True},
        ]
        pool_decisions = [
            {"id": "a1", "prediction": 1, "label": 1, "defer": False, "routing_score": 0.9},
            {"id": "a2", "prediction": 1, "label": 1, "defer": False, "routing_score": 0.8},
            {"id": "d1", "prediction": 1, "label": 0, "defer": True, "routing_score": 0.6},
            {"id": "d2", "prediction": 0, "label": 1, "defer": True, "routing_score": 0.55},
        ]
        plan = compute_crc_error_mass_plan(
            guide_decisions,
            pool_decisions,
            budget=4,
            temperature=2.0,
            lambda_hat=0.8,
            alpha=0.1,
        )

        selection = select_training_ids(
            pool_decisions,
            method="crc-error-mass",
            budget=4,
            seed=1,
            crc_error_mass_plan=plan,
        )

        self.assertEqual(4, selection.requested_budget)
        self.assertEqual(selection.selected_ids, [*selection.accept_ids, *selection.defer_ids])
        self.assertGreaterEqual(len(selection.defer_ids), 1)

    def test_random_selection_ignores_defer_sides(self) -> None:
        pool_decisions = [
            {"id": f"p{i}", "prediction": 1, "label": 1, "defer": i % 2 == 0}
            for i in range(6)
        ]

        selection = select_training_ids(pool_decisions, method="random", budget=3, seed=3)

        self.assertEqual(3, selection.selected_budget)
        self.assertEqual(3, len(selection.selected_ids))


if __name__ == "__main__":
    unittest.main()
