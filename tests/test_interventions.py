from __future__ import annotations

import unittest

from mias_dcms.interventions import (
    apply_class_intercept,
    apply_length_coefficient,
    entropy_scores_from_logits,
    fixed_budget_response_curve,
    normalized_length_gap,
)


class InterventionToolsTest(unittest.TestCase):
    def test_class_intercept_only_changes_target_logit(self) -> None:
        logits = [[1.0, 2.0, 3.0], [0.0, -1.0, 4.0]]

        shifted = apply_class_intercept(logits, target_class=1, alpha=0.5)

        self.assertEqual([[1.0, 2.5, 3.0], [0.0, -0.5, 4.0]], shifted)
        self.assertEqual([[1.0, 2.0, 3.0], [0.0, -1.0, 4.0]], logits)

    def test_entropy_scores_are_recomputed_after_intercept(self) -> None:
        logits = [[0.0, 0.0], [3.0, 0.0]]
        base = entropy_scores_from_logits(logits)
        shifted = entropy_scores_from_logits(apply_class_intercept(logits, target_class=1, alpha=3.0))

        self.assertGreater(base[0], shifted[0])
        self.assertGreater(shifted[1], base[1])

    def test_length_coefficient_adds_gamma_scaled_normalized_gap(self) -> None:
        base_margins = [0.2, 0.2]
        gaps = [
            normalized_length_gap(response_a_length=30, response_b_length=10),
            normalized_length_gap(response_a_length=10, response_b_length=30),
        ]

        adjusted = apply_length_coefficient(base_margins, gaps, gamma=0.4)

        self.assertEqual([0.4, 0.0], adjusted)

    def test_fixed_budget_response_curve_records_propensity_by_intervention_value(self) -> None:
        curve = fixed_budget_response_curve(
            sample_ids=["a", "b", "c", "d"],
            groups=["target", "other", "target", "other"],
            score_by_value={
                -1.0: [0.1, 0.9, 0.2, 0.8],
                0.0: [0.9, 0.8, 0.1, 0.2],
                1.0: [0.9, 0.1, 0.8, 0.2],
            },
            budget=2,
            target_group="target",
        )

        by_value = {point.value: point for point in curve.points}
        self.assertEqual(["b", "d"], by_value[-1.0].selected_ids)
        self.assertEqual(["a", "b"], by_value[0.0].selected_ids)
        self.assertEqual(["a", "c"], by_value[1.0].selected_ids)
        self.assertEqual(0.0, by_value[-1.0].target_group_propensity)
        self.assertEqual(0.5, by_value[0.0].target_group_propensity)
        self.assertEqual(1.0, by_value[1.0].target_group_propensity)
        self.assertEqual(2, by_value[1.0].budget)


if __name__ == "__main__":
    unittest.main()
