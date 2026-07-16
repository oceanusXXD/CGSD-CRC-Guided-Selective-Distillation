from __future__ import annotations

import unittest

from mias_dcms.preference_gradient_utility import (
    gradient_dpo_utility,
    select_gradient_dpo_candidates,
)


class PreferenceGradientUtilityTest(unittest.TestCase):
    def test_stage_one_keeps_top_four_budget_candidates_with_audit_fields(self) -> None:
        rows = [
            {"sample_id": f"p{index}", "gradient_dpo_cheap_score": float(index) / 10.0}
            for index in range(10)
        ]

        selected = select_gradient_dpo_candidates(rows, budget=2, candidate_multiplier=4)

        self.assertEqual(["p9", "p8", "p7", "p6", "p5", "p4", "p3", "p2"], [row["sample_id"] for row in selected])
        self.assertEqual(1, selected[0]["gradient_dpo_stage1_rank"])
        self.assertEqual(8, selected[0]["gradient_dpo_stage1_candidate_count"])
        self.assertEqual(4, selected[0]["gradient_dpo_candidate_multiplier"])

    def test_gradient_utility_is_length_normalized(self) -> None:
        short = gradient_dpo_utility(cheap_score=0.8, gradient_norm=3.0, pair_token_count=9)
        long = gradient_dpo_utility(cheap_score=0.8, gradient_norm=3.0, pair_token_count=36)

        self.assertAlmostEqual(0.8, short)
        self.assertAlmostEqual(0.4, long)

    def test_stage_one_reserves_observable_coverage_before_filling_by_score(self) -> None:
        rows = [
            {"sample_id": "a-high", "gradient_dpo_cheap_score": 0.9, "prompt_cluster": "a"},
            {"sample_id": "a-low", "gradient_dpo_cheap_score": 0.8, "prompt_cluster": "a"},
            {"sample_id": "b-low", "gradient_dpo_cheap_score": 0.1, "prompt_cluster": "b"},
        ]

        selected = select_gradient_dpo_candidates(
            rows,
            budget=1,
            candidate_multiplier=2,
            coverage_fields=("prompt_cluster",),
        )

        self.assertEqual({"a-high", "b-low"}, {row["sample_id"] for row in selected})
        self.assertEqual(1, sum(row["gradient_dpo_stage1_coverage_seed"] for row in selected if row["sample_id"] == "b-low"))

    def test_gradient_utility_rejects_invalid_inputs(self) -> None:
        with self.assertRaisesRegex(ValueError, "pair_token_count"):
            gradient_dpo_utility(cheap_score=1.0, gradient_norm=1.0, pair_token_count=0)


if __name__ == "__main__":
    unittest.main()
