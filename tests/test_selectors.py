from __future__ import annotations

import unittest

from mias_dcms.selectors import (
    assert_selector_rows_are_label_safe,
    entropy_uncertainty_scores,
    margin_uncertainty_scores,
    moment_matched_random,
    random_without_replacement,
    select_top_budget,
)


class SelectorToolsTest(unittest.TestCase):
    def test_random_without_replacement_is_seeded_and_budget_exact(self) -> None:
        first = random_without_replacement(["a", "b", "c", "d"], budget=2, seed=13)
        second = random_without_replacement(["a", "b", "c", "d"], budget=2, seed=13)

        self.assertEqual(first, second)
        self.assertEqual(2, len(first))
        self.assertEqual(2, len(set(first)))

    def test_select_top_budget_uses_score_descending_then_stable_id_tie_break(self) -> None:
        selected = select_top_budget(
            sample_ids=["b", "a", "c"],
            scores=[0.9, 0.9, 0.1],
            budget=2,
        )

        self.assertEqual(["a", "b"], selected)

    def test_entropy_uncertainty_scores_are_highest_for_uniform_probabilities(self) -> None:
        scores = entropy_uncertainty_scores([[0.5, 0.5], [0.99, 0.01]])

        self.assertGreater(scores[0], scores[1])

    def test_margin_uncertainty_scores_are_highest_for_small_class_margin(self) -> None:
        scores = margin_uncertainty_scores([[0.5, 0.5], [0.8, 0.2], [0.4, 0.35, 0.25]])

        self.assertGreater(scores[0], scores[1])
        self.assertGreater(scores[2], scores[1])

    def test_selector_label_safety_rejects_hidden_oracle_fields(self) -> None:
        assert_selector_rows_are_label_safe([{"sample_id": "safe", "score": 0.2}])

        with self.assertRaises(ValueError):
            assert_selector_rows_are_label_safe([{"sample_id": "leaky", "oracle_label": 1}])

    def test_moment_matched_random_matches_target_moments_without_scores(self) -> None:
        result = moment_matched_random(
            sample_ids=["a0", "a1", "a2", "b0", "b1", "b2"],
            group_membership=[
                {"A": 1.0, "B": 0.0},
                {"A": 1.0, "B": 0.0},
                {"A": 1.0, "B": 0.0},
                {"A": 0.0, "B": 1.0},
                {"A": 0.0, "B": 1.0},
                {"A": 0.0, "B": 1.0},
            ],
            budget=4,
            target_moments={"A": 0.5, "B": 0.5},
            tolerance=0.0,
            seed=17,
        )

        self.assertEqual(4, len(result.selected_ids))
        self.assertEqual(4, len(set(result.selected_ids)))
        self.assertEqual({"A": 0.5, "B": 0.5}, result.rounded_moments)
        self.assertEqual(0.0, result.max_constraint_violation)
        self.assertEqual(17, result.seed)

    def test_moment_matched_random_raises_when_no_batch_matches_target(self) -> None:
        with self.assertRaises(ValueError):
            moment_matched_random(
                sample_ids=["a0", "a1", "a2"],
                group_membership=[
                    {"A": 1.0, "B": 0.0},
                    {"A": 1.0, "B": 0.0},
                    {"A": 1.0, "B": 0.0},
                ],
                budget=2,
                target_moments={"A": 0.5, "B": 0.5},
                tolerance=0.0,
                seed=3,
            )


if __name__ == "__main__":
    unittest.main()
