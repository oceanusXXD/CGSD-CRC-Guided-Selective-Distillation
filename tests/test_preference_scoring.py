from __future__ import annotations

import math
import unittest

from mias_dcms.preference_scoring import (
    active_dpo_scores,
    apl_scores,
    build_preference_baseline_score_rows,
    gradient_dpo_cheap_score_components,
    reward_margin_scores,
)


class PreferenceScoringTest(unittest.TestCase):
    def test_reward_margin_scores_are_high_for_uncertain_preference_pairs(self) -> None:
        rows = [
            {"sample_id": "uncertain", "probability_response_1": 0.52},
            {"sample_id": "confident", "probability_response_1": 0.90},
            {"sample_id": "tie", "probability_response_1": 0.50},
        ]

        scores = reward_margin_scores(rows)

        self.assertAlmostEqual(0.96, scores["uncertain"])
        self.assertAlmostEqual(0.20, scores["confident"])
        self.assertAlmostEqual(1.00, scores["tie"])

    def test_reward_margin_prefers_implicit_reward_gap_when_available(self) -> None:
        scores = reward_margin_scores([
            {"sample_id": "implicit", "implicit_reward_gap": 0.0, "probability_response_1": 0.99},
        ])

        self.assertAlmostEqual(1.0, scores["implicit"])

    def test_apl_scores_combine_preference_uncertainty_with_prompt_entropy(self) -> None:
        rows = [
            {
                "sample_id": "diverse",
                "probability_response_1": 0.55,
                "prompt_cluster_probabilities": [0.5, 0.5],
            },
            {
                "sample_id": "single_cluster",
                "probability_response_1": 0.55,
                "prompt_cluster_probabilities": [1.0, 0.0],
            },
        ]

        scores = apl_scores(rows, prompt_entropy_weight=0.25)

        self.assertGreater(scores["diverse"], scores["single_cluster"])
        self.assertAlmostEqual(0.90 + 0.25 * math.log(2), scores["diverse"])
        self.assertAlmostEqual(0.90, scores["single_cluster"])

    def test_active_dpo_scores_use_policy_reference_gap_magnitude(self) -> None:
        rows = [
            {
                "sample_id": "large_gradient_proxy",
                "policy_logprob_response_1": -1.0,
                "policy_logprob_response_2": -2.0,
                "reference_logprob_response_1": -1.8,
                "reference_logprob_response_2": -1.2,
                "probability_response_1": 0.65,
            },
            {
                "sample_id": "small_gradient_proxy",
                "policy_logprob_response_1": -1.0,
                "policy_logprob_response_2": -1.3,
                "reference_logprob_response_1": -1.1,
                "reference_logprob_response_2": -1.25,
                "probability_response_1": 0.65,
            },
        ]

        scores = active_dpo_scores(rows)

        self.assertAlmostEqual(1.6, scores["large_gradient_proxy"])
        self.assertAlmostEqual(0.15, scores["small_gradient_proxy"])
        self.assertGreater(scores["large_gradient_proxy"], scores["small_gradient_proxy"])

    def test_active_dpo_scores_can_length_normalize_gradient_proxy(self) -> None:
        rows = [
            {
                "sample_id": "short_pair",
                "policy_logprob_gap": 2.0,
                "reference_logprob_gap": 0.0,
                "token_count_response_1": 2,
                "token_count_response_2": 2,
            },
            {
                "sample_id": "long_pair",
                "policy_logprob_gap": 3.0,
                "reference_logprob_gap": 0.0,
                "token_count_response_1": 10,
                "token_count_response_2": 10,
            },
        ]

        scores = active_dpo_scores(rows, length_normalize=True)

        self.assertAlmostEqual(0.5, scores["short_pair"])
        self.assertAlmostEqual(0.15, scores["long_pair"])
        self.assertGreater(scores["short_pair"], scores["long_pair"])

    def test_active_dpo_accepts_logprob_generator_token_count_schema(self) -> None:
        rows = [
            {
                "sample_id": "generated",
                "policy_logprob_gap": 2.0,
                "reference_logprob_gap": 0.0,
                "response_1_token_count": 4,
                "response_2_token_count": 6,
            }
        ]

        scores = active_dpo_scores(rows, length_normalize=True)

        self.assertAlmostEqual(0.2, scores["generated"])

    def test_active_dpo_rows_include_fixed_pool_adaptation_components(self) -> None:
        rows = [
            {
                "sample_id": "ambiguous_cluster",
                "probability_response_1": 0.6,
                "policy_logprob_gap": 1.0,
                "reference_logprob_gap": 0.0,
                "token_count_response_1": 5,
                "token_count_response_2": 5,
                "prompt_cluster_probabilities": [0.5, 0.5],
            },
            {
                "sample_id": "single_cluster",
                "probability_response_1": 0.6,
                "policy_logprob_gap": 1.0,
                "reference_logprob_gap": 0.0,
                "token_count_response_1": 5,
                "token_count_response_2": 5,
                "prompt_cluster_probabilities": [1.0, 0.0],
            },
        ]

        scored = build_preference_baseline_score_rows(
            rows,
            methods=("active_dpo",),
            active_dpo_length_normalize=True,
            active_dpo_novelty_weight=0.25,
        )
        by_id = {row["sample_id"]: row for row in scored}

        self.assertAlmostEqual(1.0, by_id["single_cluster"]["active_dpo_gradient_proxy"])
        self.assertAlmostEqual(0.1, by_id["single_cluster"]["active_dpo_length_normalized_proxy"])
        self.assertAlmostEqual(0.0, by_id["single_cluster"]["active_dpo_novelty_score"])
        self.assertGreater(
            by_id["ambiguous_cluster"]["active_dpo_score"],
            by_id["single_cluster"]["active_dpo_score"],
        )
        self.assertEqual(
            by_id["ambiguous_cluster"]["active_dpo_score"],
            by_id["ambiguous_cluster"]["selector_scores"]["active_dpo"],
        )

    def test_gradient_dpo_cheap_score_uses_uncertainty_and_dpo_sensitivity(self) -> None:
        components = gradient_dpo_cheap_score_components(
            [
                {
                    "sample_id": "uncertain_sensitive",
                    "probability_response_1": 0.5,
                    "policy_logprob_gap": 0.0,
                    "reference_logprob_gap": 0.0,
                    "response_1_token_count": 2,
                    "response_2_token_count": 3,
                },
                {
                    "sample_id": "confident_shifted",
                    "probability_response_1": 0.9,
                    "policy_logprob_gap": 3.0,
                    "reference_logprob_gap": 0.0,
                    "response_1_token_count": 2,
                    "response_2_token_count": 3,
                },
            ]
        )

        self.assertAlmostEqual(1.0, components["uncertain_sensitive"]["gradient_dpo_cheap_score"])
        self.assertAlmostEqual(0.2 * 0.25, components["confident_shifted"]["gradient_dpo_cheap_score"])
        self.assertGreater(
            components["uncertain_sensitive"]["gradient_dpo_cheap_score"],
            components["confident_shifted"]["gradient_dpo_cheap_score"],
        )

    def test_build_preference_baseline_score_rows_rejects_hidden_label_fields(self) -> None:
        rows = [
            {
                "sample_id": "leaky",
                "probability_response_1": 0.5,
                "preference_label": "A",
            }
        ]

        with self.assertRaisesRegex(ValueError, "hidden fields"):
            build_preference_baseline_score_rows(rows, methods=("reward_margin",))

    def test_build_preference_baseline_score_rows_preserves_auditable_inputs(self) -> None:
        rows = [
            {
                "sample_id": "p1",
                "probability_response_1": 0.6,
                "length_gap": 0.25,
                "prompt_cluster_probabilities": [0.25, 0.75],
                "policy_logprob_response_1": -0.5,
                "policy_logprob_response_2": -0.8,
                "reference_logprob_response_1": -0.7,
                "reference_logprob_response_2": -0.6,
            }
        ]

        scored = build_preference_baseline_score_rows(
            rows,
            methods=("reward_margin", "apl", "active_dpo"),
            prompt_entropy_weight=0.5,
        )

        self.assertEqual(1, len(scored))
        self.assertEqual("p1", scored[0]["sample_id"])
        self.assertIn("reward_margin_score", scored[0])
        self.assertIn("apl_score", scored[0])
        self.assertIn("active_dpo_score", scored[0])
        self.assertIn("selector_scores", scored[0])
        self.assertEqual(scored[0]["reward_margin_score"], scored[0]["selector_scores"]["reward_margin"])
        self.assertEqual(scored[0]["apl_score"], scored[0]["selector_scores"]["apl"])
        self.assertEqual(scored[0]["active_dpo_score"], scored[0]["selector_scores"]["active_dpo"])
        self.assertNotIn("preference_label", scored[0])


if __name__ == "__main__":
    unittest.main()
