from __future__ import annotations

import unittest

from mias_dcms.preference_intervention_inputs import (
    build_preference_intervention_rows,
    length_gap_bin,
)


class PreferenceInterventionInputsTest(unittest.TestCase):
    def test_builds_selector_safe_intervention_rows_from_pool_logprobs_and_scores(self) -> None:
        active_pool = [
            {
                "sample_id": "p1",
                "prompt": "Prompt",
                "response_a": "one two three four",
                "response_b": "one",
                "source_pair": "human|model",
                "ab_position": "original",
                "prompt_cluster": "c1",
            },
            {
                "sample_id": "p2",
                "prompt": "Prompt",
                "response_a": "one",
                "response_b": "one two three four",
                "source_pair": "model|human",
                "ab_position": "swapped",
            },
        ]
        logprobs = [
            {"sample_id": "p1", "policy_logprob_gap": 1.0, "reference_logprob_gap": 0.2},
            {"sample_id": "p2", "implicit_reward_gap": -0.6},
        ]
        scores = [
            {
                "sample_id": "p1",
                "selector_scores": {"apl": 0.7, "active_dpo": 0.8},
            },
            {"sample_id": "p2", "apl_score": 0.6},
        ]

        rows = build_preference_intervention_rows(
            active_pool_rows=active_pool,
            logprob_rows=logprobs,
            score_rows=scores,
        )

        self.assertEqual(2, len(rows))
        by_id = {row["sample_id"]: row for row in rows}
        self.assertAlmostEqual(0.8, by_id["p1"]["base_margin"])
        self.assertEqual("a_longer", by_id["p1"]["length_gap_bin"])
        self.assertEqual("b_longer", by_id["p2"]["length_gap_bin"])
        self.assertEqual("human|model", by_id["p1"]["source_pair"])
        self.assertEqual("original", by_id["p1"]["ab_position"])
        self.assertEqual("c1", by_id["p1"]["prompt_cluster"])
        self.assertAlmostEqual(0.7, by_id["p1"]["apl_score"])
        self.assertAlmostEqual(0.8, by_id["p1"]["active_dpo_score"])
        self.assertAlmostEqual(0.6, by_id["p2"]["apl_score"])

    def test_length_gap_bin_uses_fixed_edges(self) -> None:
        self.assertEqual("b_longer", length_gap_bin(-0.21, edges=(-0.2, 0.2)))
        self.assertEqual("balanced", length_gap_bin(-0.2, edges=(-0.2, 0.2)))
        self.assertEqual("balanced", length_gap_bin(0.2, edges=(-0.2, 0.2)))
        self.assertEqual("a_longer", length_gap_bin(0.21, edges=(-0.2, 0.2)))

    def test_rejects_hidden_labels_and_missing_base_margin(self) -> None:
        with self.assertRaisesRegex(ValueError, "hidden fields"):
            build_preference_intervention_rows(
                active_pool_rows=[
                    {
                        "sample_id": "leaky",
                        "response_a": "A",
                        "response_b": "B",
                        "preference_label": "A",
                    }
                ]
            )

        with self.assertRaisesRegex(ValueError, "missing base margin inputs"):
            build_preference_intervention_rows(
                active_pool_rows=[
                    {
                        "sample_id": "missing",
                        "response_a": "A",
                        "response_b": "B",
                    }
                ]
            )


if __name__ == "__main__":
    unittest.main()
