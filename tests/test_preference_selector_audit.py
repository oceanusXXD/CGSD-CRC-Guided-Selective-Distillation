from __future__ import annotations

import unittest

from mias_dcms.preference_selector_audit import audit_preference_selector_scores


class PreferenceSelectorAuditTest(unittest.TestCase):
    def test_audit_reports_score_sanity_top_budget_and_swap_deltas(self) -> None:
        rows = [
            {
                "sample_id": "p1_original",
                "reward_margin_score": 0.90,
                "length_gap": 0.10,
                "swap_pair_id": "p1",
                "ab_position": "original",
            },
            {
                "sample_id": "p1_swapped",
                "reward_margin_score": 0.40,
                "length_gap": 0.10,
                "swap_pair_id": "p1",
                "ab_position": "swapped",
            },
            {
                "sample_id": "p2_original",
                "reward_margin_score": 0.80,
                "length_gap": 0.30,
                "swap_pair_id": "p2",
                "ab_position": "original",
            },
            {
                "sample_id": "p2_swapped",
                "reward_margin_score": 0.20,
                "length_gap": 0.30,
                "swap_pair_id": "p2",
                "ab_position": "swapped",
            },
        ]

        summary = audit_preference_selector_scores(
            rows,
            method="reward_margin",
            budget=2,
            selector_compute_seconds=1.25,
        )

        self.assertEqual("reward_margin", summary["method"])
        self.assertEqual("reward_margin_score", summary["score_field"])
        self.assertEqual(4, summary["pool_size"])
        self.assertEqual(2, summary["budget"])
        self.assertTrue(summary["score_not_all_equal"])
        self.assertGreater(summary["score_variance"], 0.0)
        self.assertEqual(["p1_original", "p2_original"], summary["selected_ids"])
        self.assertEqual(2, summary["selected_count"])
        self.assertTrue(summary["top_budget_reproducible"])
        self.assertFalse(summary["selected_ids_have_duplicates"])
        self.assertEqual(2, summary["expected_oracle_calls_after_reveal"])
        self.assertTrue(summary["oracle_calls_equal_budget"])
        self.assertAlmostEqual(1.25, summary["selector_compute_seconds"])
        self.assertLess(summary["score_length_correlation"], 0.0)
        self.assertEqual(2, summary["ab_swap_pair_count"])
        self.assertAlmostEqual(0.55, summary["ab_swap_mean_abs_score_delta"])
        self.assertAlmostEqual(0.60, summary["ab_swap_max_abs_score_delta"])

    def test_audit_rejects_hidden_preference_labels(self) -> None:
        rows = [
            {
                "sample_id": "leaky",
                "reward_margin_score": 0.5,
                "preference_label": "A",
            }
        ]

        with self.assertRaisesRegex(ValueError, "hidden fields"):
            audit_preference_selector_scores(rows, method="reward_margin", budget=1)

    def test_audit_rejects_degenerate_scores_by_default(self) -> None:
        rows = [
            {"sample_id": "p1", "apl_score": 0.5},
            {"sample_id": "p2", "apl_score": 0.5},
        ]

        with self.assertRaisesRegex(ValueError, "score is degenerate"):
            audit_preference_selector_scores(rows, method="apl", budget=1)


if __name__ == "__main__":
    unittest.main()
