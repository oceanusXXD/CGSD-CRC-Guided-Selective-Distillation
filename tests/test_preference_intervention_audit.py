from __future__ import annotations

import unittest

from mias_dcms.preference_intervention_audit import (
    audit_ab_position_intervention,
    audit_length_gamma_intervention,
    audit_selector_replacement,
)


class PreferenceInterventionAuditTest(unittest.TestCase):
    def test_length_gamma_intervention_reports_response_and_linked_group_coverage(self) -> None:
        rows = [
            {"sample_id": "short_a", "base_margin": 0.20, "length_gap": -0.50, "length_gap_bin": "short", "source_pair": "human|model", "prompt_cluster": "c1"},
            {"sample_id": "long_a", "base_margin": 0.20, "length_gap": 0.50, "length_gap_bin": "long", "source_pair": "model|human", "prompt_cluster": "c2"},
            {"sample_id": "short_b", "base_margin": 0.10, "length_gap": -0.40, "length_gap_bin": "short", "source_pair": "human|model", "prompt_cluster": "c1"},
            {"sample_id": "long_b", "base_margin": 0.10, "length_gap": 0.40, "length_gap_bin": "long", "source_pair": "model|human", "prompt_cluster": "c2"},
        ]

        summary = audit_length_gamma_intervention(
            rows,
            gammas=(-1.0, 0.0, 1.0),
            budget=2,
            target_length_bin="long",
            linked_group_fields=("source_pair", "prompt_cluster"),
        )

        self.assertEqual([-1.0, 0.0, 1.0], summary["gammas"])
        self.assertTrue(summary["gamma_grid_has_negative_zero_positive"])
        self.assertTrue(summary["gamma_zero_matches_base_score"])
        by_gamma = {point["gamma"]: point for point in summary["points"]}
        self.assertEqual(["short_a", "short_b"], by_gamma[-1.0]["selected_ids"])
        self.assertEqual(["long_a", "short_a"], by_gamma[0.0]["selected_ids"])
        self.assertEqual(["long_a", "long_b"], by_gamma[1.0]["selected_ids"])
        self.assertAlmostEqual(0.0, by_gamma[-1.0]["target_length_bin_propensity"])
        self.assertAlmostEqual(0.5, by_gamma[0.0]["target_length_bin_propensity"])
        self.assertAlmostEqual(1.0, by_gamma[1.0]["target_length_bin_propensity"])
        self.assertGreater(summary["target_propensity_slope"], 0.0)
        self.assertIn("source_pair", by_gamma[1.0]["linked_group_distribution"])
        self.assertIn("prompt_cluster", by_gamma[1.0]["linked_group_distribution"])

    def test_selector_replacement_reports_rank_overlap_and_attribute_coverage_delta(self) -> None:
        rows = [
            {"sample_id": "p1", "selector_a_score": 0.9, "selector_b_score": 0.1, "length_gap_bin": "short"},
            {"sample_id": "p2", "selector_a_score": 0.8, "selector_b_score": 0.2, "length_gap_bin": "short"},
            {"sample_id": "p3", "selector_a_score": 0.2, "selector_b_score": 0.8, "length_gap_bin": "long"},
            {"sample_id": "p4", "selector_a_score": 0.1, "selector_b_score": 0.9, "length_gap_bin": "long"},
        ]

        summary = audit_selector_replacement(
            rows,
            selector_a_score_field="selector_a_score",
            selector_b_score_field="selector_b_score",
            budget=2,
            group_fields=("length_gap_bin",),
        )

        self.assertEqual(["p1", "p2"], summary["selector_a_selected_ids"])
        self.assertEqual(["p4", "p3"], summary["selector_b_selected_ids"])
        self.assertAlmostEqual(-1.0, summary["score_rank_correlation"])
        self.assertAlmostEqual(0.0, summary["selected_set_overlap"])
        self.assertAlmostEqual(1.0, summary["attribute_coverage_delta"]["length_gap_bin"]["selected_distribution_tv"])
        self.assertIn("length_gap_bin", summary["selector_a_group_propensities"])
        self.assertIn("length_gap_bin", summary["selector_b_group_propensities"])

    def test_ab_position_intervention_reports_score_rank_overlap_and_position_propensity(self) -> None:
        rows = [
            {"sample_id": "pair1_original", "swap_pair_id": "pair1", "ab_position": "original", "score": 0.9},
            {"sample_id": "pair1_swapped", "swap_pair_id": "pair1", "ab_position": "swapped", "score": 0.2},
            {"sample_id": "pair2_original", "swap_pair_id": "pair2", "ab_position": "original", "score": 0.8},
            {"sample_id": "pair2_swapped", "swap_pair_id": "pair2", "ab_position": "swapped", "score": 0.1},
        ]

        summary = audit_ab_position_intervention(rows, score_field="score", budget=2)

        self.assertEqual(2, summary["pair_count"])
        self.assertAlmostEqual(1.0, summary["original_swapped_rank_correlation"])
        self.assertEqual(["pair1_original", "pair2_original"], summary["selected_ids"])
        self.assertAlmostEqual(1.0, summary["position_propensity"]["original"])
        self.assertAlmostEqual(0.0, summary["position_propensity"]["swapped"])
        self.assertAlmostEqual(0.5, summary["position_acquisition_tv"])

    def test_rejects_hidden_preference_labels(self) -> None:
        rows = [{"sample_id": "p1", "base_margin": 0.1, "length_gap": 0.2, "length_gap_bin": "long", "preference_label": "A"}]

        with self.assertRaisesRegex(ValueError, "hidden fields"):
            audit_length_gamma_intervention(rows, gammas=(-1, 0, 1), budget=1, target_length_bin="long")


if __name__ == "__main__":
    unittest.main()
