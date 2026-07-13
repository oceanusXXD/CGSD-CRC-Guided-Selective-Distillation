from __future__ import annotations

import unittest

from mias_dcms.preference_evaluation import (
    area_under_learning_curve,
    build_preference_evaluation_metrics,
    capability_regression,
    length_controlled_win_rate,
    preference_accuracy,
    worst_group_preference_accuracy,
)


class PreferenceEvaluationTest(unittest.TestCase):
    def test_preference_accuracy_and_worst_group_accuracy(self) -> None:
        rows = [
            {"sample_id": "p1", "oracle_preference": "A", "predicted_preference": "A", "length_gap_bin": "short"},
            {"sample_id": "p2", "oracle_preference": "B", "predicted_preference": "A", "length_gap_bin": "short"},
            {"sample_id": "p3", "oracle_preference": "B", "predicted_preference": "B", "length_gap_bin": "long"},
            {"sample_id": "p4", "oracle_preference": "A", "predicted_preference": "A", "length_gap_bin": "long"},
        ]

        self.assertAlmostEqual(0.75, preference_accuracy(rows))
        self.assertAlmostEqual(0.50, worst_group_preference_accuracy(rows, group_field="length_gap_bin"))

    def test_length_controlled_win_rate_averages_within_length_bins(self) -> None:
        rows = [
            {"sample_id": "g1", "judge_win": 1.0, "length_gap_bin": "short"},
            {"sample_id": "g2", "judge_win": 0.0, "length_gap_bin": "short"},
            {"sample_id": "g3", "judge_win": 1.0, "length_gap_bin": "long"},
        ]

        self.assertAlmostEqual(2.0 / 3.0, sum(row["judge_win"] for row in rows) / len(rows))
        self.assertAlmostEqual(0.75, length_controlled_win_rate(rows, length_bin_field="length_gap_bin"))

    def test_capability_regression_is_baseline_minus_policy_score(self) -> None:
        rows = [
            {"task_id": "c1", "baseline_score": 0.80, "policy_score": 0.72},
            {"task_id": "c2", "baseline_score": 0.60, "policy_score": 0.57},
        ]

        self.assertAlmostEqual(0.055, capability_regression(rows))

    def test_build_preference_evaluation_metrics_combines_available_sections(self) -> None:
        preference_rows = [
            {"sample_id": "p1", "oracle_preference": "A", "predicted_preference": "A", "source_pair": "human|model"},
            {"sample_id": "p2", "oracle_preference": "B", "predicted_preference": "A", "source_pair": "human|model"},
            {"sample_id": "p3", "oracle_preference": "B", "predicted_preference": "B", "source_pair": "model|human"},
        ]
        judge_rows = [
            {"sample_id": "g1", "judge_win": 1.0, "length_gap_bin": "short"},
            {"sample_id": "g2", "judge_win": 0.0, "length_gap_bin": "short"},
            {"sample_id": "g3", "judge_win": 1.0, "length_gap_bin": "long"},
        ]
        capability_rows = [
            {"task_id": "c1", "baseline_score": 0.90, "policy_score": 0.85},
            {"task_id": "c2", "baseline_score": 0.70, "policy_score": 0.65},
        ]

        metrics = build_preference_evaluation_metrics(
            preference_rows=preference_rows,
            judge_rows=judge_rows,
            capability_rows=capability_rows,
            group_field="source_pair",
            length_bin_field="length_gap_bin",
        )

        self.assertAlmostEqual(2.0 / 3.0, metrics["preference_accuracy"])
        self.assertAlmostEqual(0.5, metrics["worst_group_preference_accuracy"])
        self.assertAlmostEqual(2.0 / 3.0, metrics["raw_judge_win_rate"])
        self.assertAlmostEqual(0.75, metrics["length_controlled_win_rate"])
        self.assertAlmostEqual(0.05, metrics["capability_regression"])
        self.assertEqual(3, metrics["preference_eval_count"])
        self.assertEqual(3, metrics["judge_eval_count"])
        self.assertEqual(2, metrics["capability_eval_count"])

    def test_rejects_empty_metric_inputs_when_metric_is_requested(self) -> None:
        with self.assertRaisesRegex(ValueError, "preference rows must not be empty"):
            preference_accuracy([])

        with self.assertRaisesRegex(ValueError, "judge rows must not be empty"):
            length_controlled_win_rate([])

        with self.assertRaisesRegex(ValueError, "capability rows must not be empty"):
            capability_regression([])

    def test_area_under_learning_curve_is_normalized(self) -> None:
        rows = [
            {"budget": 0, "performance": 0.5},
            {"budget": 10, "performance": 0.7},
            {"budget": 20, "performance": 0.9},
        ]

        self.assertAlmostEqual(0.7, area_under_learning_curve(rows))

    def test_area_under_learning_curve_keeps_best_duplicate_budget(self) -> None:
        rows = [
            {"budget": 0, "performance": 0.5},
            {"budget": 10, "performance": 0.6},
            {"budget": 10, "performance": 0.8},
            {"budget": 20, "performance": 0.9},
        ]

        self.assertAlmostEqual(0.75, area_under_learning_curve(rows))


if __name__ == "__main__":
    unittest.main()
