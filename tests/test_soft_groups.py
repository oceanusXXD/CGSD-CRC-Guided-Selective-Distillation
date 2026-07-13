from __future__ import annotations

import unittest

from mias_dcms.soft_groups import (
    build_soft_group_intervals,
    build_soft_group_intervals_from_rows,
    interval_coverage_report,
    soft_group_calibration_report,
)


class SoftGroupIntervalTest(unittest.TestCase):
    def test_build_soft_group_intervals_from_ensemble_draws(self) -> None:
        report = build_soft_group_intervals(
            sample_ids=["s1", "s2"],
            membership_draws=[
                [
                    {"A": 0.2, "B": 0.8},
                    {"A": 0.4, "B": 0.6},
                    {"A": 0.6, "B": 0.4},
                ],
                [
                    {"A": 0.9, "B": 0.1},
                    {"A": 0.7, "B": 0.3},
                    {"A": 0.8, "B": 0.2},
                ],
            ],
            confidence=1.0,
        )

        by_id = {row.sample_id: row for row in report.rows}

        self.assertEqual(["A", "B"], report.groups)
        self.assertEqual(2, report.sample_count)
        self.assertEqual(3, by_id["s1"].draw_count)
        self.assertAlmostEqual(0.4, by_id["s1"].group_membership["A"])
        self.assertAlmostEqual(0.2, by_id["s1"].membership_lower["A"])
        self.assertAlmostEqual(0.6, by_id["s1"].membership_upper["A"])

    def test_build_soft_group_intervals_rejects_hidden_labels_in_pool_rows(self) -> None:
        with self.assertRaises(ValueError):
            build_soft_group_intervals_from_rows(
                [
                    {
                        "sample_id": "leaky",
                        "oracle_label": "A",
                        "ensemble_memberships": [{"A": 0.9}, {"A": 0.8}],
                    }
                ],
                confidence=1.0,
            )

    def test_build_soft_group_intervals_requires_consistent_unique_ids(self) -> None:
        with self.assertRaises(ValueError):
            build_soft_group_intervals(
                sample_ids=["s1", "s1"],
                membership_draws=[[{"A": 1.0}], [{"A": 0.0}]],
                confidence=1.0,
            )

    def test_soft_group_calibration_report_summarizes_group_errors(self) -> None:
        report = soft_group_calibration_report(
            predicted_memberships=[
                {"A": 0.8, "B": 0.2},
                {"A": 0.3, "B": 0.7},
            ],
            observed_memberships=[
                {"A": 1.0, "B": 0.0},
                {"A": 0.0, "B": 1.0},
            ],
        )

        self.assertEqual(["A", "B"], report.groups)
        self.assertEqual(2, report.sample_count)
        self.assertAlmostEqual(0.065, report.per_group["A"]["brier_score"])
        self.assertAlmostEqual(0.55, report.per_group["A"]["mean_predicted"])
        self.assertAlmostEqual(0.5, report.per_group["A"]["mean_observed"])
        self.assertAlmostEqual(0.065, report.overall_brier_score)

    def test_interval_coverage_report_checks_observed_values_against_bounds(self) -> None:
        interval_report = build_soft_group_intervals(
            sample_ids=["s1", "s2"],
            membership_draws=[
                [{"A": 0.2, "B": 0.8}, {"A": 0.4, "B": 0.6}, {"A": 0.6, "B": 0.4}],
                [{"A": 0.9, "B": 0.1}, {"A": 0.7, "B": 0.3}, {"A": 0.8, "B": 0.2}],
            ],
            confidence=1.0,
        )

        coverage = interval_coverage_report(
            interval_report.rows,
            observed_memberships=[
                {"A": 0.4, "B": 0.6},
                {"A": 1.0, "B": 0.2},
            ],
        )

        self.assertEqual(2, coverage.sample_count)
        self.assertAlmostEqual(0.75, coverage.overall_coverage_rate)
        self.assertAlmostEqual(0.5, coverage.per_group["A"]["coverage_rate"])
        self.assertAlmostEqual(1.0, coverage.per_group["B"]["coverage_rate"])


if __name__ == "__main__":
    unittest.main()
