from __future__ import annotations

import unittest

from mias_dcms.composition import (
    coverage_deviation,
    matched_utility_report,
    utility_quantile_profile,
)


class CompositionTest(unittest.TestCase):
    def test_utility_quantile_profile_sorts_and_splits_values(self) -> None:
        profile = utility_quantile_profile([0.1, 0.9, 0.4, 0.8], bins=2)

        self.assertEqual(0.55, profile.mean)
        self.assertEqual([0.25, 0.75], profile.quantile_midpoints)
        self.assertEqual(0.25, profile.bin_means[0])
        self.assertAlmostEqual(0.85, profile.bin_means[1])

    def test_coverage_deviation_computes_total_variation_from_target(self) -> None:
        deviation = coverage_deviation(
            observed_moments={"A": 0.75, "B": 0.25},
            target_moments={"A": 0.5, "B": 0.5},
        )

        self.assertEqual(0.25, deviation)

    def test_matched_utility_report_flags_utility_match_and_coverage_difference(self) -> None:
        report = matched_utility_report(
            baseline_utilities=[0.9, 0.7, 0.3, 0.1],
            treatment_utilities=[0.88, 0.72, 0.31, 0.09],
            baseline_moments={"A": 0.5, "B": 0.5},
            treatment_moments={"A": 0.8, "B": 0.2},
            target_moments={"A": 0.5, "B": 0.5},
            mean_tolerance=0.03,
            quantile_tolerance=0.03,
        )

        self.assertTrue(report.utility_matched)
        self.assertGreater(report.treatment_coverage_deviation, report.baseline_coverage_deviation)
        self.assertAlmostEqual(0.0, report.mean_delta)
        self.assertLessEqual(report.max_quantile_delta, 0.03)
        self.assertEqual(4, report.baseline_count)
        self.assertEqual(4, report.treatment_count)

    def test_matched_utility_report_rejects_different_batch_sizes(self) -> None:
        with self.assertRaises(ValueError):
            matched_utility_report(
                baseline_utilities=[0.9, 0.1],
                treatment_utilities=[0.9],
                baseline_moments={"A": 0.5},
                treatment_moments={"A": 0.5},
                target_moments={"A": 0.5},
            )


if __name__ == "__main__":
    unittest.main()
