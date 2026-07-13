from __future__ import annotations

import unittest

from mias_dcms.statistics import (
    MetricSummary,
    bootstrap_mean_ci,
    paired_mean_delta,
    paired_permutation_test,
    summarize_metric_by_method,
)


class StatisticsTest(unittest.TestCase):
    def test_bootstrap_mean_ci_is_deterministic_and_contains_mean(self) -> None:
        summary = bootstrap_mean_ci([1.0, 2.0, 3.0, 4.0], confidence=0.95, resamples=500, seed=7)

        self.assertIsInstance(summary, MetricSummary)
        self.assertEqual(2.5, summary.mean)
        self.assertLessEqual(summary.ci_low, summary.mean)
        self.assertGreaterEqual(summary.ci_high, summary.mean)
        self.assertEqual(
            summary,
            bootstrap_mean_ci([1.0, 2.0, 3.0, 4.0], confidence=0.95, resamples=500, seed=7),
        )

    def test_paired_mean_delta_aligns_on_seed_and_method(self) -> None:
        rows = [
            {"seed": 1, "method": "Random", "macro_f1": 0.50},
            {"seed": 1, "method": "DCMS", "macro_f1": 0.60},
            {"seed": 2, "method": "Random", "macro_f1": 0.40},
            {"seed": 2, "method": "DCMS", "macro_f1": 0.55},
        ]

        delta = paired_mean_delta(
            rows,
            baseline_method="Random",
            treatment_method="DCMS",
            metric_field="macro_f1",
        )

        self.assertAlmostEqual(0.125, delta)

    def test_paired_permutation_test_reports_two_sided_p_value(self) -> None:
        result = paired_permutation_test(
            baseline=[0.5, 0.4, 0.45],
            treatment=[0.8, 0.7, 0.75],
            permutations=512,
            seed=3,
        )

        self.assertAlmostEqual(0.3, result.observed_delta)
        self.assertGreaterEqual(result.p_value, 0.0)
        self.assertLessEqual(result.p_value, 1.0)
        self.assertEqual(3, result.paired_count)

    def test_summarize_metric_by_method_groups_values_and_ci(self) -> None:
        rows = [
            {"method": "Random", "seed": 1, "acquisition_tv": 0.20},
            {"method": "Random", "seed": 2, "acquisition_tv": 0.30},
            {"method": "DCMS", "seed": 1, "acquisition_tv": 0.05},
            {"method": "DCMS", "seed": 2, "acquisition_tv": 0.10},
        ]

        summary = summarize_metric_by_method(
            rows,
            metric_field="acquisition_tv",
            method_field="method",
            resamples=200,
            seed=5,
        )

        self.assertEqual({"DCMS", "Random"}, set(summary))
        self.assertAlmostEqual(0.25, summary["Random"].mean)
        self.assertAlmostEqual(0.075, summary["DCMS"].mean)


if __name__ == "__main__":
    unittest.main()
