from __future__ import annotations

import unittest

from mias_dcms.records import RunRecord
from mias_dcms.run_metric_comparison import compare_run_metrics_to_baseline


class RunMetricComparisonTest(unittest.TestCase):
    def test_compares_treatment_to_baseline_with_paired_seed_statistics(self) -> None:
        runs = [
            _run_record(method="Random", seed=1, preference_accuracy=0.60, acquisition_tv=0.20),
            _run_record(method="Random", seed=2, preference_accuracy=0.64, acquisition_tv=0.24),
            _run_record(method="APL+DCMS", seed=1, preference_accuracy=0.66, acquisition_tv=0.08),
            _run_record(method="APL+DCMS", seed=2, preference_accuracy=0.70, acquisition_tv=0.10),
        ]

        report = compare_run_metrics_to_baseline(
            runs,
            baseline_method="Random",
            treatment_methods=["APL+DCMS"],
            evaluation_metrics=["preference_accuracy"],
            selection_metrics=["acquisition_tv"],
            confidence=0.95,
            resamples=200,
            permutations=200,
            seed=7,
        )

        self.assertEqual([], report.issues)
        self.assertEqual(2, len(report.comparisons))
        by_metric = {row["metric"]: row for row in report.comparisons}
        self.assertAlmostEqual(0.06, by_metric["preference_accuracy"]["delta_mean"])
        self.assertAlmostEqual(-0.13, by_metric["acquisition_tv"]["delta_mean"])
        self.assertEqual([1, 2], by_metric["preference_accuracy"]["paired_seeds"])
        self.assertIn("p_value", by_metric["preference_accuracy"])
        self.assertIn("delta_ci_low", by_metric["preference_accuracy"])

    def test_reports_missing_paired_seed_without_dropping_it_silently(self) -> None:
        runs = [
            _run_record(method="Random", seed=1, preference_accuracy=0.60, acquisition_tv=0.20),
            _run_record(method="Random", seed=2, preference_accuracy=0.64, acquisition_tv=0.24),
            _run_record(method="APL+DCMS", seed=1, preference_accuracy=0.66, acquisition_tv=0.08),
        ]

        report = compare_run_metrics_to_baseline(
            runs,
            baseline_method="Random",
            treatment_methods=["APL+DCMS"],
            evaluation_metrics=["preference_accuracy"],
            expected_seeds=[1, 2],
            minimum_paired_seeds=2,
        )

        self.assertFalse(report.is_ready)
        self.assertIn("missing_treatment_seed", {issue["code"] for issue in report.issues})
        self.assertIn("insufficient_paired_seeds", {issue["code"] for issue in report.issues})

    def test_rejects_missing_required_metric_for_completed_comparison(self) -> None:
        bad = _run_record(method="APL+DCMS", seed=1, preference_accuracy=0.66, acquisition_tv=0.08)
        del bad.evaluation_metrics["preference_accuracy"]

        report = compare_run_metrics_to_baseline(
            [
                _run_record(method="Random", seed=1, preference_accuracy=0.60, acquisition_tv=0.20),
                bad,
            ],
            baseline_method="Random",
            treatment_methods=["APL+DCMS"],
            evaluation_metrics=["preference_accuracy"],
        )

        self.assertFalse(report.is_ready)
        self.assertIn("missing_metric", {issue["code"] for issue in report.issues})


def _run_record(
    *,
    method: str,
    seed: int,
    preference_accuracy: float,
    acquisition_tv: float,
) -> RunRecord:
    return RunRecord(
        dataset="helpsteer2",
        model="qwen",
        method=method,
        budget=100,
        seed=seed,
        selected_count=100,
        config_hash=f"cfg-{method}-{seed}",
        selection_metrics={"acquisition_tv": acquisition_tv},
        training_metrics={},
        evaluation_metrics={"preference_accuracy": preference_accuracy},
        cost_metrics={
            "seed_label_count": 25,
            "active_label_count": 100,
            "evaluation_label_count": 500,
            "judge_calls": 500,
            "train_tokens": 12000,
            "selector_compute_seconds": 3.5,
        },
    )


if __name__ == "__main__":
    unittest.main()
