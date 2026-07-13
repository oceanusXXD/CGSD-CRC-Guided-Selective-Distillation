from __future__ import annotations

import unittest

from mias_dcms.budgeting import (
    BudgetInputs,
    build_budget_report,
    compare_budget_reports,
)


class BudgetingTest(unittest.TestCase):
    def test_budget_report_keeps_supervision_and_evaluation_costs_separate(self) -> None:
        report = build_budget_report(
            BudgetInputs(
                method="Entropy+DCMS",
                seed_label_count=8,
                active_label_count=12,
                guide_label_count=3,
                calibration_label_count=2,
                group_estimator_label_count=5,
                evaluation_label_count=100,
                certification_label_count=20,
                judge_calls=40,
                train_tokens=4096,
                selector_compute_seconds=1.5,
            )
        )

        as_dict = report.as_dict()

        self.assertEqual(30, as_dict["supervision_budget_total"])
        self.assertEqual(120, as_dict["evaluation_resource_total"])
        self.assertEqual(40, as_dict["judge_calls"])
        self.assertEqual(4096, as_dict["train_tokens"])
        self.assertEqual(1.5, as_dict["selector_compute_seconds"])

    def test_budget_report_rejects_negative_costs(self) -> None:
        with self.assertRaises(ValueError):
            build_budget_report(
                BudgetInputs(
                    method="bad",
                    seed_label_count=-1,
                    active_label_count=0,
                )
            )

    def test_compare_budget_reports_flags_unfair_supervision_and_token_differences(self) -> None:
        random_report = build_budget_report(
            BudgetInputs(
                method="Random",
                seed_label_count=8,
                active_label_count=12,
                train_tokens=1000,
            )
        )
        active_report = build_budget_report(
            BudgetInputs(
                method="Entropy",
                seed_label_count=8,
                active_label_count=12,
                guide_label_count=5,
                train_tokens=1250,
            )
        )

        comparison = compare_budget_reports(
            [random_report, active_report],
            train_token_tolerance=100,
        )

        self.assertFalse(comparison["supervision_budget_equal"])
        self.assertFalse(comparison["train_tokens_within_tolerance"])
        self.assertEqual({"Entropy": 25, "Random": 20}, comparison["supervision_budget_by_method"])


if __name__ == "__main__":
    unittest.main()
