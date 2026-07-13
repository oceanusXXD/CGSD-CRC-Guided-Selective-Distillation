from __future__ import annotations

import unittest

from mias_dcms.intervention_statistics import audit_intervention_response_statistics


class InterventionStatisticsTest(unittest.TestCase):
    def test_complete_response_curve_reports_monotonicity_and_slope_ci(self) -> None:
        report = audit_intervention_response_statistics(
            _curve_rows(
                setting="helpsteer2_qwen",
                values=[-2, -1, 0, 1, 2],
                responses=[0.10, 0.20, 0.50, 0.70, 0.90],
            ),
            expected_settings=["helpsteer2_qwen"],
            minimum_values=5,
            resamples=200,
            seed=11,
        )

        self.assertTrue(report.is_ready)
        self.assertEqual([], report.issues)
        setting = report.by_setting["helpsteer2_qwen"]
        self.assertEqual(5, setting["intervention_value_count"])
        self.assertAlmostEqual(1.0, setting["spearman_monotonicity"])
        self.assertGreater(setting["slope"], 0.0)
        self.assertIn("slope_ci_low", setting)
        self.assertIn("slope_ci_high", setting)

    def test_insufficient_intervention_values_are_rejected(self) -> None:
        report = audit_intervention_response_statistics(
            _curve_rows(
                setting="helpsteer2_qwen",
                values=[-1, 0, 1],
                responses=[0.20, 0.50, 0.70],
            ),
            expected_settings=["helpsteer2_qwen"],
            minimum_values=5,
        )

        self.assertFalse(report.is_ready)
        self.assertIn("insufficient_intervention_values", {issue["code"] for issue in report.issues})

    def test_expected_settings_cannot_be_silently_hidden(self) -> None:
        report = audit_intervention_response_statistics(
            _curve_rows(
                setting="helpsteer2_qwen",
                values=[-2, -1, 0, 1, 2],
                responses=[0.10, 0.20, 0.50, 0.70, 0.90],
            ),
            expected_settings=["helpsteer2_qwen", "tldr_llama"],
            minimum_values=5,
        )

        self.assertFalse(report.is_ready)
        self.assertIn("missing_expected_setting", {issue["code"] for issue in report.issues})

    def test_failed_setting_requires_reason_and_is_counted_separately(self) -> None:
        rows = [
            *_curve_rows(
                setting="helpsteer2_qwen",
                values=[-2, -1, 0, 1, 2],
                responses=[0.10, 0.20, 0.50, 0.70, 0.90],
            ),
            {
                "setting": "tldr_llama",
                "status": "failed",
                "intervention_value": 0.0,
                "target_group_propensity": None,
            },
        ]

        report = audit_intervention_response_statistics(
            rows,
            expected_settings=["helpsteer2_qwen", "tldr_llama"],
            minimum_values=5,
        )

        self.assertFalse(report.is_ready)
        self.assertEqual(1, report.failed_setting_count)
        self.assertIn("failed_setting_missing_reason", {issue["code"] for issue in report.issues})


def _curve_rows(*, setting: str, values: list[float], responses: list[float]) -> list[dict[str, object]]:
    return [
        {
            "setting": setting,
            "status": "completed",
            "intervention_value": value,
            "target_group_propensity": response,
        }
        for value, response in zip(values, responses, strict=True)
    ]


if __name__ == "__main__":
    unittest.main()
