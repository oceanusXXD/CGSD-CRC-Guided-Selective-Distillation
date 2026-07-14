from __future__ import annotations

import unittest

from mias_dcms.preference_run_summary import (
    build_preference_run_record,
    estimate_preference_train_tokens,
)


class PreferenceRunSummaryTest(unittest.TestCase):
    def test_estimates_train_tokens_from_prompt_and_response_text_when_missing(self) -> None:
        rows = [
            {
                "prompt": "short prompt",
                "response_1": "chosen answer",
                "response_2": "rejected answer here",
            },
            {
                "prompt": "another prompt",
                "response_1": "alpha",
                "response_2": "beta gamma",
                "train_tokens": 99,
            },
        ]

        self.assertEqual(110, estimate_preference_train_tokens(rows))

    def test_builds_run_record_with_required_cost_metrics_from_selection_and_reveal(self) -> None:
        run = build_preference_run_record(
            dataset="helpsteer2_preference",
            model="qwen-policy",
            method="APL",
            budget=2,
            seed=7,
            config_hash="cfg-pref",
            selection_summary={
                "selected_count": 2,
                "pool_size": 10,
                "selected_score_min": 0.7,
                "selected_score_max": 0.9,
                "continuous_moments": {"prompt_cluster=c0": 0.4},
                "rounded_moments": {"prompt_cluster=c0": 0.5},
                "robust_lower_moments": {"prompt_cluster=c0": 0.3},
                "robust_upper_moments": {"prompt_cluster=c0": 0.6},
                "utility_retained": 0.96,
                "max_constraint_violation": 0.04,
                "solver_status": "scalable_slsqp",
                "selected_slack": 0.2,
                "rounding_seed": 7,
                "selection_metrics": {"acquisition_tv": 0.12},
            },
            reveal_summary={
                "revealed_count": 2,
                "dpo_train_row_count": 1,
                "unrevealed_count": 8,
            },
            training_rows=[
                {
                    "prompt": "prompt text",
                    "response_1": "chosen response",
                    "response_2": "rejected response",
                }
            ],
            training_metrics={"dpo_train_row_count": 1},
            evaluation_metrics={"preference_accuracy": 0.61, "worst_group_accuracy": 0.5},
            seed_label_count=4,
            evaluation_label_count=6,
            judge_calls=3,
            selector_compute_seconds=1.5,
        )

        payload = run.as_dict()
        self.assertEqual("helpsteer2_preference", payload["dataset"])
        self.assertEqual("APL", payload["method"])
        self.assertEqual(2, payload["budget"])
        self.assertEqual(2, payload["selected_count"])
        self.assertEqual(0.12, payload["selection_metrics"]["acquisition_tv"])
        self.assertEqual(0.7, payload["selection_metrics"]["selected_score_min"])
        self.assertEqual({"prompt_cluster=c0": 0.4}, payload["continuous_moments"])
        self.assertEqual({"prompt_cluster=c0": 0.5}, payload["rounded_moments"])
        self.assertEqual(0.96, payload["utility_retained"])
        self.assertEqual(0.04, payload["max_constraint_violation"])
        self.assertEqual("scalable_slsqp", payload["solver_status"])
        self.assertEqual(0.2, payload["selected_slack"])
        self.assertEqual(7, payload["rounding_seed"])
        self.assertEqual(1, payload["training_metrics"]["dpo_train_row_count"])
        self.assertEqual(4, payload["cost_metrics"]["seed_label_count"])
        self.assertEqual(2, payload["cost_metrics"]["active_label_count"])
        self.assertEqual(6, payload["cost_metrics"]["evaluation_label_count"])
        self.assertEqual(3, payload["cost_metrics"]["judge_calls"])
        self.assertEqual(10, payload["cost_metrics"]["train_tokens"])
        self.assertEqual(1.5, payload["cost_metrics"]["selector_compute_seconds"])
        self.assertEqual(2, payload["cost_metrics"]["oracle_label_calls"])

    def test_rejects_mismatched_budget_and_revealed_count(self) -> None:
        with self.assertRaisesRegex(ValueError, "revealed_count"):
            build_preference_run_record(
                dataset="toy",
                model="model",
                method="Random",
                budget=2,
                seed=1,
                config_hash="cfg",
                selection_summary={"selected_count": 2},
                reveal_summary={"revealed_count": 1, "dpo_train_row_count": 1},
                training_rows=[],
                evaluation_metrics={"preference_accuracy": 0.5},
            )


if __name__ == "__main__":
    unittest.main()
