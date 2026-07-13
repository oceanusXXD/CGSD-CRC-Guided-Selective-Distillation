from __future__ import annotations

import unittest

from mias_dcms.dpo_run_pack import (
    DPO_MAIN_METHODS,
    validate_dpo_run_pack,
    validate_paper_artifact_manifest,
)


class DPORunPackValidationTest(unittest.TestCase):
    def test_complete_dpo_run_pack_is_ready_for_aggregation(self) -> None:
        rows = [
            _run_row(method=method, seed=seed)
            for seed in (1, 2)
            for method in DPO_MAIN_METHODS
        ]

        report = validate_dpo_run_pack(
            rows,
            expected_datasets=["helpsteer2"],
            expected_models=["qwen"],
            expected_budgets=[100],
            expected_seeds=[1, 2],
        )

        self.assertTrue(report.is_ready)
        self.assertEqual(12, report.expected_run_count)
        self.assertEqual(12, report.completed_run_count)
        self.assertEqual([], report.issues)
        self.assertEqual(list(DPO_MAIN_METHODS), report.covered_methods)

    def test_missing_method_seed_combination_is_reported(self) -> None:
        rows = [
            _run_row(method=method, seed=seed)
            for seed in (1, 2)
            for method in DPO_MAIN_METHODS
            if not (method == "ActiveDPO+DCMS" and seed == 2)
        ]

        report = validate_dpo_run_pack(
            rows,
            expected_datasets=["helpsteer2"],
            expected_models=["qwen"],
            expected_budgets=[100],
            expected_seeds=[1, 2],
        )

        self.assertFalse(report.is_ready)
        self.assertEqual(1, report.missing_run_count)
        self.assertIn("missing_run", {issue["code"] for issue in report.issues})
        self.assertIn(
            "helpsteer2|qwen|100|2|ActiveDPO+DCMS",
            {issue["run_key"] for issue in report.issues},
        )

    def test_failed_runs_are_visible_and_require_reasons(self) -> None:
        rows = [_run_row(method=method, seed=1) for method in DPO_MAIN_METHODS]
        rows[0]["run_status"] = "failed"
        rows[0]["failure_reason"] = "checkpoint diverged"
        rows[1]["run_status"] = "failed"

        report = validate_dpo_run_pack(
            rows,
            expected_datasets=["helpsteer2"],
            expected_models=["qwen"],
            expected_budgets=[100],
            expected_seeds=[1],
        )

        self.assertFalse(report.is_ready)
        self.assertEqual(2, report.failed_run_count)
        issue_codes = {issue["code"] for issue in report.issues}
        self.assertIn("failed_run", issue_codes)
        self.assertIn("failed_run_missing_reason", issue_codes)

    def test_required_metrics_are_checked_for_completed_runs(self) -> None:
        row = _run_row(method="Random", seed=1)
        del row["evaluation_metrics"]["capability_regression"]

        report = validate_dpo_run_pack(
            [row],
            expected_datasets=["helpsteer2"],
            expected_models=["qwen"],
            expected_budgets=[100],
            expected_seeds=[1],
            required_methods=["Random"],
        )

        self.assertFalse(report.is_ready)
        self.assertIn("missing_metric", {issue["code"] for issue in report.issues})
        self.assertIn(
            "evaluation_metrics.capability_regression",
            {issue["metric"] for issue in report.issues if "metric" in issue},
        )

    def test_paper_artifact_manifest_requires_traceable_figures_and_tables(self) -> None:
        manifest = {
            "results_manifest": {"run_records_path": "experiments/runs/dpo.jsonl"},
            "figures": {
                "fig1": _artifact(seed_count=2),
                "fig2": _artifact(seed_count=2),
                "fig3": {
                    "input_result_files": ["experiments/results/fig3.csv"],
                    "aggregation_rule": "matched utility bins",
                    "seed_count": 2,
                    "includes_failed_runs": True,
                },
            },
            "tables": {
                "table1": _artifact(seed_count=2),
                "table2": _artifact(seed_count=2),
            },
        }

        issues = validate_paper_artifact_manifest(
            manifest,
            expected_figures=["fig1", "fig2", "fig3"],
            expected_tables=["table1", "table2", "table3"],
            expected_seed_count=2,
        )

        issue_codes = {issue["code"] for issue in issues}
        self.assertIn("artifact_missing_error_bar", issue_codes)
        self.assertIn("missing_artifact", issue_codes)
        self.assertIn("table3", {issue.get("artifact") for issue in issues})


def _run_row(*, method: str, seed: int) -> dict[str, object]:
    return {
        "dataset": "helpsteer2",
        "model": "qwen",
        "method": method,
        "budget": 100,
        "seed": seed,
        "selected_count": 100,
        "config_hash": "frozen-config",
        "run_status": "completed",
        "selection_metrics": {
            "acquisition_tv": 0.12,
            "utility_retained": 0.97,
            "max_constraint_violation": 0.0,
        },
        "training_metrics": {
            "dpo_train_row_count": 100,
            "update_steps": 20,
            "training_token_budget": 4096,
        },
        "evaluation_metrics": {
            "preference_accuracy": 0.61,
            "worst_group_preference_accuracy": 0.54,
            "length_controlled_win_rate": 0.58,
            "capability_regression": -0.01,
            "aulc": 0.59,
        },
        "cost_metrics": {
            "seed_label_count": 25,
            "active_label_count": 100,
            "evaluation_label_count": 500,
            "judge_calls": 500,
            "train_tokens": 12000,
            "selector_compute_seconds": 3.5,
            "oracle_label_calls": 100,
        },
    }


def _artifact(*, seed_count: int) -> dict[str, object]:
    return {
        "input_result_files": ["experiments/results/source.jsonl"],
        "aggregation_rule": "mean with bootstrap CI",
        "seed_count": seed_count,
        "error_bar": "bootstrap 95% CI",
        "includes_failed_runs": True,
    }


if __name__ == "__main__":
    unittest.main()
