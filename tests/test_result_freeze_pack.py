from __future__ import annotations

import unittest

from mias_dcms.result_freeze_pack import validate_result_freeze_pack


class ResultFreezePackTest(unittest.TestCase):
    def test_complete_freeze_pack_is_ready(self) -> None:
        pack = {
            "results_manifest": _artifact("results_manifest", artifact_type="manifest"),
            "main_tables": {
                "table1": _artifact("table1", artifact_type="main_table"),
                "table2": _artifact("table2", artifact_type="main_table"),
                "table3": _artifact("table3", artifact_type="main_table"),
            },
            "appendix_tables": {
                "appendix_cost": _artifact("appendix_cost", artifact_type="appendix_table"),
            },
            "figure_data": {
                "fig1": _artifact("fig1", artifact_type="figure_data"),
                "fig2": _artifact("fig2", artifact_type="figure_data"),
                "fig3": _artifact("fig3", artifact_type="figure_data"),
            },
            "claim_evidence_map": _artifact("claim_evidence_map", artifact_type="claim_evidence_map"),
            "frozen_protocol": {
                "metrics": ["preference_accuracy", "worst_group_preference_accuracy"],
                "baselines": ["Random", "Reward Margin", "APL", "ActiveDPO", "APL+DCMS", "ActiveDPO+DCMS"],
                "judge_version": "judge-v1",
                "freeze_policy": "bug-fixes-only",
            },
        }

        report = validate_result_freeze_pack(
            pack,
            expected_main_tables=["table1", "table2", "table3"],
            expected_figures=["fig1", "fig2", "fig3"],
            expected_metrics=["preference_accuracy", "worst_group_preference_accuracy"],
            expected_baselines=["Random", "Reward Margin", "APL", "ActiveDPO", "APL+DCMS", "ActiveDPO+DCMS"],
        )

        self.assertTrue(report.is_ready)
        self.assertEqual([], report.issues)
        self.assertEqual(3, report.main_table_count)
        self.assertEqual(3, report.figure_data_count)

    def test_missing_required_artifacts_are_reported(self) -> None:
        report = validate_result_freeze_pack(
            {
                "results_manifest": _artifact("results_manifest", artifact_type="manifest"),
                "main_tables": {"table1": _artifact("table1", artifact_type="main_table")},
                "figure_data": {"fig1": _artifact("fig1", artifact_type="figure_data")},
            },
            expected_main_tables=["table1", "table2"],
            expected_figures=["fig1", "fig2"],
            expected_metrics=["preference_accuracy"],
            expected_baselines=["Random"],
        )

        self.assertFalse(report.is_ready)
        codes = {issue["code"] for issue in report.issues}
        self.assertIn("missing_main_table", codes)
        self.assertIn("missing_figure_data", codes)
        self.assertIn("missing_claim_evidence_map", codes)
        self.assertIn("missing_frozen_protocol", codes)

    def test_artifacts_must_expose_traceability_and_failed_run_policy(self) -> None:
        pack = {
            "results_manifest": {"artifact_type": "manifest", "path": "experiments/results_manifest.json"},
            "main_tables": {"table1": _artifact("table1", artifact_type="main_table")},
            "figure_data": {"fig1": _artifact("fig1", artifact_type="figure_data")},
            "claim_evidence_map": _artifact("claim_evidence_map", artifact_type="claim_evidence_map"),
            "frozen_protocol": {
                "metrics": ["preference_accuracy"],
                "baselines": ["Random"],
                "judge_version": "judge-v1",
                "freeze_policy": "bug-fixes-only",
            },
        }

        report = validate_result_freeze_pack(
            pack,
            expected_main_tables=["table1"],
            expected_figures=["fig1"],
            expected_metrics=["preference_accuracy"],
            expected_baselines=["Random"],
        )

        self.assertFalse(report.is_ready)
        self.assertIn("artifact_missing_input_files", {issue["code"] for issue in report.issues})
        self.assertIn("artifact_missing_failed_run_policy", {issue["code"] for issue in report.issues})

    def test_freeze_protocol_must_match_expected_metrics_and_baselines(self) -> None:
        pack = {
            "results_manifest": _artifact("results_manifest", artifact_type="manifest"),
            "main_tables": {"table1": _artifact("table1", artifact_type="main_table")},
            "figure_data": {"fig1": _artifact("fig1", artifact_type="figure_data")},
            "claim_evidence_map": _artifact("claim_evidence_map", artifact_type="claim_evidence_map"),
            "frozen_protocol": {
                "metrics": ["preference_accuracy"],
                "baselines": ["Random"],
                "judge_version": "",
                "freeze_policy": "replace-main-metrics-if-needed",
            },
        }

        report = validate_result_freeze_pack(
            pack,
            expected_main_tables=["table1"],
            expected_figures=["fig1"],
            expected_metrics=["preference_accuracy", "worst_group_preference_accuracy"],
            expected_baselines=["Random", "APL"],
        )

        codes = {issue["code"] for issue in report.issues}
        self.assertIn("missing_frozen_metric", codes)
        self.assertIn("missing_frozen_baseline", codes)
        self.assertIn("missing_judge_version", codes)
        self.assertIn("invalid_freeze_policy", codes)


def _artifact(name: str, *, artifact_type: str) -> dict[str, object]:
    return {
        "artifact_type": artifact_type,
        "path": f"experiments/reports/{name}.json",
        "input_result_files": ["experiments/runs/run_records.jsonl"],
        "aggregation_rule": "frozen protocol aggregation",
        "seed_count": 5,
        "error_bar": "bootstrap 95% CI",
        "includes_failed_runs": True,
    }


if __name__ == "__main__":
    unittest.main()
