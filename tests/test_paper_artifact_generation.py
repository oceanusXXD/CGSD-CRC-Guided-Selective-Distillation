from __future__ import annotations

import unittest

from mias_dcms.paper_artifacts import build_paper_artifact_pack
from mias_dcms.records import RunRecord


class PaperArtifactGenerationTest(unittest.TestCase):
    def test_builds_three_figure_three_table_artifact_pack(self) -> None:
        runs = [
            _run_record(method="Random", seed=1, preference_accuracy=0.60, worst_group=0.50, acquisition_tv=0.20),
            _run_record(method="Random", seed=2, preference_accuracy=0.64, worst_group=0.52, acquisition_tv=0.22),
            _run_record(method="APL", seed=1, preference_accuracy=0.63, worst_group=0.51, acquisition_tv=0.18),
            _run_record(method="APL", seed=2, preference_accuracy=0.66, worst_group=0.54, acquisition_tv=0.19),
            _run_record(method="APL+DCMS", seed=1, preference_accuracy=0.67, worst_group=0.58, acquisition_tv=0.08),
            _run_record(method="APL+DCMS", seed=2, preference_accuracy=0.70, worst_group=0.61, acquisition_tv=0.09),
        ]
        intervention_statistics = {
            "is_ready": True,
            "by_setting": {
                "helpsteer2_qwen": {
                    "setting": "helpsteer2_qwen",
                    "intervention_value_count": 5,
                    "spearman_monotonicity": 1.0,
                    "slope": 0.2,
                    "slope_ci_low": 0.1,
                    "slope_ci_high": 0.3,
                }
            },
        }
        matched_utility = {
            "coverage_axis": "coverage_deviation",
            "points": [
                {"coverage_deviation": 0.05, "worst_group_delta": 0.01, "base_utility": 0.8},
                {"coverage_deviation": 0.20, "worst_group_delta": -0.04, "base_utility": 0.79},
            ],
        }
        claim_audit = {"is_ready": True, "issues": []}

        pack = build_paper_artifact_pack(
            runs,
            intervention_statistics=intervention_statistics,
            matched_utility=matched_utility,
            claim_audit=claim_audit,
            output_root="experiments/reports/paper",
            expected_main_tables=["table1", "table2", "table3"],
            expected_figures=["fig1", "fig2", "fig3"],
            evaluation_metrics=["preference_accuracy", "worst_group_preference_accuracy"],
            selection_metrics=["acquisition_tv", "utility_retained"],
            cost_metrics=["train_tokens", "judge_calls"],
            expected_baselines=["Random", "APL", "APL+DCMS"],
            judge_version="judge-v1",
        )

        self.assertEqual({"fig1", "fig2", "fig3"}, set(pack["figure_data"]))
        self.assertEqual({"table1", "table2", "table3"}, set(pack["main_tables"]))
        self.assertEqual("bug-fixes-only", pack["frozen_protocol"]["freeze_policy"])
        self.assertEqual("judge-v1", pack["frozen_protocol"]["judge_version"])
        self.assertEqual("pre_label_acquisition", pack["figure_data"]["fig1"]["dcms_stage"])
        self.assertEqual("coverage_deviation", pack["figure_data"]["fig3"]["x_axis"])
        self.assertEqual(["Random", "APL", "APL+DCMS"], pack["frozen_protocol"]["baselines"])
        self.assertIn("input_result_files", pack["main_tables"]["table2"])
        self.assertTrue(pack["claim_evidence_map"]["claim_audit_ready"])

    def test_rejects_pack_when_claim_audit_has_unresolved_issues(self) -> None:
        with self.assertRaises(ValueError):
            build_paper_artifact_pack(
                [_run_record(method="Random", seed=1, preference_accuracy=0.60, worst_group=0.50, acquisition_tv=0.20)],
                intervention_statistics={"is_ready": True, "by_setting": {}},
                matched_utility={"points": []},
                claim_audit={"is_ready": False, "issues": [{"code": "missing_required_evidence_type"}]},
                output_root="experiments/reports/paper",
                expected_main_tables=["table1", "table2", "table3"],
                expected_figures=["fig1", "fig2", "fig3"],
                evaluation_metrics=["preference_accuracy", "worst_group_preference_accuracy"],
                selection_metrics=["acquisition_tv", "utility_retained"],
                cost_metrics=["train_tokens"],
                expected_baselines=["Random"],
                judge_version="judge-v1",
            )

    def test_rejects_missing_expected_baseline_before_artifact_freeze(self) -> None:
        with self.assertRaises(ValueError):
            build_paper_artifact_pack(
                [_run_record(method="Random", seed=1, preference_accuracy=0.60, worst_group=0.50, acquisition_tv=0.20)],
                intervention_statistics={"is_ready": True, "by_setting": {}},
                matched_utility={"points": []},
                claim_audit={"is_ready": True, "issues": []},
                output_root="experiments/reports/paper",
                expected_main_tables=["table1", "table2", "table3"],
                expected_figures=["fig1", "fig2", "fig3"],
                evaluation_metrics=["preference_accuracy"],
                selection_metrics=["acquisition_tv"],
                cost_metrics=["train_tokens"],
                expected_baselines=["Random", "APL+DCMS"],
                judge_version="judge-v1",
            )


def _run_record(
    *,
    method: str,
    seed: int,
    preference_accuracy: float,
    worst_group: float,
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
        selection_metrics={
            "acquisition_tv": acquisition_tv,
            "utility_retained": 0.96 if "DCMS" in method else 1.0,
        },
        training_metrics={"dpo_train_row_count": 100},
        evaluation_metrics={
            "preference_accuracy": preference_accuracy,
            "worst_group_preference_accuracy": worst_group,
        },
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
