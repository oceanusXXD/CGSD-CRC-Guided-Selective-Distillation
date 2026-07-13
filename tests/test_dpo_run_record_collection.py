from __future__ import annotations

import unittest

from mias_dcms.dpo_execution_manifest import build_dpo_execution_manifest
from mias_dcms.dpo_run_record_collection import collect_dpo_run_records
from mias_dcms.experiment_run_matrix import build_experiment_run_matrix


class DPORunRecordCollectionTest(unittest.TestCase):
    def test_collects_completed_run_record_from_execution_artifacts(self) -> None:
        manifest = build_dpo_execution_manifest(_run_matrix(methods=["Random"]))
        run = manifest["runs"][0]
        artifacts = run["artifacts"]
        artifact_payloads = {
            artifacts["selection_summary_path"]: {
                "selected_count": 100,
                "pool_size": 250,
                "selection_metrics": {
                    "acquisition_tv": 0.1,
                    "utility_retained": 0.96,
                    "max_constraint_violation": 0.0,
                },
            },
            artifacts["revealed_rows_path"]: [
                {"sample_id": "p1", "oracle_label": "A"},
                {"sample_id": "p2", "oracle_label": "B"},
            ],
            artifacts["dpo_train_rows_path"]: [
                {"prompt": "prompt one", "response_1": "chosen", "response_2": "rejected"},
                {"prompt": "prompt two", "response_1": "chosen two", "response_2": "rejected two"},
            ],
            artifacts["training_summary_path"]: {
                "training_metrics": {
                    "dpo_train_row_count": 100,
                    "update_steps": 20,
                    "training_token_budget": 4096,
                }
            },
            artifacts["evaluation_metrics_path"]: {
                "preference_accuracy": 0.61,
                "worst_group_preference_accuracy": 0.54,
                "length_controlled_win_rate": 0.58,
                "capability_regression": -0.01,
                "aulc": 0.59,
            },
            artifacts["cost_report_path"]: {
                "seed_label_count": 25,
                "evaluation_label_count": 500,
                "judge_calls": 500,
                "selector_compute_seconds": 3.5,
            },
        }

        report = collect_dpo_run_records(manifest, artifact_payloads=artifact_payloads)

        self.assertTrue(report.is_ready)
        self.assertEqual(1, report.completed_run_count)
        self.assertEqual(1, len(report.records))
        record = report.records[0]
        self.assertEqual("completed", record["run_status"])
        self.assertEqual("Random", record["method"])
        self.assertEqual(100, record["selected_count"])
        self.assertEqual(0.96, record["selection_metrics"]["utility_retained"])
        self.assertEqual(20, record["training_metrics"]["update_steps"])
        self.assertEqual(0.61, record["evaluation_metrics"]["preference_accuracy"])
        self.assertEqual(100, record["cost_metrics"]["active_label_count"])
        self.assertEqual(100, record["cost_metrics"]["oracle_label_calls"])
        self.assertEqual(25, record["cost_metrics"]["seed_label_count"])

    def test_preserves_failed_and_incomplete_runs_as_visible_records(self) -> None:
        rows = _run_matrix(methods=["Random", "APL"])
        rows[0]["run_status"] = "failed"
        rows[0]["failure_reason"] = "training_oom"
        manifest = build_dpo_execution_manifest(rows)
        artifact_payloads = {
            manifest["runs"][1]["artifacts"]["selection_summary_path"]: {
                "selected_count": 100,
                "selection_metrics": {"acquisition_tv": 0.2},
            }
        }

        report = collect_dpo_run_records(manifest, artifact_payloads=artifact_payloads)

        self.assertFalse(report.is_ready)
        self.assertEqual(1, report.failed_run_count)
        self.assertEqual(1, report.incomplete_run_count)
        by_method = {record["method"]: record for record in report.records}
        self.assertEqual("failed", by_method["Random"]["run_status"])
        self.assertEqual("training_oom", by_method["Random"]["failure_reason"])
        self.assertEqual("incomplete", by_method["APL"]["run_status"])
        self.assertIn("missing_required_artifact_payload", {issue["code"] for issue in report.issues})

    def test_reports_required_metric_gaps_before_run_pack_validation(self) -> None:
        manifest = build_dpo_execution_manifest(_run_matrix(methods=["Random"]))
        artifacts = manifest["runs"][0]["artifacts"]
        artifact_payloads = {
            artifacts["selection_summary_path"]: {
                "selected_count": 100,
                "selection_metrics": {"acquisition_tv": 0.1},
            },
            artifacts["revealed_rows_path"]: [{"sample_id": "p1", "oracle_label": "A"}],
            artifacts["dpo_train_rows_path"]: [
                {"prompt": "prompt", "response_1": "chosen", "response_2": "rejected"}
            ],
            artifacts["training_summary_path"]: {"training_metrics": {"update_steps": 20}},
            artifacts["evaluation_metrics_path"]: {"preference_accuracy": 0.61},
            artifacts["cost_report_path"]: {},
        }

        report = collect_dpo_run_records(manifest, artifact_payloads=artifact_payloads)

        self.assertFalse(report.is_ready)
        codes = {issue["code"] for issue in report.issues}
        self.assertIn("missing_required_metric", codes)
        self.assertIn("selection_metrics.utility_retained", {issue.get("metric") for issue in report.issues})
        self.assertIn("evaluation_metrics.aulc", {issue.get("metric") for issue in report.issues})


def _run_matrix(*, methods: list[str]) -> list[dict[str, object]]:
    return build_experiment_run_matrix(
        datasets=["helpsteer2"],
        models=["qwen-0.6b"],
        budgets=[100],
        seeds=[1],
        methods=methods,
        artifact_root="experiments/runs/dpo_main",
        training_config={
            "initialization": "shared_seed_policy_v1",
            "optimizer": "adamw",
            "learning_rate": 5e-6,
            "batch_size": 8,
            "update_steps": 120,
            "train_token_budget": 240000,
            "data_accumulation": "cumulative",
            "prompt_format": "chatml_pairwise_v1",
            "generation_parameters": {"temperature": 0.0, "max_new_tokens": 256},
        },
        judge_config={
            "judge_version": "fixed-human-labels-primary",
            "judge_prompt_hash": "prompt-sha256",
            "evaluator": "held_out_preference_labels",
        },
        data_config={
            "active_pool_path": "experiments/inputs/preference/active_pool.jsonl",
            "oracle_store_path": "experiments/inputs/preference/oracle_store.jsonl",
            "logprobs_path": "experiments/inputs/preference/logprobs.jsonl",
            "split_manifest_path": "experiments/inputs/preference/split_manifest.json",
        },
    )


if __name__ == "__main__":
    unittest.main()
