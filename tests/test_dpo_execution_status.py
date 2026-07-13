from __future__ import annotations

import unittest

from mias_dcms.dpo_execution_manifest import build_dpo_execution_manifest
from mias_dcms.dpo_execution_status import audit_dpo_execution_status
from mias_dcms.experiment_run_matrix import build_experiment_run_matrix


class DPOExecutionStatusTest(unittest.TestCase):
    def test_marks_completed_stages_and_next_blocker_from_existing_artifacts(self) -> None:
        manifest = build_dpo_execution_manifest(_run_matrix())
        existing = {
            manifest["runs"][0]["artifacts"]["selected_ids_path"],
            manifest["runs"][0]["artifacts"]["selection_summary_path"],
            manifest["runs"][0]["artifacts"]["revealed_rows_path"],
            manifest["runs"][0]["artifacts"]["dpo_train_rows_path"],
        }

        report = audit_dpo_execution_status(manifest, existing_paths=existing)

        self.assertFalse(report.is_complete)
        self.assertEqual(2, report.run_count)
        self.assertEqual(1, report.in_progress_run_count)
        self.assertEqual("training", report.runs[0]["next_stage"])
        self.assertEqual("awaiting_reveal", report.runs[1]["stages"][2]["blocker"])
        self.assertEqual("complete", report.runs[0]["stages"][0]["status"])
        self.assertEqual("complete", report.runs[0]["stages"][1]["status"])
        self.assertEqual("blocked", report.runs[0]["stages"][2]["status"])
        self.assertEqual(
            ["experiments/runs/dpo_main/helpsteer2/qwen-0.6b/budget_100/seed_1/Random/dpo_train_rows.jsonl"],
            report.runs[0]["stages"][2]["present_inputs"],
        )
        self.assertEqual(
            [
                "experiments/runs/dpo_main/helpsteer2/qwen-0.6b/budget_100/seed_1/Random/cost_report.json",
                "experiments/runs/dpo_main/helpsteer2/qwen-0.6b/budget_100/seed_1/Random/training_summary.json",
            ],
            report.runs[0]["stages"][2]["missing_outputs"],
        )

    def test_marks_completed_run_when_all_outputs_exist(self) -> None:
        manifest = build_dpo_execution_manifest(_run_matrix())
        first = manifest["runs"][0]
        existing = set()
        for stage in first["stages"]:
            existing.update(str(path) for path in stage["outputs"].values())

        report = audit_dpo_execution_status({"runs": [first]}, existing_paths=existing)

        self.assertTrue(report.is_complete)
        self.assertEqual(1, report.completed_run_count)
        self.assertIsNone(report.runs[0]["next_stage"])
        self.assertTrue(all(stage["status"] == "complete" for stage in report.runs[0]["stages"]))

    def test_preserves_failed_run_reason_and_reports_missing_reason(self) -> None:
        rows = _run_matrix()
        rows[0]["run_status"] = "failed"
        rows[0]["failure_reason"] = "training_oom"
        rows[1]["run_status"] = "failed"
        rows[1]["failure_reason"] = ""
        manifest = build_dpo_execution_manifest(rows)

        report = audit_dpo_execution_status(manifest, existing_paths=set())

        self.assertFalse(report.is_complete)
        self.assertEqual(2, report.failed_run_count)
        self.assertEqual("training_oom", report.runs[0]["failure_reason"])
        self.assertIn("failed_run_missing_reason", {issue["code"] for issue in report.issues})


def _run_matrix() -> list[dict[str, object]]:
    return build_experiment_run_matrix(
        datasets=["helpsteer2"],
        models=["qwen-0.6b"],
        budgets=[100],
        seeds=[1],
        methods=["Random", "APL"],
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
