from __future__ import annotations

import unittest

from mias_dcms.dpo_execution_manifest import (
    DPO_EXECUTION_STAGES,
    build_dpo_execution_manifest,
    validate_dpo_execution_manifest,
)
from mias_dcms.experiment_run_matrix import build_experiment_run_matrix


class DPOExecutionManifestTest(unittest.TestCase):
    def test_builds_ordered_manifest_from_planned_run_matrix(self) -> None:
        manifest = build_dpo_execution_manifest(_run_matrix())

        self.assertEqual(2, manifest["run_count"])
        self.assertEqual(list(DPO_EXECUTION_STAGES), manifest["stage_order"])
        first_run = manifest["runs"][0]
        self.assertEqual("helpsteer2__qwen-0_6b__budget100__seed1__random", first_run["run_id"])
        self.assertEqual("planned", first_run["run_status"])
        self.assertEqual("Random", first_run["method"])
        self.assertEqual([stage["stage"] for stage in first_run["stages"]], list(DPO_EXECUTION_STAGES))

        selection_stage = first_run["stages"][0]
        self.assertEqual("selection", selection_stage["stage"])
        self.assertNotIn("logprobs_path", selection_stage["inputs"])
        self.assertIn("selected_ids_path", selection_stage["outputs"])
        self.assertEqual("blocked", selection_stage["status"])
        self.assertEqual("awaiting_execution", selection_stage["blocker"])

        score_selection_stage = manifest["runs"][1]["stages"][0]
        self.assertIn("logprobs_path", score_selection_stage["inputs"])

        reveal_stage = first_run["stages"][1]
        self.assertEqual(["selection"], reveal_stage["depends_on"])
        self.assertIn("oracle_store_path", reveal_stage["inputs"])
        self.assertIn("dpo_train_rows_path", reveal_stage["outputs"])

        training_stage = first_run["stages"][2]
        self.assertEqual(["reveal"], training_stage["depends_on"])
        self.assertEqual(first_run["artifacts"]["dpo_train_rows_path"], training_stage["inputs"]["dpo_train_rows_path"])
        self.assertEqual(first_run["artifacts"]["training_summary_path"], training_stage["outputs"]["training_summary_path"])
        self.assertIn("scripts/train_preference_dpo_run.py", training_stage["commands"][0])

        evaluation_stage = first_run["stages"][3]
        self.assertEqual(["training"], evaluation_stage["depends_on"])
        self.assertIn("preference_predictions_path", evaluation_stage["inputs"])
        self.assertIn("scripts/audit_preference_evaluation.py", evaluation_stage["commands"][0])

        summary_stage = first_run["stages"][-1]
        self.assertEqual("summary", summary_stage["stage"])
        self.assertEqual(["evaluation"], summary_stage["depends_on"])
        self.assertIn("reveal_summary_path", summary_stage["inputs"])
        self.assertIn("run_record_path", summary_stage["outputs"])
        self.assertIn("scripts/build_dpo_run_record.py", summary_stage["commands"][0])

    def test_failed_runs_are_preserved_without_actionable_stages(self) -> None:
        rows = _run_matrix()
        rows[0]["run_status"] = "failed"
        rows[0]["failure_reason"] = "preflight_missing_logprobs"

        manifest = build_dpo_execution_manifest(rows)

        failed_run = manifest["runs"][0]
        self.assertEqual("failed", failed_run["run_status"])
        self.assertEqual("preflight_missing_logprobs", failed_run["failure_reason"])
        self.assertEqual([], failed_run["stages"])

    def test_validation_reports_missing_artifacts_duplicate_runs_and_stage_order(self) -> None:
        manifest = build_dpo_execution_manifest(_run_matrix())
        manifest["runs"].append(dict(manifest["runs"][0]))
        manifest["runs"][0]["stages"] = list(reversed(manifest["runs"][0]["stages"]))
        del manifest["runs"][1]["artifacts"]["training_summary_path"]

        report = validate_dpo_execution_manifest(manifest)

        self.assertFalse(report.is_ready)
        codes = {issue["code"] for issue in report.issues}
        self.assertIn("duplicate_run_id", codes)
        self.assertIn("stage_order_mismatch", codes)
        self.assertIn("missing_required_artifact", codes)

    def test_manifest_selection_commands_include_prompt_cluster_metadata_when_configured(self) -> None:
        rows = build_experiment_run_matrix(
            datasets=["helpsteer2"],
            models=["qwen-0.6b"],
            budgets=[100],
            seeds=[1],
            methods=["ActiveDPO+DCMS"],
            artifact_root="experiments/runs/dpo_main",
            training_config={
                "initialization": "shared_seed_policy_v1",
                "model_name_or_path": "qwen-0.6b",
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
                "prompt_clusters_path": "experiments/inputs/preference/prompt_clusters.jsonl",
                "active_dpo_length_normalize": True,
                "active_dpo_novelty_weight": 0.25,
            },
        )

        manifest = build_dpo_execution_manifest(rows)

        commands = manifest["runs"][0]["stages"][0]["commands"]
        self.assertIn(
            "prompt_clusters_path",
            manifest["runs"][0]["stages"][0]["inputs"],
        )
        score_command = commands[0]
        self.assertIn("--metadata_path experiments/inputs/preference/prompt_clusters.jsonl", score_command)
        self.assertIn("--active_dpo_length_normalize", score_command)
        self.assertIn("--active_dpo_novelty_weight 0.25", score_command)

    def test_manifest_builds_formal_heldout_evaluation_commands_when_inputs_are_configured(self) -> None:
        training_config = {
            "initialization": "shared_seed_policy_v1",
            "model_name_or_path": "qwen-0.6b",
            "optimizer": "adamw",
            "learning_rate": 5e-6,
            "batch_size": 8,
            "update_steps": 120,
            "train_token_budget": 240000,
            "data_accumulation": "cumulative",
            "prompt_format": "chatml_pairwise_v1",
            "generation_parameters": {"temperature": 0.0, "max_new_tokens": 256},
            "initial_policy_adapter_path": "experiments/inputs/preference/initial_policy_adapter",
            "seed_label_count": 8,
        }
        rows = build_experiment_run_matrix(
            datasets=["helpsteer2"],
            models=["qwen-0.6b"],
            budgets=[4],
            seeds=[1],
            methods=["Random"],
            artifact_root="experiments/runs/dpo_main",
            training_config=training_config,
            judge_config={
                "judge_version": "fixed-human-labels-primary",
                "judge_prompt_hash": "prompt-sha256",
                "evaluator": "held_out_preference_labels",
            },
            data_config={},
            evaluation_config={
                "heldout_pool_path": "experiments/inputs/preference/heldout_pool.jsonl",
                "heldout_oracle_store_path": "experiments/inputs/preference/heldout_oracle.json",
                "heldout_logprobs_path_template": "{run_dir}/heldout_logprobs.jsonl",
                "preference_predictions_path_template": "{run_dir}/heldout_preference_predictions.jsonl",
                "judge_rows_path_template": "{run_dir}/judge_rows.jsonl",
                "capability_rows_path_template": "{run_dir}/capability_rows.jsonl",
                "aulc_rows_path_template": "{run_dir}/aulc_rows.jsonl",
                "group_field": "ab_position",
            },
        )

        manifest = build_dpo_execution_manifest(rows)
        stage = manifest["runs"][0]["stages"][3]

        self.assertEqual(3, len(stage["commands"]))
        self.assertIn("scripts/generate_preference_logprobs.py", stage["commands"][0])
        self.assertIn("scripts/materialize_preference_dpo_evaluation.py", stage["commands"][1])
        self.assertIn("--group_field ab_position", stage["commands"][1])
        self.assertIn("scripts/audit_preference_evaluation.py", stage["commands"][2])
        self.assertIn("--capability_rows_path", stage["commands"][2])
        self.assertIn("heldout_pool_path", stage["inputs"])
        self.assertIn("heldout_logprobs_path", stage["outputs"])

    def test_gradient_dpo_manifest_uses_direct_gradients_and_full_pool_targets(self) -> None:
        rows = build_experiment_run_matrix(
            datasets=["helpsteer2"],
            models=["qwen-0.6b"],
            budgets=[4],
            seeds=[1],
            methods=["GradientDPO", "GradientDPO+DCMS"],
            artifact_root="experiments/runs/dpo_main",
            training_config={
                "initialization": "shared_seed_policy_v1",
                "model_name_or_path": "qwen-0.6b",
                "initial_policy_adapter_path": "experiments/inputs/preference/initial_policy_adapter",
                "optimizer": "adamw",
                "learning_rate": 5e-6,
                "batch_size": 2,
                "update_steps": 10,
                "train_token_budget": 1000,
                "data_accumulation": "cumulative",
                "prompt_format": "chatml_pairwise_v1",
                "generation_parameters": {"temperature": 0.0, "max_new_tokens": 64},
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
                "gradient_dpo_candidate_multiplier": 4,
                "gradient_dpo_kappa": 0.1,
            },
        )

        manifest = build_dpo_execution_manifest(rows)

        base_commands = manifest["runs"][0]["stages"][0]["commands"]
        dcms_commands = manifest["runs"][1]["stages"][0]["commands"]
        self.assertIn("scripts/score_preference_gradients.py", base_commands[1])
        self.assertIn("--candidate_multiplier 4", base_commands[1])
        self.assertIn("scripts/select_preference_baseline.py", base_commands[2])
        self.assertIn("--target_moments_path", dcms_commands[-1])
        self.assertIn("gradient_target_moments.json", dcms_commands[-1])

    def test_mias_manifest_changes_only_pretraining_selection(self) -> None:
        rows = build_experiment_run_matrix(
            datasets=["helpsteer2"],
            models=["qwen-0.6b"],
            budgets=[4],
            seeds=[1],
            methods=["MIAS", "MIAS+DCMS"],
            artifact_root="experiments/runs/dpo_main",
            training_config={
                "initialization": "shared_seed_policy_v1",
                "model_name_or_path": "qwen-0.6b",
                "optimizer": "adamw",
                "learning_rate": 5e-6,
                "batch_size": 2,
                "update_steps": 10,
                "train_token_budget": 1000,
                "data_accumulation": "cumulative",
                "prompt_format": "chatml_pairwise_v1",
                "generation_parameters": {"temperature": 0.0, "max_new_tokens": 64},
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
                "prompt_clusters_path": "experiments/inputs/preference/prompt_clusters.jsonl",
                "mias_seed_rows_path": "experiments/inputs/preference/seed_rows.jsonl",
                "mias_seed_features_path": "experiments/inputs/preference/seed_features.jsonl",
                "mias_pool_features_path": "experiments/inputs/preference/pool_features.jsonl",
                "mias_kappa": 0.1,
            },
        )

        manifest = build_dpo_execution_manifest(rows)
        base_selection = manifest["runs"][0]["stages"][0]
        dcms_selection = manifest["runs"][1]["stages"][0]
        self.assertIn("mias_seed_rows_path", base_selection["inputs"])
        self.assertIn("scripts/select_mias.py", base_selection["commands"][0])
        self.assertNotIn("--dcms", base_selection["commands"][0])
        self.assertIn("--dcms", dcms_selection["commands"][0])
        self.assertNotIn("scripts/score_preference_gradients.py", dcms_selection["commands"][0])
        for run in manifest["runs"]:
            training_command = run["stages"][2]["commands"][0]
            self.assertIn("scripts/train_preference_dpo_run.py", training_command)
            self.assertIn("train_token_budget", training_command)


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
            "model_name_or_path": "qwen-0.6b",
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
        evaluation_config={
            "preference_predictions_path_template": "{run_dir}/heldout_preference_predictions.jsonl",
            "judge_rows_path_template": "{run_dir}/judge_rows.jsonl",
            "capability_rows_path_template": "{run_dir}/capability_rows.jsonl",
        },
    )


if __name__ == "__main__":
    unittest.main()
