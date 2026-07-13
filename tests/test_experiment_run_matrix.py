from __future__ import annotations

import unittest

from mias_dcms.dpo_run_pack import DPO_MAIN_METHODS
from mias_dcms.experiment_run_matrix import (
    REQUIRED_JUDGE_CONFIG_FIELDS,
    REQUIRED_TRAINING_CONFIG_FIELDS,
    build_experiment_run_matrix,
    validate_experiment_run_matrix,
)


class ExperimentRunMatrixTest(unittest.TestCase):
    def test_builds_complete_cartesian_matrix_with_stable_artifact_paths(self) -> None:
        rows = build_experiment_run_matrix(
            datasets=["helpsteer2", "tldr"],
            models=["qwen-0.6b"],
            budgets=[100, 250],
            seeds=[1, 2],
            artifact_root="experiments/runs/dpo_main",
            training_config=_training_config(),
            judge_config=_judge_config(),
            data_config={"split_manifest_path": "experiments/inputs/preference/split_manifest.json"},
        )

        self.assertEqual(48, len(rows))
        first = rows[0]
        self.assertEqual(
            "helpsteer2__qwen-0_6b__budget100__seed1__random",
            first["run_id"],
        )
        self.assertEqual("planned", first["run_status"])
        self.assertEqual("Random", first["method"])
        self.assertEqual(100, first["budget"])
        self.assertEqual(1, first["seed"])
        self.assertEqual("pending", first["failure_reason"])
        self.assertEqual(_training_config(), first["training_config"])
        self.assertEqual(_judge_config(), first["judge_config"])
        self.assertEqual({}, first["evaluation_config"])
        self.assertTrue(first["training_config_hash"])
        self.assertTrue(first["judge_config_hash"])
        self.assertTrue(first["evaluation_config_hash"])
        self.assertTrue(first["config_hash"])
        self.assertEqual(
            "experiments/runs/dpo_main/helpsteer2/qwen-0.6b/budget_100/seed_1/Random/active_pool.jsonl",
            first["artifacts"]["active_pool_path"],
        )
        self.assertEqual(
            "experiments/runs/dpo_main/helpsteer2/qwen-0.6b/budget_100/seed_1/Random/evaluation_metrics.json",
            first["artifacts"]["evaluation_metrics_path"],
        )
        self.assertEqual(
            {
                "active_pool_path",
                "oracle_store_path",
                "logprobs_path",
                "selection_summary_path",
                "selected_ids_path",
                "revealed_rows_path",
                "dpo_train_rows_path",
                "training_summary_path",
                "evaluation_metrics_path",
                "cost_report_path",
            },
            set(first["artifacts"]),
        )
        self.assertEqual(
            "experiments/inputs/preference/split_manifest.json",
            first["data_config"]["split_manifest_path"],
        )

        ids = [row["run_id"] for row in rows]
        self.assertEqual(len(ids), len(set(ids)))

    def test_validation_rejects_duplicate_ids_missing_seeds_and_config_drift(self) -> None:
        rows = build_experiment_run_matrix(
            datasets=["helpsteer2"],
            models=["qwen-0.6b"],
            budgets=[100],
            seeds=[1, 2],
            artifact_root="experiments/runs/dpo_main",
            training_config=_training_config(),
            judge_config=_judge_config(),
            data_config={},
        )
        broken_rows = [dict(row) for row in rows if not (row["seed"] == 2 and row["method"] == "Random")]
        duplicate = dict(broken_rows[0])
        drift = dict(broken_rows[1])
        drift["training_config_hash"] = "different-training-hash"
        eval_drift = dict(broken_rows[2])
        eval_drift["evaluation_config_hash"] = "different-evaluation-hash"
        broken_rows.extend([duplicate, drift, eval_drift])

        report = validate_experiment_run_matrix(
            broken_rows,
            expected_datasets=["helpsteer2"],
            expected_models=["qwen-0.6b"],
            expected_budgets=[100],
            expected_seeds=[1, 2],
            expected_methods=DPO_MAIN_METHODS,
        )

        self.assertFalse(report.is_ready)
        codes = {issue["code"] for issue in report.issues}
        self.assertIn("duplicate_run_id", codes)
        self.assertIn("missing_planned_run", codes)
        self.assertIn("training_config_hash_drift", codes)
        self.assertIn("evaluation_config_hash_drift", codes)

    def test_builds_rows_with_explicit_evaluation_config(self) -> None:
        evaluation_config = {
            "preference_predictions_path_template": "{run_dir}/heldout_preference_predictions.jsonl",
            "judge_rows_path_template": "{run_dir}/judge_rows.jsonl",
            "capability_rows_path_template": "{run_dir}/capability_rows.jsonl",
        }

        rows = build_experiment_run_matrix(
            datasets=["helpsteer2"],
            models=["qwen-0.6b"],
            budgets=[100],
            seeds=[1],
            artifact_root="experiments/runs/dpo_main",
            training_config=_training_config(),
            judge_config=_judge_config(),
            data_config={},
            evaluation_config=evaluation_config,
            methods=["Random"],
        )

        self.assertEqual(evaluation_config, rows[0]["evaluation_config"])
        self.assertTrue(rows[0]["evaluation_config_hash"])

    def test_build_rejects_missing_required_training_and_judge_fields(self) -> None:
        training_config = _training_config()
        training_config.pop(REQUIRED_TRAINING_CONFIG_FIELDS[0])
        with self.assertRaisesRegex(ValueError, REQUIRED_TRAINING_CONFIG_FIELDS[0]):
            build_experiment_run_matrix(
                datasets=["helpsteer2"],
                models=["qwen-0.6b"],
                budgets=[100],
                seeds=[1],
                artifact_root="experiments/runs/dpo_main",
                training_config=training_config,
                judge_config=_judge_config(),
                data_config={},
            )

        judge_config = _judge_config()
        judge_config.pop(REQUIRED_JUDGE_CONFIG_FIELDS[0])
        with self.assertRaisesRegex(ValueError, REQUIRED_JUDGE_CONFIG_FIELDS[0]):
            build_experiment_run_matrix(
                datasets=["helpsteer2"],
                models=["qwen-0.6b"],
                budgets=[100],
                seeds=[1],
                artifact_root="experiments/runs/dpo_main",
                training_config=_training_config(),
                judge_config=judge_config,
                data_config={},
            )


def _training_config() -> dict[str, object]:
    return {
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
    }


def _judge_config() -> dict[str, object]:
    return {
        "judge_version": "fixed-human-labels-primary",
        "judge_prompt_hash": "prompt-sha256",
        "evaluator": "held_out_preference_labels",
    }


if __name__ == "__main__":
    unittest.main()
