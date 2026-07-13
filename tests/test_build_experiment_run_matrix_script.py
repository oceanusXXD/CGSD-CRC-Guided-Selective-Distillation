from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from mias_dcms.dpo_run_pack import DPO_MAIN_METHODS
from mias_dcms.utils import write_json


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class BuildExperimentRunMatrixScriptTest(unittest.TestCase):
    def test_script_writes_matrix_and_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            config_path = tmp / "matrix_config.json"
            matrix_path = tmp / "run_matrix.jsonl"
            summary_path = tmp / "run_matrix_summary.json"
            write_json(_matrix_config(), config_path)

            completed = subprocess.run(
                [
                    sys.executable,
                    str(PROJECT_ROOT / "scripts" / "build_experiment_run_matrix.py"),
                    "--config_path",
                    str(config_path),
                    "--output_matrix_path",
                    str(matrix_path),
                    "--output_summary_path",
                    str(summary_path),
                ],
                cwd=PROJECT_ROOT,
                text=True,
                capture_output=True,
                check=False,
            )

            self.assertEqual(0, completed.returncode, completed.stderr)
            rows = [
                json.loads(line)
                for line in matrix_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            self.assertEqual(12, len(rows))
            self.assertEqual(_matrix_config()["evaluation_config"], rows[0]["evaluation_config"])
            self.assertEqual("qwen-0.6b", rows[0]["training_config"]["model_name_or_path"])
            self.assertTrue(summary["is_ready"])
            self.assertEqual(12, summary["planned_run_count"])
            self.assertEqual(list(DPO_MAIN_METHODS), summary["covered_methods"])

    def test_script_returns_nonzero_for_incomplete_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            config_path = tmp / "matrix_config.json"
            matrix_path = tmp / "run_matrix.jsonl"
            summary_path = tmp / "run_matrix_summary.json"
            config = _matrix_config()
            config["training_config"].pop("optimizer")
            write_json(config, config_path)

            completed = subprocess.run(
                [
                    sys.executable,
                    str(PROJECT_ROOT / "scripts" / "build_experiment_run_matrix.py"),
                    "--config_path",
                    str(config_path),
                    "--output_matrix_path",
                    str(matrix_path),
                    "--output_summary_path",
                    str(summary_path),
                ],
                cwd=PROJECT_ROOT,
                text=True,
                capture_output=True,
                check=False,
            )

            self.assertEqual(1, completed.returncode)
            self.assertFalse(matrix_path.exists())
            self.assertIn("optimizer", completed.stderr)


def _matrix_config() -> dict[str, object]:
    return {
        "datasets": ["helpsteer2"],
        "models": ["qwen-0.6b"],
        "budgets": [100],
        "seeds": [1, 2],
        "artifact_root": "experiments/runs/dpo_main",
        "training_config": {
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
        "judge_config": {
            "judge_version": "fixed-human-labels-primary",
            "judge_prompt_hash": "prompt-sha256",
            "evaluator": "held_out_preference_labels",
        },
        "evaluation_config": {
            "preference_predictions_path_template": "{run_dir}/heldout_preference_predictions.jsonl",
            "judge_rows_path_template": "{run_dir}/judge_rows.jsonl",
            "capability_rows_path_template": "{run_dir}/capability_rows.jsonl",
        },
        "data_config": {
            "split_manifest_path": "experiments/inputs/preference/split_manifest.json"
        },
    }


if __name__ == "__main__":
    unittest.main()
