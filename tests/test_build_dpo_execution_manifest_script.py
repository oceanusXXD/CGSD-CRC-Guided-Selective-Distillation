from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from mias_dcms.experiment_run_matrix import build_experiment_run_matrix
from mias_dcms.utils import write_jsonl


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class BuildDPOExecutionManifestScriptTest(unittest.TestCase):
    def test_script_writes_ready_execution_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            run_matrix_path = tmp / "run_matrix.jsonl"
            output_path = tmp / "execution_manifest.json"
            write_jsonl(_run_matrix(), run_matrix_path)

            completed = subprocess.run(
                [
                    sys.executable,
                    str(PROJECT_ROOT / "scripts" / "build_dpo_execution_manifest.py"),
                    "--run_matrix_path",
                    str(run_matrix_path),
                    "--output_path",
                    str(output_path),
                ],
                cwd=PROJECT_ROOT,
                text=True,
                capture_output=True,
                check=False,
            )

            self.assertEqual(0, completed.returncode, completed.stderr)
            payload = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertTrue(payload["is_ready"])
            self.assertEqual(2, payload["run_count"])
            self.assertEqual(5, len(payload["runs"][0]["stages"]))

    def test_script_returns_nonzero_for_invalid_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            run_matrix_path = tmp / "run_matrix.jsonl"
            output_path = tmp / "execution_manifest.json"
            rows = _run_matrix()
            del rows[0]["artifacts"]["selected_ids_path"]
            write_jsonl(rows, run_matrix_path)

            completed = subprocess.run(
                [
                    sys.executable,
                    str(PROJECT_ROOT / "scripts" / "build_dpo_execution_manifest.py"),
                    "--run_matrix_path",
                    str(run_matrix_path),
                    "--output_path",
                    str(output_path),
                ],
                cwd=PROJECT_ROOT,
                text=True,
                capture_output=True,
                check=False,
            )

            self.assertEqual(1, completed.returncode)
            payload = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertFalse(payload["is_ready"])
            self.assertIn("missing_required_artifact", {issue["code"] for issue in payload["issues"]})


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
