from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from mias_dcms.dpo_execution_manifest import build_dpo_execution_manifest
from mias_dcms.experiment_run_matrix import build_experiment_run_matrix
from mias_dcms.utils import write_json


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class AuditDPOExecutionStatusScriptTest(unittest.TestCase):
    def test_script_writes_status_report_from_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            manifest = build_dpo_execution_manifest(_run_matrix())
            _materialize_first_run_selection_outputs(tmp, manifest)
            manifest_path = tmp / "execution_manifest.json"
            output_path = tmp / "execution_status.json"
            write_json(manifest, manifest_path)

            completed = subprocess.run(
                [
                    sys.executable,
                    str(PROJECT_ROOT / "scripts" / "audit_dpo_execution_status.py"),
                    "--manifest_path",
                    str(manifest_path),
                    "--output_path",
                    str(output_path),
                ],
                cwd=PROJECT_ROOT,
                text=True,
                capture_output=True,
                check=False,
            )

            self.assertEqual(1, completed.returncode)
            stdout_payload = json.loads(completed.stdout)
            self.assertNotIn("runs", stdout_payload)
            self.assertEqual(2, stdout_payload["run_count"])
            payload = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertFalse(payload["is_complete"])
            self.assertEqual(2, payload["run_count"])
            self.assertEqual("reveal", payload["runs"][0]["next_stage"])

    def test_script_returns_zero_when_all_run_outputs_exist(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            manifest = build_dpo_execution_manifest(_run_matrix()[:1])
            _materialize_all_outputs(tmp, manifest)
            manifest_path = tmp / "execution_manifest.json"
            output_path = tmp / "execution_status.json"
            write_json(manifest, manifest_path)

            completed = subprocess.run(
                [
                    sys.executable,
                    str(PROJECT_ROOT / "scripts" / "audit_dpo_execution_status.py"),
                    "--manifest_path",
                    str(manifest_path),
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
            self.assertTrue(payload["is_complete"])
            self.assertEqual(1, payload["completed_run_count"])


def _materialize_first_run_selection_outputs(tmp: Path, manifest: dict[str, object]) -> None:
    first = manifest["runs"][0]
    for path in first["stages"][0]["outputs"].values():
        target = tmp / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("{}", encoding="utf-8")


def _materialize_all_outputs(tmp: Path, manifest: dict[str, object]) -> None:
    for run in manifest["runs"]:
        for stage in run["stages"]:
            for path in stage["outputs"].values():
                target = tmp / path
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text("{}", encoding="utf-8")


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
