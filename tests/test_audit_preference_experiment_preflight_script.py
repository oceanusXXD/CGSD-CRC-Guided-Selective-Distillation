from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from mias_dcms.experiment_run_matrix import build_experiment_run_matrix
from mias_dcms.utils import write_json, write_jsonl


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class AuditPreferenceExperimentPreflightScriptTest(unittest.TestCase):
    def test_script_writes_ready_report(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            paths = _write_inputs(tmp)
            output_path = tmp / "preflight_report.json"

            completed = subprocess.run(
                [
                    sys.executable,
                    str(PROJECT_ROOT / "scripts" / "audit_preference_experiment_preflight.py"),
                    "--active_pool_path",
                    str(paths["active_pool_path"]),
                    "--oracle_store_path",
                    str(paths["oracle_store_path"]),
                    "--logprobs_path",
                    str(paths["logprobs_path"]),
                    "--split_manifest_path",
                    str(paths["split_manifest_path"]),
                    "--run_matrix_path",
                    str(paths["run_matrix_path"]),
                    "--output_path",
                    str(output_path),
                    "--expected_methods",
                    "Random,APL",
                    "--expected_seeds",
                    "1",
                ],
                cwd=PROJECT_ROOT,
                text=True,
                capture_output=True,
                check=False,
            )

            self.assertEqual(0, completed.returncode, completed.stderr)
            payload = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertTrue(payload["is_ready"])
            self.assertEqual(2, payload["active_pool_count"])
            self.assertEqual(2, payload["planned_run_count"])

    def test_script_returns_nonzero_for_label_leakage(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            paths = _write_inputs(tmp)
            active_rows = _active_pool()
            active_rows[0]["chosen"] = "A"
            write_jsonl(active_rows, paths["active_pool_path"])
            output_path = tmp / "preflight_report.json"

            completed = subprocess.run(
                [
                    sys.executable,
                    str(PROJECT_ROOT / "scripts" / "audit_preference_experiment_preflight.py"),
                    "--active_pool_path",
                    str(paths["active_pool_path"]),
                    "--oracle_store_path",
                    str(paths["oracle_store_path"]),
                    "--logprobs_path",
                    str(paths["logprobs_path"]),
                    "--split_manifest_path",
                    str(paths["split_manifest_path"]),
                    "--run_matrix_path",
                    str(paths["run_matrix_path"]),
                    "--output_path",
                    str(output_path),
                    "--expected_methods",
                    "Random,APL",
                    "--expected_seeds",
                    "1",
                ],
                cwd=PROJECT_ROOT,
                text=True,
                capture_output=True,
                check=False,
            )

            self.assertEqual(1, completed.returncode)
            payload = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertFalse(payload["is_ready"])
            self.assertIn("hidden_label_leakage", {issue["code"] for issue in payload["issues"]})


def _write_inputs(tmp: Path) -> dict[str, Path]:
    paths = {
        "active_pool_path": tmp / "active_pool.jsonl",
        "oracle_store_path": tmp / "oracle_store.jsonl",
        "logprobs_path": tmp / "logprobs.jsonl",
        "split_manifest_path": tmp / "split_manifest.json",
        "run_matrix_path": tmp / "run_matrix.jsonl",
    }
    write_jsonl(_active_pool(), paths["active_pool_path"])
    write_jsonl(
        [
            {"sample_id": "p1", "preference_label": "A"},
            {"sample_id": "p2", "preference_label": "B"},
        ],
        paths["oracle_store_path"],
    )
    write_jsonl(_logprobs(), paths["logprobs_path"])
    write_json(
        {
            "seed_ids": ["s1"],
            "active_pool_ids": ["p1", "p2"],
            "heldout_ids": ["h1"],
            "test_ids": ["t1"],
        },
        paths["split_manifest_path"],
    )
    write_jsonl(_run_matrix(paths), paths["run_matrix_path"])
    return paths


def _active_pool() -> list[dict[str, object]]:
    return [
        {"sample_id": "p1", "prompt": "P1", "response_a": "A1", "response_b": "B1"},
        {"sample_id": "p2", "prompt": "P2", "response_a": "A2", "response_b": "B2"},
    ]


def _logprobs() -> list[dict[str, object]]:
    return [
        {
            "sample_id": "p1",
            "policy_logprob_response_1": -1.0,
            "policy_logprob_response_2": -2.0,
            "reference_logprob_response_1": -1.5,
            "reference_logprob_response_2": -1.6,
        },
        {
            "sample_id": "p2",
            "policy_logprob_response_1": -2.0,
            "policy_logprob_response_2": -1.0,
            "reference_logprob_response_1": -1.4,
            "reference_logprob_response_2": -1.8,
        },
    ]


def _run_matrix(paths: dict[str, Path]) -> list[dict[str, object]]:
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
            "active_pool_path": str(paths["active_pool_path"]),
            "oracle_store_path": str(paths["oracle_store_path"]),
            "logprobs_path": str(paths["logprobs_path"]),
            "split_manifest_path": str(paths["split_manifest_path"]),
        },
    )


if __name__ == "__main__":
    unittest.main()
