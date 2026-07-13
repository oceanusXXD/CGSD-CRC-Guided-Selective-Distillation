from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from mias_dcms.dpo_execution_manifest import build_dpo_execution_manifest
from mias_dcms.experiment_run_matrix import build_experiment_run_matrix
from mias_dcms.utils import write_json, write_jsonl


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class CollectDPORunRecordsScriptTest(unittest.TestCase):
    def test_script_writes_run_records_jsonl_and_report(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            manifest = build_dpo_execution_manifest(_run_matrix())
            _write_complete_artifacts(tmp, manifest["runs"][0])
            manifest_path = tmp / "execution_manifest.json"
            output_records_path = tmp / "run_records.jsonl"
            output_report_path = tmp / "collection_report.json"
            write_json(manifest, manifest_path)

            completed = subprocess.run(
                [
                    sys.executable,
                    str(PROJECT_ROOT / "scripts" / "collect_dpo_run_records.py"),
                    "--manifest_path",
                    str(manifest_path),
                    "--output_records_path",
                    str(output_records_path),
                    "--output_report_path",
                    str(output_report_path),
                ],
                cwd=PROJECT_ROOT,
                text=True,
                capture_output=True,
                check=False,
            )

            self.assertEqual(0, completed.returncode, completed.stderr)
            records = [
                json.loads(line)
                for line in output_records_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            report = json.loads(output_report_path.read_text(encoding="utf-8"))
            self.assertTrue(report["is_ready"])
            self.assertEqual(1, len(records))
            self.assertEqual("completed", records[0]["run_status"])
            self.assertEqual(1, report["completed_run_count"])

    def test_script_returns_nonzero_when_required_artifacts_are_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            manifest = build_dpo_execution_manifest(_run_matrix())
            manifest_path = tmp / "execution_manifest.json"
            output_records_path = tmp / "run_records.jsonl"
            output_report_path = tmp / "collection_report.json"
            write_json(manifest, manifest_path)

            completed = subprocess.run(
                [
                    sys.executable,
                    str(PROJECT_ROOT / "scripts" / "collect_dpo_run_records.py"),
                    "--manifest_path",
                    str(manifest_path),
                    "--output_records_path",
                    str(output_records_path),
                    "--output_report_path",
                    str(output_report_path),
                ],
                cwd=PROJECT_ROOT,
                text=True,
                capture_output=True,
                check=False,
            )

            self.assertEqual(1, completed.returncode)
            report = json.loads(output_report_path.read_text(encoding="utf-8"))
            self.assertFalse(report["is_ready"])
            self.assertIn("missing_required_artifact_payload", {issue["code"] for issue in report["issues"]})


def _write_complete_artifacts(tmp: Path, run: dict[str, object]) -> None:
    artifacts = run["artifacts"]
    payloads = {
        artifacts["selection_summary_path"]: {
            "selected_count": 100,
            "selection_metrics": {
                "acquisition_tv": 0.1,
                "utility_retained": 0.96,
                "max_constraint_violation": 0.0,
            },
        },
        artifacts["revealed_rows_path"]: [{"sample_id": "p1", "oracle_label": "A"}],
        artifacts["dpo_train_rows_path"]: [
            {"prompt": "prompt", "response_1": "chosen", "response_2": "rejected"}
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
    for relative_path, payload in payloads.items():
        target = tmp / relative_path
        target.parent.mkdir(parents=True, exist_ok=True)
        if str(target).endswith(".jsonl"):
            write_jsonl(payload, target)
        else:
            write_json(payload, target)


def _run_matrix() -> list[dict[str, object]]:
    return build_experiment_run_matrix(
        datasets=["helpsteer2"],
        models=["qwen-0.6b"],
        budgets=[100],
        seeds=[1],
        methods=["Random"],
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
