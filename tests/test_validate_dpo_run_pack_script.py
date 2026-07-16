from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from mias_dcms.dpo_run_pack import DPO_MAIN_METHODS
from mias_dcms.utils import write_json, write_jsonl


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class ValidateDPORunPackScriptTest(unittest.TestCase):
    def test_script_writes_ready_report_for_complete_pack(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            run_path = tmp / "runs.jsonl"
            manifest_path = tmp / "manifest.json"
            output_path = tmp / "report.json"
            rows = [
                _run_row(method=method, seed=seed)
                for seed in (1, 2)
                for method in DPO_MAIN_METHODS
            ]
            write_jsonl(rows, run_path)
            write_json(_manifest(), manifest_path)

            completed = subprocess.run(
                [
                    sys.executable,
                    str(PROJECT_ROOT / "scripts" / "validate_dpo_run_pack.py"),
                    "--run_records_path",
                    str(run_path),
                    "--paper_manifest_path",
                    str(manifest_path),
                    "--output_path",
                    str(output_path),
                    "--datasets",
                    "helpsteer2",
                    "--models",
                    "qwen",
                    "--budgets",
                    "100",
                    "--seeds",
                    "1,2",
                    "--expected_seed_count",
                    "2",
                ],
                cwd=PROJECT_ROOT,
                text=True,
                capture_output=True,
                check=False,
            )

            self.assertEqual(0, completed.returncode, completed.stderr)
            payload = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertTrue(payload["is_ready"])
            self.assertEqual(6, payload["expected_run_count"])
            self.assertEqual(6, payload["completed_run_count"])
            self.assertEqual([], payload["issues"])

    def test_script_returns_nonzero_when_required_run_is_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            run_path = tmp / "runs.jsonl"
            output_path = tmp / "report.json"
            rows = [_run_row(method=method, seed=1) for method in DPO_MAIN_METHODS[:-1]]
            write_jsonl(rows, run_path)

            completed = subprocess.run(
                [
                    sys.executable,
                    str(PROJECT_ROOT / "scripts" / "validate_dpo_run_pack.py"),
                    "--run_records_path",
                    str(run_path),
                    "--output_path",
                    str(output_path),
                    "--datasets",
                    "helpsteer2",
                    "--models",
                    "qwen",
                    "--budgets",
                    "100",
                    "--seeds",
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
            self.assertIn("missing_run", {issue["code"] for issue in payload["issues"]})


def _run_row(*, method: str, seed: int) -> dict[str, object]:
    return {
        "dataset": "helpsteer2",
        "model": "qwen",
        "method": method,
        "budget": 100,
        "seed": seed,
        "selected_count": 100,
        "config_hash": "frozen-config",
        "run_status": "completed",
        "selection_metrics": {
            "acquisition_tv": 0.12,
            "utility_retained": 0.97,
            "max_constraint_violation": 0.0,
        },
        "training_metrics": {
            "dpo_train_row_count": 100,
            "update_steps": 20,
            "training_token_budget": 4096,
        },
        "evaluation_metrics": {
            "preference_accuracy": 0.61,
            "worst_group_preference_accuracy": 0.54,
            "length_controlled_win_rate": 0.58,
            "capability_regression": -0.01,
            "aulc": 0.59,
        },
        "cost_metrics": {
            "seed_label_count": 25,
            "active_label_count": 100,
            "evaluation_label_count": 500,
            "judge_calls": 500,
            "train_tokens": 12000,
            "selector_compute_seconds": 3.5,
            "oracle_label_calls": 100,
        },
    }


def _manifest() -> dict[str, object]:
    artifact = {
        "input_result_files": ["experiments/results/source.jsonl"],
        "aggregation_rule": "mean with bootstrap CI",
        "seed_count": 2,
        "error_bar": "bootstrap 95% CI",
        "includes_failed_runs": True,
    }
    return {
        "results_manifest": {"run_records_path": "experiments/runs/dpo.jsonl"},
        "figures": {"fig1": artifact, "fig2": artifact, "fig3": artifact},
        "tables": {"table1": artifact, "table2": artifact, "table3": artifact},
    }


if __name__ == "__main__":
    unittest.main()
