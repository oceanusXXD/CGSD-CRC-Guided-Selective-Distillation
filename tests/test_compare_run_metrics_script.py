from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from mias_dcms.records import RunRecord
from mias_dcms.utils import write_jsonl


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class CompareRunMetricsScriptTest(unittest.TestCase):
    def test_script_writes_paired_metric_comparison_report(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            input_path = tmp / "runs.jsonl"
            output_path = tmp / "comparison.json"
            write_jsonl(
                [
                    _run_row(method="Random", seed=1, preference_accuracy=0.60),
                    _run_row(method="Random", seed=2, preference_accuracy=0.64),
                    _run_row(method="APL+DCMS", seed=1, preference_accuracy=0.66),
                    _run_row(method="APL+DCMS", seed=2, preference_accuracy=0.70),
                ],
                input_path,
            )

            completed = subprocess.run(
                [
                    sys.executable,
                    str(PROJECT_ROOT / "scripts" / "compare_run_metrics.py"),
                    "--input_path",
                    str(input_path),
                    "--output_path",
                    str(output_path),
                    "--baseline_method",
                    "Random",
                    "--treatment_methods",
                    "APL+DCMS",
                    "--evaluation_metrics",
                    "preference_accuracy",
                    "--expected_seeds",
                    "1,2",
                    "--resamples",
                    "200",
                    "--permutations",
                    "200",
                ],
                cwd=PROJECT_ROOT,
                text=True,
                capture_output=True,
                check=False,
            )

            self.assertEqual(0, completed.returncode, completed.stderr)
            payload = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertTrue(payload["is_ready"])
            self.assertEqual(1, len(payload["comparisons"]))
            self.assertAlmostEqual(0.06, payload["comparisons"][0]["delta_mean"])

    def test_script_returns_nonzero_when_expected_seed_pair_is_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            input_path = tmp / "runs.jsonl"
            output_path = tmp / "comparison.json"
            write_jsonl(
                [
                    _run_row(method="Random", seed=1, preference_accuracy=0.60),
                    _run_row(method="Random", seed=2, preference_accuracy=0.64),
                    _run_row(method="APL+DCMS", seed=1, preference_accuracy=0.66),
                ],
                input_path,
            )

            completed = subprocess.run(
                [
                    sys.executable,
                    str(PROJECT_ROOT / "scripts" / "compare_run_metrics.py"),
                    "--input_path",
                    str(input_path),
                    "--output_path",
                    str(output_path),
                    "--baseline_method",
                    "Random",
                    "--treatment_methods",
                    "APL+DCMS",
                    "--evaluation_metrics",
                    "preference_accuracy",
                    "--expected_seeds",
                    "1,2",
                    "--minimum_paired_seeds",
                    "2",
                ],
                cwd=PROJECT_ROOT,
                text=True,
                capture_output=True,
                check=False,
            )

            self.assertEqual(1, completed.returncode)
            payload = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertFalse(payload["is_ready"])
            self.assertIn("missing_treatment_seed", {issue["code"] for issue in payload["issues"]})


def _run_row(*, method: str, seed: int, preference_accuracy: float) -> dict[str, object]:
    return RunRecord(
        dataset="helpsteer2",
        model="qwen",
        method=method,
        budget=100,
        seed=seed,
        selected_count=100,
        config_hash=f"cfg-{method}-{seed}",
        selection_metrics={"acquisition_tv": 0.10},
        training_metrics={},
        evaluation_metrics={"preference_accuracy": preference_accuracy},
        cost_metrics={
            "seed_label_count": 25,
            "active_label_count": 100,
            "evaluation_label_count": 500,
            "judge_calls": 500,
            "train_tokens": 12000,
            "selector_compute_seconds": 3.5,
        },
    ).as_dict()


if __name__ == "__main__":
    unittest.main()
