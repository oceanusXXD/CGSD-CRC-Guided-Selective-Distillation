from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from mias_dcms.records import RunRecord
from mias_dcms.utils import write_json, write_jsonl


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class BuildPaperArtifactsScriptTest(unittest.TestCase):
    def test_script_writes_artifact_pack_and_component_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            run_records_path = tmp / "runs.jsonl"
            intervention_path = tmp / "intervention_statistics.json"
            matched_path = tmp / "matched_utility.json"
            claim_path = tmp / "claim_audit.json"
            output_dir = tmp / "paper_artifacts"
            write_jsonl(
                [
                    _run_row("Random", 1, 0.60, 0.50, 0.20),
                    _run_row("Random", 2, 0.64, 0.52, 0.22),
                    _run_row("APL+DCMS", 1, 0.67, 0.58, 0.08),
                    _run_row("APL+DCMS", 2, 0.70, 0.61, 0.09),
                ],
                run_records_path,
            )
            write_json({"is_ready": True, "by_setting": {}}, intervention_path)
            write_json({"points": [{"coverage_deviation": 0.1, "worst_group_delta": 0.02, "base_utility": 0.8}]}, matched_path)
            write_json({"is_ready": True, "issues": []}, claim_path)

            completed = subprocess.run(
                [
                    sys.executable,
                    str(PROJECT_ROOT / "scripts" / "build_paper_artifacts.py"),
                    "--run_records_path",
                    str(run_records_path),
                    "--intervention_statistics_path",
                    str(intervention_path),
                    "--matched_utility_path",
                    str(matched_path),
                    "--claim_audit_path",
                    str(claim_path),
                    "--output_dir",
                    str(output_dir),
                    "--expected_baselines",
                    "Random,APL+DCMS",
                    "--evaluation_metrics",
                    "preference_accuracy,worst_group_preference_accuracy",
                    "--selection_metrics",
                    "acquisition_tv,utility_retained",
                    "--cost_metrics",
                    "train_tokens,judge_calls",
                    "--judge_version",
                    "judge-v1",
                    "--resamples",
                    "200",
                ],
                cwd=PROJECT_ROOT,
                text=True,
                capture_output=True,
                check=False,
            )

            self.assertEqual(0, completed.returncode, completed.stderr)
            pack = json.loads((output_dir / "freeze_pack.json").read_text(encoding="utf-8"))
            self.assertTrue((output_dir / "figure_data" / "fig1.json").exists())
            self.assertTrue((output_dir / "main_tables" / "table2.json").exists())
            self.assertEqual({"fig1", "fig2", "fig3"}, set(pack["figure_data"]))
            self.assertEqual({"table1", "table2", "table3"}, set(pack["main_tables"]))


def _run_row(method: str, seed: int, preference_accuracy: float, worst_group: float, acquisition_tv: float) -> dict[str, object]:
    return RunRecord(
        dataset="helpsteer2",
        model="qwen",
        method=method,
        budget=100,
        seed=seed,
        selected_count=100,
        config_hash=f"cfg-{method}-{seed}",
        selection_metrics={"acquisition_tv": acquisition_tv, "utility_retained": 0.96},
        training_metrics={"dpo_train_row_count": 100},
        evaluation_metrics={
            "preference_accuracy": preference_accuracy,
            "worst_group_preference_accuracy": worst_group,
        },
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
