from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from mias_dcms.utils import write_json


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class ValidateResultFreezePackScriptTest(unittest.TestCase):
    def test_script_writes_ready_freeze_pack_report(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            pack_path = tmp / "freeze_pack.json"
            output_path = tmp / "report.json"
            write_json(_complete_pack(), pack_path)

            completed = subprocess.run(
                [
                    sys.executable,
                    str(PROJECT_ROOT / "scripts" / "validate_result_freeze_pack.py"),
                    "--freeze_pack_path",
                    str(pack_path),
                    "--output_path",
                    str(output_path),
                    "--expected_main_tables",
                    "table1,table2,table3",
                    "--expected_figures",
                    "fig1,fig2,fig3",
                    "--expected_metrics",
                    "preference_accuracy,worst_group_preference_accuracy",
                    "--expected_baselines",
                    "Random,Reward Margin,APL,ActiveDPO,APL+DCMS,ActiveDPO+DCMS",
                ],
                cwd=PROJECT_ROOT,
                text=True,
                capture_output=True,
                check=False,
            )

            self.assertEqual(0, completed.returncode, completed.stderr)
            payload = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertTrue(payload["is_ready"])
            self.assertEqual([], payload["issues"])

    def test_script_returns_nonzero_for_incomplete_freeze_pack(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            pack_path = tmp / "freeze_pack.json"
            output_path = tmp / "report.json"
            write_json({"main_tables": {}, "figure_data": {}}, pack_path)

            completed = subprocess.run(
                [
                    sys.executable,
                    str(PROJECT_ROOT / "scripts" / "validate_result_freeze_pack.py"),
                    "--freeze_pack_path",
                    str(pack_path),
                    "--output_path",
                    str(output_path),
                    "--expected_main_tables",
                    "table1",
                    "--expected_figures",
                    "fig1",
                    "--expected_metrics",
                    "preference_accuracy",
                    "--expected_baselines",
                    "Random",
                ],
                cwd=PROJECT_ROOT,
                text=True,
                capture_output=True,
                check=False,
            )

            self.assertEqual(1, completed.returncode)
            payload = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertFalse(payload["is_ready"])
            self.assertIn("missing_results_manifest", {issue["code"] for issue in payload["issues"]})


def _complete_pack() -> dict[str, object]:
    return {
        "results_manifest": _artifact("results_manifest", artifact_type="manifest"),
        "main_tables": {
            "table1": _artifact("table1", artifact_type="main_table"),
            "table2": _artifact("table2", artifact_type="main_table"),
            "table3": _artifact("table3", artifact_type="main_table"),
        },
        "appendix_tables": {
            "appendix_cost": _artifact("appendix_cost", artifact_type="appendix_table"),
        },
        "figure_data": {
            "fig1": _artifact("fig1", artifact_type="figure_data"),
            "fig2": _artifact("fig2", artifact_type="figure_data"),
            "fig3": _artifact("fig3", artifact_type="figure_data"),
        },
        "claim_evidence_map": _artifact("claim_evidence_map", artifact_type="claim_evidence_map"),
        "frozen_protocol": {
            "metrics": ["preference_accuracy", "worst_group_preference_accuracy"],
            "baselines": ["Random", "Reward Margin", "APL", "ActiveDPO", "APL+DCMS", "ActiveDPO+DCMS"],
            "judge_version": "judge-v1",
            "freeze_policy": "bug-fixes-only",
        },
    }


def _artifact(name: str, *, artifact_type: str) -> dict[str, object]:
    return {
        "artifact_type": artifact_type,
        "path": f"experiments/reports/{name}.json",
        "input_result_files": ["experiments/runs/run_records.jsonl"],
        "aggregation_rule": "frozen protocol aggregation",
        "seed_count": 5,
        "error_bar": "bootstrap 95% CI",
        "includes_failed_runs": True,
    }


if __name__ == "__main__":
    unittest.main()
