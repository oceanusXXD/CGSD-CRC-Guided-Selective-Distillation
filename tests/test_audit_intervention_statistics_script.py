from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from mias_dcms.utils import write_jsonl


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class AuditInterventionStatisticsScriptTest(unittest.TestCase):
    def test_script_writes_ready_intervention_statistics_report(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            input_path = tmp / "curve_rows.jsonl"
            output_path = tmp / "statistics.json"
            write_jsonl(
                _curve_rows(
                    setting="helpsteer2_qwen",
                    values=[-2, -1, 0, 1, 2],
                    responses=[0.10, 0.20, 0.50, 0.70, 0.90],
                ),
                input_path,
            )

            completed = subprocess.run(
                [
                    sys.executable,
                    str(PROJECT_ROOT / "scripts" / "audit_intervention_statistics.py"),
                    "--input_path",
                    str(input_path),
                    "--output_path",
                    str(output_path),
                    "--expected_settings",
                    "helpsteer2_qwen",
                    "--minimum_values",
                    "5",
                    "--resamples",
                    "200",
                    "--seed",
                    "11",
                ],
                cwd=PROJECT_ROOT,
                text=True,
                capture_output=True,
                check=False,
            )

            self.assertEqual(0, completed.returncode, completed.stderr)
            payload = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertTrue(payload["is_ready"])
            self.assertEqual(1, payload["completed_setting_count"])
            self.assertEqual(5, payload["by_setting"]["helpsteer2_qwen"]["intervention_value_count"])

    def test_script_returns_nonzero_when_expected_setting_is_hidden(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            input_path = tmp / "curve_rows.jsonl"
            output_path = tmp / "statistics.json"
            write_jsonl(
                _curve_rows(
                    setting="helpsteer2_qwen",
                    values=[-2, -1, 0, 1, 2],
                    responses=[0.10, 0.20, 0.50, 0.70, 0.90],
                ),
                input_path,
            )

            completed = subprocess.run(
                [
                    sys.executable,
                    str(PROJECT_ROOT / "scripts" / "audit_intervention_statistics.py"),
                    "--input_path",
                    str(input_path),
                    "--output_path",
                    str(output_path),
                    "--expected_settings",
                    "helpsteer2_qwen,tldr_llama",
                    "--minimum_values",
                    "5",
                ],
                cwd=PROJECT_ROOT,
                text=True,
                capture_output=True,
                check=False,
            )

            self.assertEqual(1, completed.returncode)
            payload = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertIn("missing_expected_setting", {issue["code"] for issue in payload["issues"]})


def _curve_rows(*, setting: str, values: list[float], responses: list[float]) -> list[dict[str, object]]:
    return [
        {
            "setting": setting,
            "status": "completed",
            "intervention_value": value,
            "target_group_propensity": response,
        }
        for value, response in zip(values, responses, strict=True)
    ]


if __name__ == "__main__":
    unittest.main()
