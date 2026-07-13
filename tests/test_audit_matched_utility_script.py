from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path
import unittest

from mias_dcms.utils import write_jsonl


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class AuditMatchedUtilityScriptTest(unittest.TestCase):
    def test_audit_matched_utility_writes_utility_and_coverage_report(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            baseline_path = tmp_path / "baseline.jsonl"
            treatment_path = tmp_path / "treatment.jsonl"
            output_path = tmp_path / "report.json"

            write_jsonl(
                [
                    {"sample_id": "b1", "utility": 0.9, "group": "A"},
                    {"sample_id": "b2", "utility": 0.7, "group": "A"},
                    {"sample_id": "b3", "utility": 0.3, "group": "B"},
                    {"sample_id": "b4", "utility": 0.1, "group": "B"},
                ],
                baseline_path,
            )
            write_jsonl(
                [
                    {"sample_id": "t1", "utility": 0.88, "group": "A"},
                    {"sample_id": "t2", "utility": 0.72, "group": "A"},
                    {"sample_id": "t3", "utility": 0.31, "group": "A"},
                    {"sample_id": "t4", "utility": 0.09, "group": "B"},
                ],
                treatment_path,
            )

            completed = subprocess.run(
                [
                    sys.executable,
                    str(PROJECT_ROOT / "scripts" / "audit_matched_utility.py"),
                    "--baseline_path",
                    str(baseline_path),
                    "--treatment_path",
                    str(treatment_path),
                    "--output_path",
                    str(output_path),
                    "--utility_field",
                    "utility",
                    "--group_field",
                    "group",
                    "--target_moments",
                    '{"A": 0.5, "B": 0.5}',
                    "--mean_tolerance",
                    "0.03",
                    "--quantile_tolerance",
                    "0.03",
                ],
                check=True,
                cwd=PROJECT_ROOT,
                text=True,
                capture_output=True,
            )

            stdout_payload = json.loads(completed.stdout)
            file_payload = json.loads(output_path.read_text(encoding="utf-8"))

            self.assertEqual(file_payload, stdout_payload)
            self.assertTrue(file_payload["utility_matched"])
            self.assertEqual(4, file_payload["baseline_count"])
            self.assertEqual(4, file_payload["treatment_count"])
            self.assertEqual({"A": 0.5, "B": 0.5}, file_payload["baseline_moments"])
            self.assertEqual({"A": 0.75, "B": 0.25}, file_payload["treatment_moments"])
            self.assertGreater(
                file_payload["treatment_coverage_deviation"],
                file_payload["baseline_coverage_deviation"],
            )


if __name__ == "__main__":
    unittest.main()
