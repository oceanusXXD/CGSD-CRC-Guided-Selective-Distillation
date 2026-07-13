from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path
import unittest

from mias_dcms.utils import write_jsonl


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = PROJECT_ROOT / "scripts" / "prepare_soft_group_intervals.py"


class PrepareSoftGroupIntervalsScriptTest(unittest.TestCase):
    def test_script_writes_membership_intervals_and_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            input_path = tmp / "soft_groups.jsonl"
            calibration_path = tmp / "calibration.jsonl"
            output_dir = tmp / "out"
            write_jsonl(
                [
                    {
                        "sample_id": "s1",
                        "ensemble_memberships": [
                            {"A": 0.2, "B": 0.8},
                            {"A": 0.4, "B": 0.6},
                            {"A": 0.6, "B": 0.4},
                        ],
                    },
                    {
                        "sample_id": "s2",
                        "ensemble_memberships": [
                            {"A": 0.9, "B": 0.1},
                            {"A": 0.7, "B": 0.3},
                            {"A": 0.8, "B": 0.2},
                        ],
                    },
                ],
                input_path,
            )
            write_jsonl(
                [
                    {"sample_id": "s1", "observed_membership": {"A": 0.4, "B": 0.6}},
                    {"sample_id": "s2", "observed_membership": {"A": 1.0, "B": 0.0}},
                ],
                calibration_path,
            )

            subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "--input_path",
                    str(input_path),
                    "--output_dir",
                    str(output_dir),
                    "--confidence",
                    "1.0",
                    "--calibration_path",
                    str(calibration_path),
                ],
                check=True,
                cwd=PROJECT_ROOT,
            )

            rows = [
                json.loads(line)
                for line in (output_dir / "soft_group_membership.jsonl").read_text(encoding="utf-8").splitlines()
                if line
            ]
            summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
            calibration = json.loads((output_dir / "calibration_summary.json").read_text(encoding="utf-8"))

            self.assertEqual(2, len(rows))
            self.assertEqual(["A", "B"], summary["groups"])
            self.assertEqual(2, summary["sample_count"])
            self.assertIn("calibration_summary_path", summary)
            self.assertAlmostEqual(0.5, calibration["interval_coverage"]["overall_coverage_rate"])
            self.assertAlmostEqual(0.02, calibration["membership_calibration"]["overall_brier_score"])
            self.assertAlmostEqual(0.4, rows[0]["group_membership"]["A"])
            self.assertAlmostEqual(0.2, rows[0]["membership_lower"]["A"])
            self.assertAlmostEqual(0.6, rows[0]["membership_upper"]["A"])


if __name__ == "__main__":
    unittest.main()
