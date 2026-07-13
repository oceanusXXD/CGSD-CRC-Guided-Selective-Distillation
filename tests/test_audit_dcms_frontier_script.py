from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = PROJECT_ROOT / "scripts" / "audit_dcms_frontier.py"


class AuditDCMSFrontierScriptTest(unittest.TestCase):
    def test_script_writes_utility_coverage_frontier(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            input_path = tmp / "candidates.jsonl"
            output_path = tmp / "frontier.json"
            rows = [
                {"sample_id": "hi_a0", "score": 1.0, "groups": {"A": 1.0, "B": 0.0}},
                {"sample_id": "hi_a1", "score": 0.9, "groups": {"A": 1.0, "B": 0.0}},
                {"sample_id": "lo_b0", "score": 0.2, "groups": {"A": 0.0, "B": 1.0}},
                {"sample_id": "lo_b1", "score": 0.1, "groups": {"A": 0.0, "B": 1.0}},
            ]
            input_path.write_text(
                "\n".join(json.dumps(row) for row in rows) + "\n",
                encoding="utf-8",
            )

            subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "--input_path",
                    str(input_path),
                    "--output_path",
                    str(output_path),
                    "--budget",
                    "2",
                    "--target_moments",
                    "{\"A\": 0.5, \"B\": 0.5}",
                    "--slack_grid",
                    "0.0,0.5",
                    "--kappa",
                    "0.3",
                    "--id_field",
                    "sample_id",
                    "--score_field",
                    "score",
                    "--group_field",
                    "groups",
                ],
                check=True,
                cwd=PROJECT_ROOT,
            )

            payload = json.loads(output_path.read_text(encoding="utf-8"))

            self.assertEqual(0.5, payload["selected_slack"])
            self.assertEqual(2, len(payload["points"]))
            self.assertEqual([0.0, 0.5], [point["slack"] for point in payload["points"]])
            self.assertAlmostEqual(0.0, payload["points"][0]["coverage_deviation"])
            self.assertAlmostEqual(1.0, payload["points"][1]["utility_retained"])
            self.assertAlmostEqual(0.5, payload["points"][1]["coverage_deviation"])


if __name__ == "__main__":
    unittest.main()
