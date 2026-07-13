from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = PROJECT_ROOT / "scripts" / "audit_preference_selector_scores.py"


class AuditPreferenceSelectorScoresScriptTest(unittest.TestCase):
    def test_script_writes_selector_sanity_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            input_path = tmp / "baseline_scores.jsonl"
            output_path = tmp / "selector_sanity_summary.json"
            rows = [
                {"sample_id": "p1", "active_dpo_score": 1.2, "length_gap": 0.0},
                {"sample_id": "p2", "active_dpo_score": 0.7, "length_gap": 0.5},
                {"sample_id": "p3", "active_dpo_score": 0.1, "length_gap": 1.0},
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
                    "--method",
                    "active_dpo",
                    "--budget",
                    "2",
                    "--selector_compute_seconds",
                    "2.5",
                ],
                check=True,
                cwd=PROJECT_ROOT,
            )

            payload = json.loads(output_path.read_text(encoding="utf-8"))

            self.assertEqual(str(input_path), payload["input_path"])
            self.assertEqual("active_dpo", payload["method"])
            self.assertEqual(["p1", "p2"], payload["selected_ids"])
            self.assertTrue(payload["score_not_all_equal"])
            self.assertLess(payload["score_length_correlation"], 0.0)
            self.assertAlmostEqual(2.5, payload["selector_compute_seconds"])


if __name__ == "__main__":
    unittest.main()
