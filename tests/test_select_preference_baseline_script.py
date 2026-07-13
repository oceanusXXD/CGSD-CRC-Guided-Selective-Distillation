from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = PROJECT_ROOT / "scripts" / "select_preference_baseline.py"


class SelectPreferenceBaselineScriptTest(unittest.TestCase):
    def test_script_selects_top_preference_score_and_writes_membership(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            input_path = tmp / "baseline_scores.jsonl"
            output_dir = tmp / "selected"
            rows = [
                {"sample_id": "p1", "reward_margin_score": 0.95, "selector_scores": {"reward_margin": 0.95}},
                {"sample_id": "p2", "reward_margin_score": 0.20, "selector_scores": {"reward_margin": 0.20}},
                {"sample_id": "p3", "reward_margin_score": 0.75, "selector_scores": {"reward_margin": 0.75}},
                {"sample_id": "p4", "reward_margin_score": 0.95, "selector_scores": {"reward_margin": 0.95}},
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
                    "--output_dir",
                    str(output_dir),
                    "--method",
                    "reward_margin",
                    "--budget",
                    "2",
                ],
                check=True,
                cwd=PROJECT_ROOT,
            )

            selected_ids = json.loads((output_dir / "selected_ids.json").read_text(encoding="utf-8"))
            membership_rows = [
                json.loads(line)
                for line in (output_dir / "membership.jsonl").read_text(encoding="utf-8").splitlines()
                if line
            ]
            summary = json.loads((output_dir / "selection_summary.json").read_text(encoding="utf-8"))

            self.assertEqual(["p1", "p4"], selected_ids["selected_ids"])
            self.assertEqual(2, selected_ids["selected_count"])
            self.assertEqual(4, len(membership_rows))
            self.assertEqual(2, sum(row["selected"] for row in membership_rows))
            self.assertEqual("reward_margin", summary["method"])
            self.assertEqual("reward_margin_score", summary["score_field"])
            self.assertEqual(2, summary["budget"])
            self.assertEqual(2, summary["selected_count"])

    def test_script_rejects_hidden_label_fields_before_selecting(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            input_path = tmp / "leaky_scores.jsonl"
            output_dir = tmp / "selected"
            input_path.write_text(
                json.dumps(
                    {
                        "sample_id": "p1",
                        "reward_margin_score": 0.5,
                        "preference_label": "A",
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            completed = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "--input_path",
                    str(input_path),
                    "--output_dir",
                    str(output_dir),
                    "--method",
                    "reward_margin",
                    "--budget",
                    "1",
                ],
                cwd=PROJECT_ROOT,
                text=True,
                capture_output=True,
            )

            self.assertNotEqual(0, completed.returncode)
            self.assertIn("hidden fields", completed.stderr)


if __name__ == "__main__":
    unittest.main()
