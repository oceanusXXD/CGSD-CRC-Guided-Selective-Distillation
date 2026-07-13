from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = PROJECT_ROOT / "scripts" / "select_moment_matched_random.py"


class SelectMomentMatchedRandomScriptTest(unittest.TestCase):
    def test_script_writes_selected_ids_membership_and_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            input_path = tmp / "candidates.jsonl"
            output_dir = tmp / "moment_random"
            rows = [
                {"sample_id": "a0", "groups": {"A": 1.0, "B": 0.0}},
                {"sample_id": "a1", "groups": {"A": 1.0, "B": 0.0}},
                {"sample_id": "a2", "groups": {"A": 1.0, "B": 0.0}},
                {"sample_id": "b0", "groups": {"A": 0.0, "B": 1.0}},
                {"sample_id": "b1", "groups": {"A": 0.0, "B": 1.0}},
                {"sample_id": "b2", "groups": {"A": 0.0, "B": 1.0}},
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
                    "--budget",
                    "4",
                    "--target_moments",
                    "{\"A\": 0.5, \"B\": 0.5}",
                    "--tolerance",
                    "0.0",
                    "--seed",
                    "11",
                    "--group_field",
                    "groups",
                    "--id_field",
                    "sample_id",
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

            self.assertEqual(4, selected_ids["selected_count"])
            self.assertEqual(4, len(selected_ids["selected_ids"]))
            self.assertEqual(6, len(membership_rows))
            self.assertEqual(4, sum(row["selected"] for row in membership_rows))
            self.assertEqual({"A": 0.5, "B": 0.5}, summary["rounded_moments"])
            self.assertEqual("moment_matched_random", summary["method"])
            self.assertEqual(11, summary["seed"])


if __name__ == "__main__":
    unittest.main()
