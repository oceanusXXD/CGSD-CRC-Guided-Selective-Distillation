from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = PROJECT_ROOT / "scripts" / "select_dcms.py"


class SelectDCMSScriptTest(unittest.TestCase):
    def test_script_writes_selected_ids_propensity_and_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            input_path = tmp / "candidates.jsonl"
            output_dir = tmp / "dcms"
            rows = [
                {"sample_id": "a0", "score": 0.9, "groups": {"A": 1.0, "B": 0.0}},
                {"sample_id": "a1", "score": 0.8, "groups": {"A": 1.0, "B": 0.0}},
                {"sample_id": "b0", "score": 0.7, "groups": {"A": 0.0, "B": 1.0}},
                {"sample_id": "b1", "score": 0.1, "groups": {"A": 0.0, "B": 1.0}},
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
                    "2",
                    "--target_moments",
                    "{\"A\": 0.5, \"B\": 0.5}",
                    "--slack_grid",
                    "0.0,0.5",
                    "--kappa",
                    "0.1",
                    "--rounding_seed",
                    "5",
                    "--score_field",
                    "score",
                    "--group_field",
                    "groups",
                    "--id_field",
                    "sample_id",
                ],
                check=True,
                cwd=PROJECT_ROOT,
            )

            selected_ids = json.loads((output_dir / "selected_ids.json").read_text(encoding="utf-8"))
            propensity_rows = [
                json.loads(line)
                for line in (output_dir / "propensity.jsonl").read_text(encoding="utf-8").splitlines()
                if line
            ]
            summary = json.loads((output_dir / "selection_summary.json").read_text(encoding="utf-8"))

            self.assertEqual(["a0", "b0"], selected_ids["selected_ids"])
            self.assertEqual(4, len(propensity_rows))
            self.assertEqual(2, sum(row["selected"] for row in propensity_rows))
            self.assertEqual({"A": 0.5, "B": 0.5}, summary["rounded_moments"])
            self.assertEqual(0.0, summary["selected_slack"])
            self.assertEqual("feasible", summary["solver_status"])


if __name__ == "__main__":
    unittest.main()
