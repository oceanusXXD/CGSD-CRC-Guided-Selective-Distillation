from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = PROJECT_ROOT / "scripts" / "select_preference_random.py"


class SelectPreferenceRandomScriptTest(unittest.TestCase):
    def test_group_aware_random_selection_keeps_ab_variants_mutually_exclusive(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            input_path = tmp / "pool.jsonl"
            output_dir = tmp / "selected"
            rows = [
                {"sample_id": "pair1:original", "swap_pair_id": "pair1"},
                {"sample_id": "pair1:swapped", "swap_pair_id": "pair1"},
                {"sample_id": "pair2:original", "swap_pair_id": "pair2"},
                {"sample_id": "pair2:swapped", "swap_pair_id": "pair2"},
            ]
            input_path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")

            completed = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "--input_path",
                    str(input_path),
                    "--output_dir",
                    str(output_dir),
                    "--budget",
                    "2",
                    "--seed",
                    "7",
                    "--selection_group_field",
                    "swap_pair_id",
                ],
                cwd=PROJECT_ROOT,
                text=True,
                capture_output=True,
                check=False,
            )

            self.assertEqual(0, completed.returncode, completed.stderr)
            selected = json.loads((output_dir / "selected_ids.json").read_text(encoding="utf-8"))
            self.assertEqual(2, len({sample_id.split(":")[0] for sample_id in selected["selected_ids"]}))
            self.assertEqual("swap_pair_id", selected["selection_group_field"])


if __name__ == "__main__":
    unittest.main()
