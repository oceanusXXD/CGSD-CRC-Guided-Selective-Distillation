from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = PROJECT_ROOT / "scripts" / "prepare_preference_pool.py"


class PreparePreferencePoolScriptTest(unittest.TestCase):
    def test_script_writes_selector_safe_pool_oracle_store_and_swap_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            input_path = tmp / "raw_pairs.jsonl"
            output_dir = tmp / "prepared"
            input_path.write_text(
                "\n".join(
                    json.dumps(row)
                    for row in [
                        {
                            "id": "p1",
                            "prompt": "Prompt 1",
                            "response_a": "short answer",
                            "response_b": "a much longer answer",
                            "chosen": "A",
                            "preference_strength": 2,
                        },
                        {
                            "id": "p2",
                            "prompt": "Prompt 2",
                            "response_a": "alpha beta gamma",
                            "response_b": "delta",
                            "chosen": "B",
                            "preference_strength": 1,
                        },
                    ]
                )
                + "\n",
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
                    "--seed",
                    "11",
                    "--force_swap",
                    "false",
                ],
                check=True,
                cwd=PROJECT_ROOT,
            )

            active_rows = [
                json.loads(line)
                for line in (output_dir / "active_pool.jsonl").read_text(encoding="utf-8").splitlines()
                if line
            ]
            oracle_store = json.loads((output_dir / "oracle_store.json").read_text(encoding="utf-8"))
            swap_manifest = json.loads((output_dir / "swap_manifest.json").read_text(encoding="utf-8"))
            summary = json.loads((output_dir / "pool_summary.json").read_text(encoding="utf-8"))

            self.assertEqual(2, len(active_rows))
            self.assertEqual({"p1", "p2"}, set(oracle_store))
            self.assertEqual(2, len(swap_manifest))
            self.assertEqual(2, summary["active_pool_size"])
            self.assertEqual(2, summary["oracle_store_size"])
            self.assertEqual(str(output_dir / "pool_summary.json"), summary["artifacts"]["pool_summary"])
            for row in active_rows:
                self.assertNotIn("chosen", row)
                self.assertNotIn("preference_strength", row)
                self.assertIn("length_gap", row)
                self.assertEqual("original", row["ab_position"])


if __name__ == "__main__":
    unittest.main()
