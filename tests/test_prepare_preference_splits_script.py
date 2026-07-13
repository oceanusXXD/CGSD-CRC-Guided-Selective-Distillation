from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = PROJECT_ROOT / "scripts" / "prepare_preference_splits.py"


class PreparePreferenceSplitsScriptTest(unittest.TestCase):
    def test_script_writes_split_manifest_and_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            input_path = tmp / "active_pool.jsonl"
            output_dir = tmp / "splits"
            input_path.write_text(
                "\n".join(
                    json.dumps({"sample_id": f"p{i}", "prompt": f"prompt {i}"})
                    for i in range(10)
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
                    "--seed",
                    "13",
                    "--seed_size",
                    "2",
                    "--active_size",
                    "5",
                    "--heldout_size",
                    "2",
                    "--test_size",
                    "1",
                ],
                cwd=PROJECT_ROOT,
                text=True,
                capture_output=True,
                check=False,
            )

            self.assertEqual(0, completed.returncode, completed.stderr)
            manifest = json.loads((output_dir / "split_manifest.json").read_text(encoding="utf-8"))
            summary = json.loads((output_dir / "split_summary.json").read_text(encoding="utf-8"))

            self.assertEqual(2, len(manifest["seed_ids"]))
            self.assertEqual(5, len(manifest["active_pool_ids"]))
            self.assertEqual(2, len(manifest["heldout_ids"]))
            self.assertEqual(1, len(manifest["test_ids"]))
            self.assertEqual(10, summary["row_count"])
            self.assertEqual(str(output_dir / "split_manifest.json"), summary["artifacts"]["split_manifest"])
            self.assertEqual(str(output_dir / "split_summary.json"), summary["artifacts"]["split_summary"])


if __name__ == "__main__":
    unittest.main()
