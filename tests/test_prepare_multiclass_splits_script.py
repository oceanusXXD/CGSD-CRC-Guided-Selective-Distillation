from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = PROJECT_ROOT / "scripts" / "prepare_multiclass_splits.py"


class PrepareMulticlassSplitsScriptTest(unittest.TestCase):
    def test_script_writes_fixed_splits_and_pool_prior(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            input_path = tmp / "multiclass.jsonl"
            output_dir = tmp / "prepared"
            input_path.write_text(
                "\n".join(
                    json.dumps({"id": f"s{i}", "label": i % 3, "text": f"text {i}"})
                    for i in range(12)
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
                    "42",
                    "--seed_size",
                    "3",
                    "--active_size",
                    "5",
                    "--test_size",
                    "4",
                    "--label_field",
                    "label",
                ],
                check=True,
                cwd=PROJECT_ROOT,
            )

            splits = json.loads((output_dir / "split_ids.json").read_text(encoding="utf-8"))
            prior = json.loads((output_dir / "pool_prior.json").read_text(encoding="utf-8"))
            summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))

            self.assertEqual(3, len(splits["seed_ids"]))
            self.assertEqual(5, len(splits["active_pool_ids"]))
            self.assertEqual(4, len(splits["test_ids"]))
            self.assertEqual(12, prior["total_count"])
            self.assertEqual(12, summary["row_count"])
            self.assertEqual("label", summary["label_field"])


if __name__ == "__main__":
    unittest.main()
