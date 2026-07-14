from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class PreparePreferenceCPUPilotScriptTest(unittest.TestCase):
    def test_script_materializes_label_isolated_fixed_splits(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            pool_path = tmp / "active_pool.jsonl"
            oracle_path = tmp / "oracle_store.json"
            output_dir = tmp / "pilot"
            _write_pool(pool_path)
            oracle_path.write_text(json.dumps(_oracle_store()), encoding="utf-8")

            completed = subprocess.run(
                [
                    sys.executable,
                    str(PROJECT_ROOT / "scripts" / "prepare_preference_cpu_pilot.py"),
                    "--input_pool_path",
                    str(pool_path),
                    "--oracle_store_path",
                    str(oracle_path),
                    "--output_dir",
                    str(output_dir),
                    "--seed",
                    "1",
                    "--seed_size",
                    "2",
                    "--selection_size",
                    "2",
                    "--heldout_size",
                    "2",
                    "--test_size",
                    "2",
                    "--max_response_word_count",
                    "4",
                ],
                cwd=PROJECT_ROOT,
                text=True,
                capture_output=True,
                check=False,
            )

            self.assertEqual(0, completed.returncode, completed.stderr)
            summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(8, summary["candidate_row_count"])
            self.assertEqual(2, len(json.loads((output_dir / "seed_selected_ids.json").read_text(encoding="utf-8"))["selected_ids"]))
            for split in ("seed", "selection", "heldout", "test"):
                rows = [json.loads(line) for line in (output_dir / f"{split}_pool.jsonl").read_text(encoding="utf-8").splitlines()]
                self.assertEqual(2, len(rows))
                self.assertTrue(all("preference_label" not in row for row in rows))


def _write_pool(path: Path) -> None:
    rows = [
        {
            "sample_id": f"sample-{index}",
            "id": f"sample-{index}",
            "prompt": f"prompt {index}",
            "response_a": "one two",
            "response_b": "three four",
            "response_a_word_count": 2,
            "response_b_word_count": 2,
            "ab_position": "original",
        }
        for index in range(8)
    ]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _oracle_store() -> dict[str, dict[str, str]]:
    return {
        f"sample-{index}": {"sample_id": f"sample-{index}", "preference_label": "A"}
        for index in range(8)
    }


if __name__ == "__main__":
    unittest.main()
