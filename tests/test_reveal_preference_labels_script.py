from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = PROJECT_ROOT / "scripts" / "reveal_preference_labels.py"


class RevealPreferenceLabelsScriptTest(unittest.TestCase):
    def test_script_writes_revealed_rows_training_rows_and_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            active_pool_path = tmp / "active_pool.jsonl"
            oracle_store_path = tmp / "oracle_store.json"
            selected_ids_path = tmp / "selected_ids.json"
            output_dir = tmp / "revealed"

            active_pool_path.write_text(
                "\n".join(
                    json.dumps(row)
                    for row in [
                        {
                            "sample_id": "p1",
                            "prompt": "Prompt 1",
                            "response_a": "A1",
                            "response_b": "B1",
                            "length_gap": 0.0,
                        },
                        {
                            "sample_id": "p2",
                            "prompt": "Prompt 2",
                            "response_a": "A2",
                            "response_b": "B2",
                            "length_gap": 0.5,
                        },
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            oracle_store_path.write_text(
                json.dumps(
                    {
                        "p1": {"sample_id": "p1", "preference_label": "B", "preference_strength": 3},
                        "p2": {"sample_id": "p2", "preference_label": "A", "preference_strength": 1},
                    }
                ),
                encoding="utf-8",
            )
            selected_ids_path.write_text(
                json.dumps({"selected_ids": ["p1"], "method": "apl", "budget": 1}),
                encoding="utf-8",
            )

            subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "--active_pool_path",
                    str(active_pool_path),
                    "--oracle_store_path",
                    str(oracle_store_path),
                    "--selected_ids_path",
                    str(selected_ids_path),
                    "--output_dir",
                    str(output_dir),
                    "--round_index",
                    "2",
                    "--method",
                    "apl",
                ],
                check=True,
                cwd=PROJECT_ROOT,
            )

            revealed_rows = [
                json.loads(line)
                for line in (output_dir / "revealed_rows.jsonl").read_text(encoding="utf-8").splitlines()
                if line
            ]
            training_rows = [
                json.loads(line)
                for line in (output_dir / "dpo_train_rows.jsonl").read_text(encoding="utf-8").splitlines()
                if line
            ]
            summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))

            self.assertEqual(1, len(revealed_rows))
            self.assertEqual(1, len(training_rows))
            self.assertEqual(2, training_rows[0]["preferred_response"])
            self.assertEqual("B", training_rows[0]["oracle_label"])
            self.assertEqual(1, summary["revealed_count"])
            self.assertEqual(1, summary["dpo_train_row_count"])
            self.assertEqual(1, summary["unrevealed_count"])


if __name__ == "__main__":
    unittest.main()
