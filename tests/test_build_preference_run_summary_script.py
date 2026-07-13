from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = PROJECT_ROOT / "scripts" / "build_preference_run_summary.py"


class BuildPreferenceRunSummaryScriptTest(unittest.TestCase):
    def test_script_writes_run_record_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            selection_summary_path = tmp / "selection_summary.json"
            reveal_summary_path = tmp / "reveal_summary.json"
            training_rows_path = tmp / "dpo_train_rows.jsonl"
            output_path = tmp / "run_record.json"

            selection_summary_path.write_text(
                json.dumps(
                    {
                        "selected_count": 2,
                        "pool_size": 4,
                        "selection_metrics": {"acquisition_tv": 0.2},
                    }
                ),
                encoding="utf-8",
            )
            reveal_summary_path.write_text(
                json.dumps(
                    {
                        "revealed_count": 2,
                        "dpo_train_row_count": 2,
                        "unrevealed_count": 2,
                    }
                ),
                encoding="utf-8",
            )
            training_rows_path.write_text(
                "\n".join(
                    json.dumps(row)
                    for row in [
                        {"prompt": "p one", "response_1": "a", "response_2": "b"},
                        {"prompt": "p two", "response_1": "c", "response_2": "d e"},
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "--selection_summary_path",
                    str(selection_summary_path),
                    "--reveal_summary_path",
                    str(reveal_summary_path),
                    "--training_rows_path",
                    str(training_rows_path),
                    "--output_path",
                    str(output_path),
                    "--dataset",
                    "helpsteer2_preference",
                    "--model",
                    "qwen-policy",
                    "--method",
                    "Reward Margin",
                    "--budget",
                    "2",
                    "--seed",
                    "11",
                    "--config_hash",
                    "cfg11",
                    "--evaluation_metrics",
                    "{\"preference_accuracy\": 0.58}",
                    "--seed_label_count",
                    "4",
                    "--evaluation_label_count",
                    "6",
                    "--judge_calls",
                    "2",
                    "--selector_compute_seconds",
                    "0.75",
                ],
                check=True,
                cwd=PROJECT_ROOT,
            )

            payload = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertEqual("helpsteer2_preference", payload["dataset"])
            self.assertEqual("Reward Margin", payload["method"])
            self.assertEqual(2, payload["selected_count"])
            self.assertEqual(2, payload["cost_metrics"]["active_label_count"])
            self.assertEqual(17, payload["cost_metrics"]["train_tokens"])
            self.assertEqual(0.58, payload["evaluation_metrics"]["preference_accuracy"])


if __name__ == "__main__":
    unittest.main()
