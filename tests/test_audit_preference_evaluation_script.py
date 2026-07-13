from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = PROJECT_ROOT / "scripts" / "audit_preference_evaluation.py"


class AuditPreferenceEvaluationScriptTest(unittest.TestCase):
    def test_script_writes_preference_dpo_evaluation_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            preference_path = tmp / "heldout_preference_predictions.jsonl"
            judge_path = tmp / "judge_rows.jsonl"
            capability_path = tmp / "capability_rows.jsonl"
            output_path = tmp / "evaluation_metrics.json"

            _write_jsonl(
                preference_path,
                [
                    {"sample_id": "p1", "oracle_preference": "A", "predicted_preference": "A", "source_pair": "human|model"},
                    {"sample_id": "p2", "oracle_preference": "B", "predicted_preference": "A", "source_pair": "human|model"},
                    {"sample_id": "p3", "oracle_preference": "B", "predicted_preference": "B", "source_pair": "model|human"},
                ],
            )
            _write_jsonl(
                judge_path,
                [
                    {"sample_id": "g1", "judge_win": 1.0, "length_gap_bin": "short"},
                    {"sample_id": "g2", "judge_win": 0.0, "length_gap_bin": "short"},
                    {"sample_id": "g3", "judge_win": 1.0, "length_gap_bin": "long"},
                ],
            )
            _write_jsonl(
                capability_path,
                [
                    {"task_id": "c1", "baseline_score": 0.9, "policy_score": 0.8},
                    {"task_id": "c2", "baseline_score": 0.7, "policy_score": 0.68},
                ],
            )

            subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "--preference_predictions_path",
                    str(preference_path),
                    "--judge_rows_path",
                    str(judge_path),
                    "--capability_rows_path",
                    str(capability_path),
                    "--output_path",
                    str(output_path),
                    "--group_field",
                    "source_pair",
                    "--length_bin_field",
                    "length_gap_bin",
                ],
                check=True,
                cwd=PROJECT_ROOT,
            )

            payload = json.loads(output_path.read_text(encoding="utf-8"))

            self.assertEqual(str(preference_path), payload["input_paths"]["preference_predictions_path"])
            self.assertEqual(str(judge_path), payload["input_paths"]["judge_rows_path"])
            self.assertEqual(str(capability_path), payload["input_paths"]["capability_rows_path"])
            self.assertAlmostEqual(2.0 / 3.0, payload["evaluation_metrics"]["preference_accuracy"])
            self.assertAlmostEqual(0.5, payload["evaluation_metrics"]["worst_group_preference_accuracy"])
            self.assertAlmostEqual(0.75, payload["evaluation_metrics"]["length_controlled_win_rate"])
            self.assertAlmostEqual(0.06, payload["evaluation_metrics"]["capability_regression"])


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


if __name__ == "__main__":
    unittest.main()
