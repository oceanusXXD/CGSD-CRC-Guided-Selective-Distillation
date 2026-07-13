from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from mias_dcms.utils import write_json, write_jsonl


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class MaterializePreferenceDPOEvaluationScriptTest(unittest.TestCase):
    def test_writes_all_supported_heldout_evaluation_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            pool_path = tmp / "heldout_pool.jsonl"
            oracle_path = tmp / "heldout_oracle.json"
            logprobs_path = tmp / "heldout_logprobs.jsonl"
            output_dir = tmp / "output"
            write_jsonl([{"sample_id": "one", "length_gap_bin": "balanced"}], pool_path)
            write_json({"one": {"preference_label": "A"}}, oracle_path)
            write_jsonl(
                [
                    {
                        "sample_id": "one",
                        "policy_logprob_response_1": 1.0,
                        "policy_logprob_response_2": 0.0,
                        "reference_logprob_response_1": 0.0,
                        "reference_logprob_response_2": 1.0,
                    }
                ],
                logprobs_path,
            )

            completed = subprocess.run(
                [
                    sys.executable,
                    str(PROJECT_ROOT / "scripts" / "materialize_preference_dpo_evaluation.py"),
                    "--heldout_pool_path",
                    str(pool_path),
                    "--heldout_oracle_store_path",
                    str(oracle_path),
                    "--heldout_logprobs_path",
                    str(logprobs_path),
                    "--output_dir",
                    str(output_dir),
                    "--seed_budget",
                    "8",
                    "--active_budget",
                    "4",
                ],
                cwd=PROJECT_ROOT,
                text=True,
                capture_output=True,
                check=False,
            )

            self.assertEqual(0, completed.returncode, completed.stderr)
            payload = json.loads((output_dir / "evaluation_materialization.json").read_text(encoding="utf-8"))
            self.assertEqual(1.0, payload["evaluation_metrics"]["preference_accuracy"])
            self.assertFalse(payload["metadata"]["generation_judge_available"])
            self.assertTrue((output_dir / "heldout_preference_predictions.jsonl").is_file())
            self.assertTrue((output_dir / "judge_rows.jsonl").is_file())
            self.assertTrue((output_dir / "aulc_rows.jsonl").is_file())


if __name__ == "__main__":
    unittest.main()
