from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = PROJECT_ROOT / "scripts" / "audit_preference_logprobs.py"


class AuditPreferenceLogprobsScriptTest(unittest.TestCase):
    def test_script_writes_audited_rows_and_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            input_path = tmp / "active_pool_with_logprobs.jsonl"
            output_path = tmp / "audited_logprobs.jsonl"
            summary_path = tmp / "summary.json"
            rows = [
                {
                    "sample_id": "p1",
                    "policy_logprob_response_1": -1.0,
                    "policy_logprob_response_2": -2.0,
                    "reference_logprob_response_1": -1.5,
                    "reference_logprob_response_2": -1.1,
                },
                {
                    "sample_id": "p2",
                    "policy_logprob_response_1": -1.4,
                    "policy_logprob_response_2": -1.1,
                    "reference_logprob_response_1": -1.6,
                    "reference_logprob_response_2": -1.0,
                },
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
                    "--output_path",
                    str(output_path),
                    "--summary_path",
                    str(summary_path),
                ],
                check=True,
                cwd=PROJECT_ROOT,
            )

            audited_rows = [
                json.loads(line)
                for line in output_path.read_text(encoding="utf-8").splitlines()
                if line
            ]
            summary = json.loads(summary_path.read_text(encoding="utf-8"))

            self.assertEqual(2, len(audited_rows))
            self.assertIn("policy_logprob_gap", audited_rows[0])
            self.assertIn("reference_logprob_gap", audited_rows[0])
            self.assertIn("implicit_reward_gap", audited_rows[0])
            self.assertIn("absolute_implicit_margin", audited_rows[0])
            self.assertEqual(2, summary["row_count"])
            self.assertTrue(summary["implicit_margin_not_all_zero"])
            self.assertEqual(str(input_path), summary["input_path"])
            self.assertEqual(str(output_path), summary["output_path"])


if __name__ == "__main__":
    unittest.main()
