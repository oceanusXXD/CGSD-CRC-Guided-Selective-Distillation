from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = PROJECT_ROOT / "scripts" / "prepare_preference_dcms_inputs.py"


class PreparePreferenceDCMSInputsScriptTest(unittest.TestCase):
    def test_script_writes_dcms_candidates_and_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            input_path = tmp / "baseline_scores.jsonl"
            output_path = tmp / "dcms_candidates.jsonl"
            summary_path = tmp / "summary.json"
            rows = [
                {
                    "sample_id": "p1",
                    "reward_margin_score": 0.8,
                    "length_gap_bin": "short",
                    "source_pair": "human|model",
                },
                {
                    "sample_id": "p2",
                    "reward_margin_score": 0.3,
                    "length_gap_bin": "long",
                    "source_pair": "model|human",
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
                    "--method",
                    "reward_margin",
                    "--group_fields",
                    "length_gap_bin,source_pair",
                ],
                check=True,
                cwd=PROJECT_ROOT,
            )

            candidates = [
                json.loads(line)
                for line in output_path.read_text(encoding="utf-8").splitlines()
                if line
            ]
            summary = json.loads(summary_path.read_text(encoding="utf-8"))

            self.assertEqual(2, len(candidates))
            self.assertEqual("reward_margin", summary["method"])
            self.assertEqual("reward_margin_score", summary["score_field"])
            self.assertEqual(["length_gap_bin", "source_pair"], summary["group_fields"])
            self.assertEqual(2, summary["candidate_count"])
            self.assertEqual(
                {"length_gap_bin=short": 1.0, "source_pair=human|model": 1.0},
                candidates[0]["groups"],
            )


if __name__ == "__main__":
    unittest.main()
