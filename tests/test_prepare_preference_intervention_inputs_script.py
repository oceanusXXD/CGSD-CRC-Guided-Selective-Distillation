from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = PROJECT_ROOT / "scripts" / "prepare_preference_intervention_inputs.py"


class PreparePreferenceInterventionInputsScriptTest(unittest.TestCase):
    def test_script_writes_intervention_rows_and_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            active_path = tmp / "active.jsonl"
            logprobs_path = tmp / "logprobs.jsonl"
            scores_path = tmp / "scores.jsonl"
            output_path = tmp / "intervention_rows.jsonl"
            summary_path = tmp / "summary.json"
            _write_jsonl(
                active_path,
                [
                    {
                        "sample_id": "p1",
                        "response_a": "one two three four",
                        "response_b": "one",
                        "source_pair": "human|model",
                        "ab_position": "original",
                    }
                ],
            )
            _write_jsonl(
                logprobs_path,
                [{"sample_id": "p1", "implicit_reward_gap": 0.8}],
            )
            _write_jsonl(scores_path, [{"sample_id": "p1", "apl_score": 0.7}])

            completed = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "--active_pool_path",
                    str(active_path),
                    "--logprobs_path",
                    str(logprobs_path),
                    "--score_path",
                    str(scores_path),
                    "--output_path",
                    str(output_path),
                    "--summary_path",
                    str(summary_path),
                ],
                cwd=PROJECT_ROOT,
                text=True,
                capture_output=True,
                check=False,
            )

            self.assertEqual(0, completed.returncode, completed.stderr)
            rows = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            self.assertEqual(1, len(rows))
            self.assertEqual("a_longer", rows[0]["length_gap_bin"])
            self.assertAlmostEqual(0.8, rows[0]["base_margin"])
            self.assertAlmostEqual(0.7, rows[0]["apl_score"])
            self.assertEqual({"a_longer": 1}, summary["length_gap_bins"])


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


if __name__ == "__main__":
    unittest.main()
