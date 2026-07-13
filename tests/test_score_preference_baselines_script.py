from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = PROJECT_ROOT / "scripts" / "score_preference_baselines.py"


class ScorePreferenceBaselinesScriptTest(unittest.TestCase):
    def test_script_writes_scored_rows_and_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            input_path = tmp / "scored_pairs.jsonl"
            output_path = tmp / "baseline_scores.jsonl"
            summary_path = tmp / "summary.json"
            rows = [
                {
                    "sample_id": "p1",
                    "probability_response_1": 0.55,
                    "prompt_cluster_probabilities": [0.5, 0.5],
                    "policy_logprob_response_1": -1.0,
                    "policy_logprob_response_2": -2.0,
                    "reference_logprob_response_1": -1.5,
                    "reference_logprob_response_2": -1.1,
                },
                {
                    "sample_id": "p2",
                    "probability_response_1": 0.9,
                    "prompt_cluster_probabilities": [1.0, 0.0],
                    "policy_logprob_response_1": -1.0,
                    "policy_logprob_response_2": -1.2,
                    "reference_logprob_response_1": -1.05,
                    "reference_logprob_response_2": -1.15,
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
                    "--methods",
                    "reward_margin,apl,active_dpo",
                    "--prompt_entropy_weight",
                    "0.25",
                ],
                check=True,
                cwd=PROJECT_ROOT,
            )

            scored_rows = [
                json.loads(line)
                for line in output_path.read_text(encoding="utf-8").splitlines()
                if line
            ]
            summary = json.loads(summary_path.read_text(encoding="utf-8"))

            self.assertEqual(2, len(scored_rows))
            self.assertEqual(2, summary["row_count"])
            self.assertEqual(["reward_margin", "apl", "active_dpo"], summary["methods"])
            self.assertEqual(0.25, summary["prompt_entropy_weight"])
            self.assertIn("score_fields", summary)
            self.assertAlmostEqual(0.90, scored_rows[0]["reward_margin_score"])
            self.assertIn("apl_score", scored_rows[0])
            self.assertIn("active_dpo_score", scored_rows[0])

    def test_script_can_merge_prompt_cluster_metadata_before_scoring(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            input_path = tmp / "logprobs.jsonl"
            metadata_path = tmp / "clusters.jsonl"
            output_path = tmp / "baseline_scores.jsonl"
            summary_path = tmp / "summary.json"
            input_path.write_text(
                json.dumps(
                    {
                        "sample_id": "p1",
                        "probability_response_1": 0.55,
                        "policy_logprob_response_1": -1.0,
                        "policy_logprob_response_2": -2.0,
                        "reference_logprob_response_1": -1.5,
                        "reference_logprob_response_2": -1.1,
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            metadata_path.write_text(
                json.dumps({"sample_id": "p1", "prompt_cluster_probabilities": [0.5, 0.5]}) + "\n",
                encoding="utf-8",
            )

            subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "--input_path",
                    str(input_path),
                    "--metadata_path",
                    str(metadata_path),
                    "--output_path",
                    str(output_path),
                    "--summary_path",
                    str(summary_path),
                    "--methods",
                    "apl",
                    "--prompt_entropy_weight",
                    "0.25",
                ],
                check=True,
                cwd=PROJECT_ROOT,
            )

            scored_row = json.loads(output_path.read_text(encoding="utf-8").strip())
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            self.assertGreater(scored_row["apl_score"], 0.90)
            self.assertEqual([str(metadata_path)], summary["metadata_paths"])

    def test_script_exposes_active_dpo_fixed_pool_options(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            input_path = tmp / "logprobs.jsonl"
            metadata_path = tmp / "clusters.jsonl"
            output_path = tmp / "baseline_scores.jsonl"
            summary_path = tmp / "summary.json"
            input_path.write_text(
                "\n".join(
                    [
                        json.dumps(
                            {
                                "sample_id": "p1",
                                "probability_response_1": 0.6,
                                "policy_logprob_gap": 1.0,
                                "reference_logprob_gap": 0.0,
                                "token_count_response_1": 5,
                                "token_count_response_2": 5,
                            }
                        ),
                        json.dumps(
                            {
                                "sample_id": "p2",
                                "probability_response_1": 0.6,
                                "policy_logprob_gap": 1.5,
                                "reference_logprob_gap": 0.0,
                                "token_count_response_1": 20,
                                "token_count_response_2": 20,
                            }
                        ),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            metadata_path.write_text(
                "\n".join(
                    [
                        json.dumps({"sample_id": "p1", "prompt_cluster_probabilities": [0.5, 0.5]}),
                        json.dumps({"sample_id": "p2", "prompt_cluster_probabilities": [1.0, 0.0]}),
                    ]
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
                    "--metadata_path",
                    str(metadata_path),
                    "--output_path",
                    str(output_path),
                    "--summary_path",
                    str(summary_path),
                    "--methods",
                    "active_dpo",
                    "--active_dpo_length_normalize",
                    "--active_dpo_novelty_weight",
                    "0.25",
                ],
                check=True,
                cwd=PROJECT_ROOT,
            )

            scored_rows = [
                json.loads(line)
                for line in output_path.read_text(encoding="utf-8").splitlines()
                if line
            ]
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            by_id = {row["sample_id"]: row for row in scored_rows}
            self.assertGreater(by_id["p1"]["active_dpo_score"], by_id["p2"]["active_dpo_score"])
            self.assertIn("active_dpo_gradient_proxy", by_id["p1"])
            self.assertTrue(summary["active_dpo_length_normalize"])
            self.assertEqual(0.25, summary["active_dpo_novelty_weight"])


if __name__ == "__main__":
    unittest.main()
