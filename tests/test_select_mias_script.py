from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from mias_dcms.utils import write_jsonl


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class SelectMIASScriptTest(unittest.TestCase):
    def test_preference_cli_writes_auditable_seed_only_selection(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            seed_rows_path = root / "seed.jsonl"
            seed_features_path = root / "seed_features.jsonl"
            candidates_path = root / "candidates.jsonl"
            candidate_features_path = root / "candidate_features.jsonl"
            output_dir = root / "output"
            write_jsonl(
                [
                    {
                        "sample_id": f"s{index}",
                        "preferred_response": 1 if index % 2 else 2,
                    }
                    for index in range(6)
                ],
                seed_rows_path,
            )
            write_jsonl(
                [
                    {
                        "sample_id": f"s{index}",
                        "response_a_embedding": [float(index), 1.0],
                        "response_b_embedding": [1.0, float(6 - index)],
                        "response_a_word_count": 5 + index,
                        "response_b_word_count": 8,
                    }
                    for index in range(6)
                ],
                seed_features_path,
            )
            write_jsonl(
                [{"sample_id": f"c{index}", "ab_position": "original"} for index in range(4)],
                candidates_path,
            )
            write_jsonl(
                [
                    {
                        "sample_id": f"c{index}",
                        "response_a_embedding": [float(index), 0.0],
                        "response_b_embedding": [0.0, float(index)],
                        "response_a_token_count": 7 + index,
                        "response_b_token_count": 9,
                        "prompt_cluster": f"p{index % 2}",
                    }
                    for index in range(4)
                ],
                candidate_features_path,
            )

            completed = subprocess.run(
                [
                    sys.executable,
                    str(PROJECT_ROOT / "scripts" / "select_mias.py"),
                    "--task",
                    "preference",
                    "--seed_rows_path",
                    str(seed_rows_path),
                    "--candidate_rows_path",
                    str(candidates_path),
                    "--seed_feature_path",
                    str(seed_features_path),
                    "--candidate_feature_path",
                    str(candidate_features_path),
                    "--output_dir",
                    str(output_dir),
                    "--budget",
                    "2",
                    "--seed",
                    "3",
                    "--bootstrap_heads",
                    "0",
                ],
                cwd=PROJECT_ROOT,
                text=True,
                capture_output=True,
                check=False,
            )

            self.assertEqual(0, completed.returncode, completed.stderr)
            selected = json.loads((output_dir / "selected_ids.json").read_text(encoding="utf-8"))
            model = json.loads((output_dir / "mias_selector_model.json").read_text(encoding="utf-8"))
            summary = json.loads((output_dir / "selection_summary.json").read_text(encoding="utf-8"))
            scores = [
                json.loads(line)
                for line in (output_dir / "mias_scores.jsonl").read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(2, selected["selected_count"])
            self.assertEqual(4, len(scores))
            self.assertEqual(2, sum(row["selected"] for row in scores))
            self.assertEqual({"fit", "calibration", "meta_validation"}, set(model["split_ids"]))
            self.assertFalse(any("oracle_label" in row or "preferred_response" in row for row in scores))
            self.assertIn("acquisition_tv", summary["selection_metrics"])
            self.assertEqual(1.0, summary["selection_metrics"]["utility_retained"])
            self.assertGreaterEqual(summary["selector_compute_seconds"], 0.0)


if __name__ == "__main__":
    unittest.main()
