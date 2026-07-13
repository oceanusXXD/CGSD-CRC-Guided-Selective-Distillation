from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = PROJECT_ROOT / "scripts" / "benchmark_pipeline.py"


class BenchmarkPipelineOracleCliTest(unittest.TestCase):
    def test_diagnostics_join_oracle_only_after_selector_safe_scoring(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            scored_path = tmp / "scored.jsonl"
            oracle_path = tmp / "oracle.json"
            output_dir = tmp / "diagnostics"
            scored_rows = [
                {"id": "a", "dataset": "ag_news", "entropy": 0.9, "margin": 0.1, "probabilities": [0.4, 0.3, 0.2, 0.1]},
                {"id": "b", "dataset": "ag_news", "entropy": 0.8, "margin": 0.2, "probabilities": [0.45, 0.25, 0.2, 0.1]},
                {"id": "c", "dataset": "ag_news", "entropy": 0.2, "margin": 0.8, "probabilities": [0.9, 0.05, 0.03, 0.02]},
                {"id": "d", "dataset": "ag_news", "entropy": 0.1, "margin": 0.9, "probabilities": [0.95, 0.02, 0.02, 0.01]},
            ]
            scored_path.write_text(
                "\n".join(json.dumps(row) for row in scored_rows) + "\n",
                encoding="utf-8",
            )
            oracle_path.write_text(
                json.dumps(
                    {
                        "a": {"id": "a", "label": 0},
                        "b": {"id": "b", "label": 0},
                        "c": {"id": "c", "label": 1},
                        "d": {"id": "d", "label": 1},
                    }
                ),
                encoding="utf-8",
            )

            completed = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "diagnose-classification",
                    "--scored-path",
                    str(scored_path),
                    "--oracle_store_path",
                    str(oracle_path),
                    "--output-dir",
                    str(output_dir),
                    "--budgets",
                    "1",
                    "--methods",
                    "random,entropy",
                    "--seed",
                    "42",
                    "--random-repetitions",
                    "5",
                    "--dependence-permutations",
                    "5",
                ],
                cwd=PROJECT_ROOT,
                text=True,
                capture_output=True,
                check=False,
            )

            self.assertEqual(0, completed.returncode, completed.stderr)
            payload = json.loads((output_dir / "classification_diagnostics.json").read_text(encoding="utf-8"))
            self.assertEqual(str(oracle_path), payload["oracle_store_path"])
            self.assertIn("entropy", payload["methods"])


if __name__ == "__main__":
    unittest.main()
