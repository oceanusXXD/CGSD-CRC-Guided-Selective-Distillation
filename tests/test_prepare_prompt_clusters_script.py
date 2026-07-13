from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = PROJECT_ROOT / "scripts" / "prepare_prompt_clusters.py"


class PreparePromptClustersScriptTest(unittest.TestCase):
    def test_script_writes_cluster_assignments_and_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            active_path = tmp / "active.jsonl"
            embeddings_path = tmp / "embeddings.jsonl"
            output_path = tmp / "clusters.jsonl"
            summary_path = tmp / "clusters.summary.json"
            _write_jsonl(
                active_path,
                [
                    {"sample_id": "a", "prompt": "alpha"},
                    {"sample_id": "b", "prompt": "alpha near"},
                    {"sample_id": "c", "prompt": "beta"},
                    {"sample_id": "d", "prompt": "beta near"},
                ],
            )
            _write_jsonl(
                embeddings_path,
                [
                    {"sample_id": "a", "embedding": [1.0, 0.0]},
                    {"sample_id": "b", "embedding": [0.9, 0.1]},
                    {"sample_id": "c", "embedding": [0.0, 1.0]},
                    {"sample_id": "d", "embedding": [0.1, 0.9]},
                ],
            )

            completed = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "--active_pool_path",
                    str(active_path),
                    "--embeddings_path",
                    str(embeddings_path),
                    "--output_path",
                    str(output_path),
                    "--summary_path",
                    str(summary_path),
                    "--cluster_count",
                    "2",
                ],
                cwd=PROJECT_ROOT,
                text=True,
                capture_output=True,
                check=False,
            )

            self.assertEqual(0, completed.returncode, completed.stderr)
            rows = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            self.assertEqual(4, len(rows))
            self.assertEqual(2, summary["cluster_count"])
            self.assertEqual({"c0": 2, "c1": 2}, summary["cluster_counts"])
            self.assertIn("prompt_cluster_probabilities", rows[0])


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


if __name__ == "__main__":
    unittest.main()
