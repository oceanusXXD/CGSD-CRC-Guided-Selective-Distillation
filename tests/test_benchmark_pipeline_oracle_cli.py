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

    def test_merge_scored_shards_requires_complete_non_overlapping_coverage(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            source_path = tmp / "source.jsonl"
            first_shard = tmp / "first.jsonl"
            second_shard = tmp / "second.jsonl"
            output_path = tmp / "merged.jsonl"
            _write_jsonl(source_path, [{"id": "a"}, {"id": "b"}, {"id": "c"}])
            _write_jsonl(first_shard, [{"id": "c", "entropy": 0.3}, {"id": "a", "entropy": 0.1}])
            _write_jsonl(second_shard, [{"id": "b", "entropy": 0.2}])

            completed = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "merge-classification-scores",
                    "--source-path",
                    str(source_path),
                    "--scored-paths",
                    str(first_shard),
                    str(second_shard),
                    "--output-path",
                    str(output_path),
                ],
                cwd=PROJECT_ROOT,
                text=True,
                capture_output=True,
                check=False,
            )

            self.assertEqual(0, completed.returncode, completed.stderr)
            self.assertEqual(["a", "b", "c"], [row["id"] for row in _read_jsonl(output_path)])

    def test_score_classification_accepts_deterministic_length_sorting(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            source_path = tmp / "source.jsonl"
            output_path = tmp / "scored.jsonl"
            _write_jsonl(
                source_path,
                [
                    {"id": "long", "text": "one two three", "label_name": "World"},
                    {"id": "short", "text": "one", "label_name": "World"},
                ],
            )

            completed = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "score-classification",
                    "--data-path",
                    str(source_path),
                    "--output-path",
                    str(output_path),
                    "--label-names",
                    "World,Sports",
                    "--limit",
                    "0",
                    "--sort-by-text-length",
                ],
                cwd=PROJECT_ROOT,
                text=True,
                capture_output=True,
                check=False,
            )

            self.assertEqual(0, completed.returncode, completed.stderr)
            summary = json.loads(output_path.with_suffix(".summary.json").read_text(encoding="utf-8"))
            self.assertTrue(summary["sort_by_text_length"])

    def test_sanitize_scored_classification_removes_oracle_derived_fields(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            input_path = tmp / "scored.jsonl"
            output_path = tmp / "safe.jsonl"
            _write_jsonl(
                input_path,
                [
                    {
                        "id": "a",
                        "probabilities": [0.8, 0.2],
                        "label": 0,
                        "label_name": "World",
                        "prediction_correct": True,
                    }
                ],
            )

            completed = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "sanitize-classification-scores",
                    "--input-path",
                    str(input_path),
                    "--output-path",
                    str(output_path),
                ],
                cwd=PROJECT_ROOT,
                text=True,
                capture_output=True,
                check=False,
            )

            self.assertEqual(0, completed.returncode, completed.stderr)
            safe = _read_jsonl(output_path)[0]
            self.assertEqual([0.8, 0.2], safe["probabilities"])
            self.assertNotIn("label", safe)
            self.assertNotIn("label_name", safe)
            self.assertNotIn("prediction_correct", safe)

def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


if __name__ == "__main__":
    unittest.main()
