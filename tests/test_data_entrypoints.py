from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from scripts.convert_jsonl import convert_rows
from src.data import load_examples


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class DataEntrypointsTest(unittest.TestCase):
    def test_convert_rows_maps_external_fields_to_canonical_format(self) -> None:
        rows = [
            {"uid": "a", "text": "doc a", "target": "1", "source": "kept"},
            {"uid": "b", "text": "doc b", "target": 0},
        ]

        converted = convert_rows(
            rows,
            id_field="uid",
            document_field="text",
            label_field="target",
            fixed_query="is positive?",
        )

        self.assertEqual(
            [
                {
                    "uid": "a",
                    "text": "doc a",
                    "target": "1",
                    "source": "kept",
                    "id": "a",
                    "query": "is positive?",
                    "document": "doc a",
                    "groundtruth": 1,
                },
                {
                    "uid": "b",
                    "text": "doc b",
                    "target": 0,
                    "id": "b",
                    "query": "is positive?",
                    "document": "doc b",
                    "groundtruth": 0,
                },
            ],
            converted,
        )

    def test_load_examples_rejects_duplicate_ids(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "data.jsonl"
            path.write_text(
                "\n".join(
                    [
                        json.dumps({"id": "x", "query": "q", "document": "a", "groundtruth": 1}),
                        json.dumps({"id": "x", "query": "q", "document": "b", "groundtruth": 0}),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            with self.assertRaises(ValueError):
                load_examples(path)

    def test_prepare_writes_only_guide_final_and_pool_ids(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            data_path = Path(tmpdir) / "data.jsonl"
            rows = [
                {"id": f"r{i}", "query": "q", "document": f"d{i}", "groundtruth": i % 2}
                for i in range(6)
            ]
            data_path.write_text(
                "".join(json.dumps(row) + "\n" for row in rows),
                encoding="utf-8",
            )
            output_dir = Path(tmpdir) / "run"

            subprocess.run(
                [
                    sys.executable,
                    str(PROJECT_ROOT / "scripts" / "prepare.py"),
                    "--data_path",
                    str(data_path),
                    "--output_dir",
                    str(output_dir),
                    "--n_guide",
                    "2",
                    "--n_final",
                    "1",
                    "--seed",
                    "7",
                ],
                cwd=PROJECT_ROOT,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )

            split = json.loads((output_dir / "split_ids.json").read_text(encoding="utf-8"))
            self.assertEqual(2, len(split["guide_ids"]))
            self.assertEqual(1, len(split["final_ids"]))
            self.assertEqual(3, len(split["pool_ids"]))
            self.assertEqual(
                {
                    "guide_ids",
                    "final_ids",
                    "pool_ids",
                    "n_guide",
                    "n_final",
                    "seed",
                    "split_algorithm",
                    "split_strategy",
                    "label_distribution",
                },
                set(split),
            )
            self.assertEqual("stratified", split["split_strategy"])


if __name__ == "__main__":
    unittest.main()
