from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = PROJECT_ROOT / "scripts" / "audit_preference_intervention.py"


class AuditPreferenceInterventionScriptTest(unittest.TestCase):
    def test_script_writes_length_gamma_audit(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            input_path = tmp / "length_gamma_rows.jsonl"
            output_path = tmp / "length_gamma_audit.json"
            rows = [
                {"sample_id": "short_a", "base_margin": 0.2, "length_gap": -0.5, "length_gap_bin": "short", "source_pair": "human|model"},
                {"sample_id": "long_a", "base_margin": 0.2, "length_gap": 0.5, "length_gap_bin": "long", "source_pair": "model|human"},
                {"sample_id": "short_b", "base_margin": 0.1, "length_gap": -0.4, "length_gap_bin": "short", "source_pair": "human|model"},
                {"sample_id": "long_b", "base_margin": 0.1, "length_gap": 0.4, "length_gap_bin": "long", "source_pair": "model|human"},
            ]
            _write_jsonl(input_path, rows)

            subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "--input_path",
                    str(input_path),
                    "--output_path",
                    str(output_path),
                    "--mode",
                    "length_gamma",
                    "--gammas",
                    "-1,0,1",
                    "--budget",
                    "2",
                    "--target_length_bin",
                    "long",
                    "--linked_group_fields",
                    "source_pair",
                ],
                check=True,
                cwd=PROJECT_ROOT,
            )

            payload = json.loads(output_path.read_text(encoding="utf-8"))

            self.assertEqual("length_gamma", payload["mode"])
            self.assertEqual(str(input_path), payload["input_path"])
            self.assertGreater(payload["target_propensity_slope"], 0.0)

    def test_script_writes_selector_replacement_audit(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            input_path = tmp / "selector_rows.jsonl"
            output_path = tmp / "selector_replacement.json"
            rows = [
                {"sample_id": "p1", "selector_a_score": 0.9, "selector_b_score": 0.1, "length_gap_bin": "short"},
                {"sample_id": "p2", "selector_a_score": 0.8, "selector_b_score": 0.2, "length_gap_bin": "short"},
                {"sample_id": "p3", "selector_a_score": 0.2, "selector_b_score": 0.8, "length_gap_bin": "long"},
                {"sample_id": "p4", "selector_a_score": 0.1, "selector_b_score": 0.9, "length_gap_bin": "long"},
            ]
            _write_jsonl(input_path, rows)

            subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "--input_path",
                    str(input_path),
                    "--output_path",
                    str(output_path),
                    "--mode",
                    "selector_replacement",
                    "--budget",
                    "2",
                    "--selector_a_score_field",
                    "selector_a_score",
                    "--selector_b_score_field",
                    "selector_b_score",
                    "--group_fields",
                    "length_gap_bin",
                ],
                check=True,
                cwd=PROJECT_ROOT,
            )

            payload = json.loads(output_path.read_text(encoding="utf-8"))

            self.assertEqual("selector_replacement", payload["mode"])
            self.assertAlmostEqual(-1.0, payload["score_rank_correlation"])
            self.assertAlmostEqual(0.0, payload["selected_set_overlap"])


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


if __name__ == "__main__":
    unittest.main()
