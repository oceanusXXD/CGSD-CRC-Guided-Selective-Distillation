from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = PROJECT_ROOT / "scripts" / "audit_intervention_response.py"


class AuditInterventionResponseScriptTest(unittest.TestCase):
    def test_script_writes_class_intercept_response_curve(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            input_path = tmp / "logits.jsonl"
            output_path = tmp / "class_curve.json"
            rows = [
                {"sample_id": "a", "group": "target", "logits": [0.0, 3.0]},
                {"sample_id": "b", "group": "other", "logits": [0.0, 0.0]},
                {"sample_id": "c", "group": "target", "logits": [3.0, 0.0]},
                {"sample_id": "d", "group": "other", "logits": [0.0, 2.5]},
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
                    "--mode",
                    "class_intercept_entropy",
                    "--values",
                    "-3,0,3",
                    "--budget",
                    "2",
                    "--target_group",
                    "target",
                    "--target_class",
                    "1",
                    "--id_field",
                    "sample_id",
                    "--group_field",
                    "group",
                    "--logits_field",
                    "logits",
                ],
                check=True,
                cwd=PROJECT_ROOT,
            )

            payload = json.loads(output_path.read_text(encoding="utf-8"))
            by_value = {point["value"]: point for point in payload["points"]}
            self.assertEqual(3, len(payload["points"]))
            self.assertEqual(2, by_value[0.0]["budget"])
            self.assertIn("selected_ids", by_value[3.0])
            self.assertIn("target_group_propensity", by_value[3.0])
            self.assertEqual("target", by_value[3.0]["target_group"])

    def test_script_writes_length_gamma_response_curve(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            input_path = tmp / "margins.jsonl"
            output_path = tmp / "length_curve.json"
            rows = [
                {"sample_id": "short_a", "group": "short_gap", "margin": 0.1, "length_gap": -0.5},
                {"sample_id": "long_a", "group": "long_gap", "margin": 0.1, "length_gap": 0.5},
                {"sample_id": "short_b", "group": "short_gap", "margin": 0.0, "length_gap": -0.4},
                {"sample_id": "long_b", "group": "long_gap", "margin": 0.0, "length_gap": 0.4},
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
                    "--mode",
                    "length_gamma_margin",
                    "--values",
                    "-1,0,1",
                    "--budget",
                    "2",
                    "--target_group",
                    "long_gap",
                    "--id_field",
                    "sample_id",
                    "--group_field",
                    "group",
                    "--margin_field",
                    "margin",
                    "--length_gap_field",
                    "length_gap",
                ],
                check=True,
                cwd=PROJECT_ROOT,
            )

            payload = json.loads(output_path.read_text(encoding="utf-8"))
            by_value = {point["value"]: point for point in payload["points"]}
            self.assertEqual(0.0, by_value[-1.0]["target_group_propensity"])
            self.assertEqual(1.0, by_value[1.0]["target_group_propensity"])


if __name__ == "__main__":
    unittest.main()
