from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = PROJECT_ROOT / "scripts" / "audit_mias_selection.py"


class AuditSelectionScriptTest(unittest.TestCase):
    def test_script_writes_mias_selection_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            input_path = tmp / "sample_level.jsonl"
            output_path = tmp / "selection_audit.json"
            input_path.write_text(
                "\n".join(
                    json.dumps(row)
                    for row in [
                        {"sample_id": "a1", "group": "A", "selected": True},
                        {"sample_id": "a2", "group": "A", "selected": False},
                        {"sample_id": "b1", "group": "B", "selected": True},
                        {"sample_id": "b2", "group": "B", "selected": True},
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
                    "--output_path",
                    str(output_path),
                    "--group_field",
                    "group",
                    "--selected_field",
                    "selected",
                    "--dataset",
                    "toy",
                    "--method",
                    "Entropy",
                    "--model",
                    "selector-a",
                    "--budget",
                    "3",
                    "--seed",
                    "7",
                    "--config_hash",
                    "cfg",
                ],
                check=True,
                cwd=PROJECT_ROOT,
            )

            summary = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertEqual("toy", summary["dataset"])
            self.assertEqual("Entropy", summary["method"])
            self.assertEqual(4, summary["pool_size"])
            self.assertEqual(3, summary["selected_size"])
            self.assertAlmostEqual(1 / 6, summary["selection_metrics"]["acquisition_tv"])
            self.assertEqual(2.0, summary["selection_metrics"]["maximum_propensity_ratio"])
            self.assertEqual(3, summary["cost_metrics"]["oracle_label_calls"])


if __name__ == "__main__":
    unittest.main()
