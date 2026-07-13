from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path
import unittest

from mias_dcms.utils import write_jsonl


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = PROJECT_ROOT / "scripts" / "audit_budget_report.py"


class AuditBudgetReportScriptTest(unittest.TestCase):
    def test_script_writes_budget_reports_and_fairness_comparison(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            input_path = tmp / "budgets.jsonl"
            output_path = tmp / "budget_report.json"
            write_jsonl(
                [
                    {
                        "method": "Random",
                        "seed_label_count": 8,
                        "active_label_count": 12,
                        "evaluation_label_count": 100,
                        "judge_calls": 0,
                        "train_tokens": 1000,
                        "selector_compute_seconds": 0.1,
                    },
                    {
                        "method": "Entropy",
                        "seed_label_count": 8,
                        "active_label_count": 12,
                        "guide_label_count": 5,
                        "evaluation_label_count": 100,
                        "judge_calls": 0,
                        "train_tokens": 1250,
                        "selector_compute_seconds": 0.4,
                    },
                ],
                input_path,
            )

            subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "--input_path",
                    str(input_path),
                    "--output_path",
                    str(output_path),
                    "--train_token_tolerance",
                    "100",
                ],
                check=True,
                cwd=PROJECT_ROOT,
            )

            payload = json.loads(output_path.read_text(encoding="utf-8"))

            self.assertEqual(2, payload["method_count"])
            self.assertEqual(25, payload["reports"][1]["supervision_budget_total"])
            self.assertFalse(payload["comparison"]["supervision_budget_equal"])
            self.assertFalse(payload["comparison"]["train_tokens_within_tolerance"])


if __name__ == "__main__":
    unittest.main()
