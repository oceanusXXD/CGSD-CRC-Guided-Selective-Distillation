from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path
import unittest

from mias_dcms.utils import write_jsonl


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = PROJECT_ROOT / "scripts" / "audit_soft_group_error.py"


class AuditSoftGroupErrorScriptTest(unittest.TestCase):
    def test_script_writes_nominal_vs_robust_observed_coverage_audit(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            input_path = tmp / "soft_group_candidates.jsonl"
            output_path = tmp / "soft_group_error.json"
            write_jsonl(
                [
                    {
                        "sample_id": "s1",
                        "score": 1.0,
                        "groups": {"A": 0.5},
                        "membership_lower": {"A": 0.8},
                        "membership_upper": {"A": 1.0},
                        "observed_membership": {"A": 1.0},
                    },
                    {
                        "sample_id": "s2",
                        "score": 0.9,
                        "groups": {"A": 0.5},
                        "membership_lower": {"A": 0.8},
                        "membership_upper": {"A": 1.0},
                        "observed_membership": {"A": 1.0},
                    },
                    {
                        "sample_id": "s3",
                        "score": 0.8,
                        "groups": {"A": 0.5},
                        "membership_lower": {"A": 0.0},
                        "membership_upper": {"A": 0.2},
                        "observed_membership": {"A": 0.0},
                    },
                    {
                        "sample_id": "s4",
                        "score": 0.1,
                        "groups": {"A": 0.5},
                        "membership_lower": {"A": 0.0},
                        "membership_upper": {"A": 0.2},
                        "observed_membership": {"A": 0.0},
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
                    "--budget",
                    "2",
                    "--target_moments",
                    "{\"A\": 0.5}",
                    "--tolerance",
                    "0.0",
                ],
                check=True,
                cwd=PROJECT_ROOT,
            )

            payload = json.loads(output_path.read_text(encoding="utf-8"))

            self.assertEqual(["s1", "s2"], payload["nominal"]["selected_ids"])
            self.assertEqual(["s1", "s3"], payload["robust"]["selected_ids"])
            self.assertAlmostEqual(0.5, payload["nominal"]["observed_max_constraint_violation"])
            self.assertAlmostEqual(0.0, payload["robust"]["observed_max_constraint_violation"])
            self.assertTrue(payload["robust_improves_observed_coverage"])


if __name__ == "__main__":
    unittest.main()
