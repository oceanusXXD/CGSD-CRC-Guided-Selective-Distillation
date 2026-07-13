from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from mias_dcms.utils import write_json


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class AuditPaperClaimsScriptTest(unittest.TestCase):
    def test_script_writes_ready_claim_audit_report(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            claims_path = tmp / "claims.json"
            evidence_path = tmp / "evidence.json"
            requirements_path = tmp / "requirements.json"
            output_path = tmp / "report.json"
            write_json(
                [
                    {
                        "claim_id": "dcms_frontier",
                        "claim_text": "DCMS improves the utility-coverage frontier in the specified setting.",
                        "claim_type": "dcms_algorithm",
                        "evidence_ids": ["fig2", "table2", "stats"],
                    }
                ],
                claims_path,
            )
            write_json(
                [
                    _evidence("fig2", "figure"),
                    _evidence("table2", "table"),
                    _evidence("stats", "statistical_test"),
                ],
                evidence_path,
            )
            write_json({"dcms_algorithm": ["figure", "table", "statistical_test"]}, requirements_path)

            completed = subprocess.run(
                [
                    sys.executable,
                    str(PROJECT_ROOT / "scripts" / "audit_paper_claims.py"),
                    "--claims_path",
                    str(claims_path),
                    "--evidence_path",
                    str(evidence_path),
                    "--requirements_path",
                    str(requirements_path),
                    "--output_path",
                    str(output_path),
                    "--minimum_seed_count",
                    "5",
                ],
                cwd=PROJECT_ROOT,
                text=True,
                capture_output=True,
                check=False,
            )

            self.assertEqual(0, completed.returncode, completed.stderr)
            payload = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertTrue(payload["is_ready"])
            self.assertEqual([], payload["issues"])

    def test_script_returns_nonzero_for_banned_text(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            claims_path = tmp / "claims.json"
            evidence_path = tmp / "evidence.json"
            requirements_path = tmp / "requirements.json"
            text_path = tmp / "paper.txt"
            output_path = tmp / "report.json"
            write_json([], claims_path)
            write_json([], evidence_path)
            write_json({}, requirements_path)
            text_path.write_text("DCMS unconditionally improves downstream performance.", encoding="utf-8")

            completed = subprocess.run(
                [
                    sys.executable,
                    str(PROJECT_ROOT / "scripts" / "audit_paper_claims.py"),
                    "--claims_path",
                    str(claims_path),
                    "--evidence_path",
                    str(evidence_path),
                    "--requirements_path",
                    str(requirements_path),
                    "--paper_text_path",
                    str(text_path),
                    "--output_path",
                    str(output_path),
                ],
                cwd=PROJECT_ROOT,
                text=True,
                capture_output=True,
                check=False,
            )

            self.assertEqual(1, completed.returncode)
            payload = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertIn("banned_claim_unconditional_dcms_performance", {issue["code"] for issue in payload["issues"]})


def _evidence(evidence_id: str, artifact_type: str) -> dict[str, object]:
    return {
        "evidence_id": evidence_id,
        "artifact_type": artifact_type,
        "path": f"experiments/reports/{evidence_id}.json",
        "seed_count": 5,
        "includes_failed_runs": True,
    }


if __name__ == "__main__":
    unittest.main()
