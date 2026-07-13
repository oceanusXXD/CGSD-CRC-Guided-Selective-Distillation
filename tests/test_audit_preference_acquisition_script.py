from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = PROJECT_ROOT / "scripts" / "audit_preference_acquisition.py"


class AuditPreferenceAcquisitionScriptTest(unittest.TestCase):
    def test_script_writes_preference_acquisition_audit_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            membership_path = tmp / "membership.jsonl"
            random_path = tmp / "random_membership.jsonl"
            output_path = tmp / "acquisition_audit.json"
            rows = [
                {"sample_id": "p1", "selected": 1, "length_gap_bin": "short", "source_pair": "human|model"},
                {"sample_id": "p2", "selected": 1, "length_gap_bin": "short", "source_pair": "human|model"},
                {"sample_id": "p3", "selected": 0, "length_gap_bin": "long", "source_pair": "model|human"},
                {"sample_id": "p4", "selected": 0, "length_gap_bin": "long", "source_pair": "model|human"},
            ]
            random_rows = [
                {**rows[0], "selected": 1},
                {**rows[1], "selected": 0},
                {**rows[2], "selected": 1},
                {**rows[3], "selected": 0},
            ]
            _write_jsonl(membership_path, rows)
            _write_jsonl(random_path, random_rows)

            subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "--input_path",
                    str(membership_path),
                    "--output_path",
                    str(output_path),
                    "--method",
                    "APL",
                    "--group_fields",
                    "length_gap_bin,source_pair",
                    "--random_reference_path",
                    str(random_path),
                ],
                check=True,
                cwd=PROJECT_ROOT,
            )

            payload = json.loads(output_path.read_text(encoding="utf-8"))

            self.assertEqual(str(membership_path), payload["input_path"])
            self.assertEqual(str(random_path), payload["random_reference_path"])
            self.assertEqual("APL", payload["method"])
            self.assertEqual(2, payload["selected_size"])
            self.assertAlmostEqual(0.5, payload["by_group_field"]["length_gap_bin"]["acquisition_tv"])
            self.assertTrue(payload["random_reference_present"])


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


if __name__ == "__main__":
    unittest.main()
