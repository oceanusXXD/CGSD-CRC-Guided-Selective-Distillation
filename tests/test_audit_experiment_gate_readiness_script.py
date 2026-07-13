from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from mias_dcms.utils import write_json


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class AuditExperimentGateReadinessScriptTest(unittest.TestCase):
    def test_script_writes_blocked_report_for_missing_real_experiment_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            evidence_path = tmp / "evidence.json"
            output_path = tmp / "gate_readiness.json"
            write_json({"protocol.freeze": "configs/mias_dcms_freeze.v1.json"}, evidence_path)

            completed = subprocess.run(
                [
                    sys.executable,
                    str(PROJECT_ROOT / "scripts" / "audit_experiment_gate_readiness.py"),
                    "--evidence_path",
                    str(evidence_path),
                    "--output_path",
                    str(output_path),
                ],
                cwd=PROJECT_ROOT,
                text=True,
                capture_output=True,
                check=False,
            )

            self.assertEqual(1, completed.returncode)
            report = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertFalse(report["is_ready"])
            self.assertEqual(["gate_0_protocol_freeze"], report["ready_gates"])
            self.assertIn("gate_8_main_results", report["blocked_gates"])

    def test_script_can_require_evidence_paths_to_exist(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            evidence_path = tmp / "evidence.json"
            output_path = tmp / "gate_readiness.json"
            write_json({"protocol.freeze": "missing_freeze.json"}, evidence_path)

            completed = subprocess.run(
                [
                    sys.executable,
                    str(PROJECT_ROOT / "scripts" / "audit_experiment_gate_readiness.py"),
                    "--evidence_path",
                    str(evidence_path),
                    "--output_path",
                    str(output_path),
                    "--require_existing_paths",
                    "--base_dir",
                    str(tmp),
                ],
                cwd=PROJECT_ROOT,
                text=True,
                capture_output=True,
                check=False,
            )

            self.assertEqual(1, completed.returncode)
            report = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertFalse(report["is_ready"])
            self.assertIn("missing_evidence_path", {issue["code"] for issue in report["issues"]})

    def test_script_returns_zero_when_all_gates_have_declared_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            evidence_path = tmp / "evidence.json"
            output_path = tmp / "gate_readiness.json"
            write_json(_complete_evidence(), evidence_path)

            completed = subprocess.run(
                [
                    sys.executable,
                    str(PROJECT_ROOT / "scripts" / "audit_experiment_gate_readiness.py"),
                    "--evidence_path",
                    str(evidence_path),
                    "--output_path",
                    str(output_path),
                ],
                cwd=PROJECT_ROOT,
                text=True,
                capture_output=True,
                check=False,
            )

            self.assertEqual(0, completed.returncode, completed.stderr)
            report = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertTrue(report["is_ready"])
            self.assertEqual(11, len(report["ready_gates"]))


def _complete_evidence() -> dict[str, str]:
    evidence_keys = [
        "protocol.freeze",
        "binary.sample_level_records",
        "binary.budget_report",
        "binary.mechanism_statistics",
        "binary.downstream_metrics",
        "multiclass.ag_news_split",
        "multiclass.trec_split",
        "multiclass.initial_logits",
        "multiclass.baseline_selection_audits",
        "multiclass.intervention_curves",
        "multiclass.intervention_statistics",
        "multiclass.propensity_identity_audit",
        "multiclass.representation_interventions",
        "preference.active_pool",
        "preference.oracle_store",
        "preference.logprobs",
        "preference.split_manifest",
        "dpo.initial_policy_checkpoint",
        "preference.baseline_scores",
        "preference.selector_sanity_audits",
        "preference.acquisition_audits",
        "preference.random_reference",
        "dpo.length_gamma_interventions",
        "dpo.selector_replacement_interventions",
        "dpo.ab_position_interventions",
        "dpo.intervention_statistics",
        "dcms.synthetic_correctness",
        "dcms.soft_group_calibration",
        "dcms.soft_group_error_audit",
        "dcms.frontier_audit",
        "dcms.matched_utility_audit",
        "main.multiclass_run_records",
        "main.dpo_run_records",
        "main.matched_utility_results",
        "main.composition_intervention_results",
        "statistics.run_metric_comparison",
        "statistics.budget_report",
        "statistics.intervention_statistics",
        "paper.freeze_pack",
        "paper.claim_audit",
        "paper.artifact_manifest",
    ]
    return {key: f"evidence/{key}.json" for key in evidence_keys}


if __name__ == "__main__":
    unittest.main()
