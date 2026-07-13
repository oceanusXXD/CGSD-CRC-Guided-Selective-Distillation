from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from mias_dcms.experiment_gate_readiness import audit_experiment_gate_readiness


class ExperimentGateReadinessTest(unittest.TestCase):
    def test_reports_ready_gate_when_all_required_evidence_exists(self) -> None:
        report = audit_experiment_gate_readiness(
            evidence_paths={
                "protocol.freeze": True,
                "preference.active_pool": True,
                "preference.oracle_store": True,
                "preference.logprobs": True,
                "preference.split_manifest": True,
                "dpo.initial_policy_checkpoint": True,
            }
        )

        gate4 = report.gates["gate_4_preference_fixed_pool"]

        self.assertTrue(gate4["is_ready"])
        self.assertEqual("ready", gate4["status"])
        self.assertEqual([], gate4["missing_evidence"])
        self.assertIn("dpo.initial_policy_checkpoint", gate4["required_evidence"])
        self.assertIn("gate_4_preference_fixed_pool", report.ready_gates)

    def test_keeps_real_experiment_gates_incomplete_without_evidence(self) -> None:
        report = audit_experiment_gate_readiness(evidence_paths={"protocol.freeze": True})

        self.assertFalse(report.is_ready)
        self.assertEqual(["gate_0_protocol_freeze"], report.ready_gates)
        self.assertIn("gate_4_preference_fixed_pool", report.blocked_gates)
        self.assertIn(
            "preference.active_pool",
            report.gates["gate_4_preference_fixed_pool"]["missing_evidence"],
        )
        self.assertIn(
            "dpo.initial_policy_checkpoint",
            report.gates["gate_4_preference_fixed_pool"]["missing_evidence"],
        )

    def test_accepts_path_like_evidence_values_and_preserves_stage_order(self) -> None:
        report = audit_experiment_gate_readiness(
            evidence_paths={
                "protocol.freeze": "configs/mias_dcms_freeze.v1.json",
                "binary.sample_level_records": "",
                "binary.budget_report": None,
            }
        )

        self.assertEqual(
            [
                "gate_0_protocol_freeze",
                "gate_1_binary_reaudit",
                "gate_2_multiclass_environment",
                "gate_3_multiclass_mias",
                "gate_4_preference_fixed_pool",
                "gate_5_preference_baselines",
                "gate_6_dpo_mias",
                "gate_7_dcms_correctness",
                "gate_8_main_results",
                "gate_9_statistics_fairness",
                "gate_10_paper_claim_freeze",
            ],
            list(report.gates),
        )
        self.assertTrue(report.gates["gate_0_protocol_freeze"]["is_ready"])
        self.assertFalse(report.gates["gate_1_binary_reaudit"]["is_ready"])
        self.assertEqual("blocked", report.gates["gate_1_binary_reaudit"]["status"])

    def test_can_require_declared_path_evidence_to_exist(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            existing = tmp / "freeze.json"
            existing.write_text("{}", encoding="utf-8")

            report = audit_experiment_gate_readiness(
                evidence_paths={
                    "protocol.freeze": "freeze.json",
                    "preference.active_pool": "missing_active_pool.jsonl",
                },
                require_existing_paths=True,
                base_dir=tmp,
            )

        self.assertTrue(report.gates["gate_0_protocol_freeze"]["is_ready"])
        self.assertIn(
            "preference.active_pool",
            report.gates["gate_4_preference_fixed_pool"]["missing_evidence"],
        )
        self.assertIn("missing_evidence_path", {issue["code"] for issue in report.issues})


if __name__ == "__main__":
    unittest.main()
