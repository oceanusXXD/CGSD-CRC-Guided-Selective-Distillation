from __future__ import annotations

import json
import unittest
from pathlib import Path


FREEZE_PATH = Path("configs/mias_dcms_freeze.v1.json")


class ProtocolFreezeTest(unittest.TestCase):
    def test_freeze_protocol_declares_main_tasks_and_datasets(self) -> None:
        protocol = json.loads(FREEZE_PATH.read_text(encoding="utf-8"))

        self.assertEqual("Active Preference Acquisition", protocol["tasks"]["primary"])
        self.assertEqual("Multi-class Active Distillation", protocol["tasks"]["controlled_validation"])
        self.assertEqual("Binary Selective Distillation", protocol["tasks"]["legacy_evidence"])
        self.assertEqual(["HelpSteer2-Preference", "TL;DR human comparisons"], protocol["datasets"]["preference_main"])
        self.assertEqual(["AG News", "TREC"], protocol["datasets"]["multiclass_main"])
        self.assertEqual(4, protocol["datasets"]["legacy_binary"]["source_count"])
        self.assertEqual(7, protocol["datasets"]["legacy_binary"]["predicate_count"])

    def test_freeze_protocol_declares_baselines_metrics_and_budget_rules(self) -> None:
        protocol = json.loads(FREEZE_PATH.read_text(encoding="utf-8"))

        self.assertIn("Random", protocol["baselines"]["preference"])
        self.assertIn("ActiveDPO+DCMS", protocol["baselines"]["preference"])
        self.assertIn("BADGE+DCMS", protocol["baselines"]["multiclass"])
        self.assertIn("acquisition_tv", protocol["metrics"]["selection"])
        self.assertIn("maximum_propensity_ratio", protocol["metrics"]["selection"])
        self.assertIn("length_controlled_win_rate", protocol["metrics"]["preference"])
        self.assertEqual("B_total = B_seed + sum(B_active_rounds)", protocol["budget"]["total_formula"])
        self.assertTrue(protocol["budget"]["seed_labels_counted"])
        self.assertTrue(protocol["budget"]["group_estimator_labels_counted"])
        self.assertTrue(protocol["budget"]["evaluation_labels_reported_separately"])

    def test_freeze_protocol_records_label_isolation_and_immutability(self) -> None:
        protocol = json.loads(FREEZE_PATH.read_text(encoding="utf-8"))

        forbidden = set(protocol["label_isolation"]["selector_forbidden_fields"])
        self.assertIn("chosen", forbidden)
        self.assertIn("rejected", forbidden)
        self.assertIn("preference_strength", forbidden)
        self.assertIn("oracle_label", forbidden)
        self.assertIn("datasets", protocol["frozen_before_main_experiments"])
        self.assertIn("target_moments", protocol["frozen_before_main_experiments"])
        self.assertIn("slack_grid", protocol["frozen_before_main_experiments"])


if __name__ == "__main__":
    unittest.main()
