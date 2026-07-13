from __future__ import annotations

import unittest

from mias_dcms.auditing.mias import (
    acquisition_tv,
    maximum_propensity_ratio,
    mias_selection_audit,
    propensity_identity_report,
)


class MIASAuditTest(unittest.TestCase):
    def test_propensity_identity_predicts_selected_distribution(self) -> None:
        rows = [
            {"id": "a1", "group": "A", "selected": True},
            {"id": "a2", "group": "A", "selected": False},
            {"id": "b1", "group": "B", "selected": True},
            {"id": "b2", "group": "B", "selected": True},
        ]

        report = propensity_identity_report(rows, group_field="group", selected_field="selected")

        self.assertEqual(4, report.pool_size)
        self.assertEqual(3, report.selected_size)
        self.assertAlmostEqual(0.5, report.groups["A"].pool_share)
        self.assertAlmostEqual(1 / 3, report.groups["A"].actual_selected_share)
        self.assertAlmostEqual(1 / 3, report.groups["A"].predicted_selected_share)
        self.assertAlmostEqual(0.0, report.total_absolute_prediction_error)

    def test_acquisition_tv_measures_pool_to_selected_shift(self) -> None:
        rows = [
            {"id": "a1", "group": "A", "selected": True},
            {"id": "a2", "group": "A", "selected": False},
            {"id": "b1", "group": "B", "selected": True},
            {"id": "b2", "group": "B", "selected": True},
        ]

        tv = acquisition_tv(rows, group_field="group", selected_field="selected")

        self.assertAlmostEqual(1 / 6, tv)

    def test_maximum_propensity_ratio_ignores_zero_zero_groups(self) -> None:
        rows = [
            {"id": "a1", "group": "A", "selected": True},
            {"id": "a2", "group": "A", "selected": False},
            {"id": "b1", "group": "B", "selected": True},
            {"id": "b2", "group": "B", "selected": True},
            {"id": "c1", "group": "C", "selected": False},
        ]

        ratio = maximum_propensity_ratio(rows, group_field="group", selected_field="selected")

        self.assertEqual(2.0, ratio)

    def test_mias_selection_audit_collects_required_selection_metrics(self) -> None:
        rows = [
            {"id": "a1", "group": "A", "selected": True},
            {"id": "a2", "group": "A", "selected": False},
            {"id": "b1", "group": "B", "selected": True},
            {"id": "b2", "group": "B", "selected": True},
        ]

        audit = mias_selection_audit(rows, group_field="group", selected_field="selected")
        as_dict = audit.as_dict()

        self.assertEqual(4, as_dict["pool_size"])
        self.assertEqual(3, as_dict["selected_size"])
        self.assertAlmostEqual(1 / 6, as_dict["acquisition_tv"])
        self.assertEqual(2.0, as_dict["maximum_propensity_ratio"])
        self.assertEqual(0.0, as_dict["total_absolute_prediction_error"])
        self.assertIn("A", as_dict["groups"])


if __name__ == "__main__":
    unittest.main()
