from __future__ import annotations

import unittest

from mias_dcms.soft_group_error import soft_group_error_audit


class SoftGroupErrorAuditTest(unittest.TestCase):
    def test_soft_group_error_audit_compares_nominal_and_robust_observed_coverage(self) -> None:
        audit = soft_group_error_audit(
            sample_ids=["s1", "s2", "s3", "s4"],
            utilities=[1.0, 0.9, 0.8, 0.1],
            group_membership=[
                {"A": 0.5},
                {"A": 0.5},
                {"A": 0.5},
                {"A": 0.5},
            ],
            membership_lower=[
                {"A": 0.8},
                {"A": 0.8},
                {"A": 0.0},
                {"A": 0.0},
            ],
            membership_upper=[
                {"A": 1.0},
                {"A": 1.0},
                {"A": 0.2},
                {"A": 0.2},
            ],
            observed_membership=[
                {"A": 1.0},
                {"A": 1.0},
                {"A": 0.0},
                {"A": 0.0},
            ],
            budget=2,
            target_moments={"A": 0.5},
            tolerance=0.0,
        )

        self.assertEqual(["s1", "s2"], audit.nominal.selected_ids)
        self.assertEqual(["s1", "s3"], audit.robust.selected_ids)
        self.assertAlmostEqual(0.5, audit.nominal.observed_max_constraint_violation)
        self.assertAlmostEqual(0.0, audit.robust.observed_max_constraint_violation)
        self.assertAlmostEqual(0.5, audit.observed_violation_delta)
        self.assertTrue(audit.robust_improves_observed_coverage)


if __name__ == "__main__":
    unittest.main()
