from __future__ import annotations

import unittest

from mias_dcms.preference_logprob_audit import audit_preference_logprobs


class PreferenceLogprobAuditTest(unittest.TestCase):
    def test_audit_adds_policy_reference_gaps_and_implicit_margins(self) -> None:
        rows = [
            {
                "sample_id": "p1",
                "policy_logprob_response_1": -1.0,
                "policy_logprob_response_2": -2.0,
                "reference_logprob_response_1": -1.5,
                "reference_logprob_response_2": -1.1,
            },
            {
                "sample_id": "p2",
                "policy_logprob_response_1": -2.2,
                "policy_logprob_response_2": -1.8,
                "reference_logprob_response_1": -2.0,
                "reference_logprob_response_2": -1.7,
            },
        ]

        audited_rows, summary = audit_preference_logprobs(rows)

        self.assertEqual(2, len(audited_rows))
        self.assertAlmostEqual(1.0, audited_rows[0]["policy_logprob_gap"])
        self.assertAlmostEqual(-0.4, audited_rows[0]["reference_logprob_gap"])
        self.assertAlmostEqual(1.4, audited_rows[0]["implicit_reward_gap"])
        self.assertAlmostEqual(1.4, audited_rows[0]["absolute_implicit_margin"])
        self.assertAlmostEqual(-0.4, audited_rows[1]["policy_logprob_gap"])
        self.assertAlmostEqual(-0.3, audited_rows[1]["reference_logprob_gap"])
        self.assertAlmostEqual(-0.1, audited_rows[1]["implicit_reward_gap"])
        self.assertAlmostEqual(0.1, audited_rows[1]["absolute_implicit_margin"])
        self.assertEqual(2, summary["row_count"])
        self.assertEqual(2, summary["finite_row_count"])
        self.assertTrue(summary["implicit_margin_not_all_zero"])
        self.assertGreater(summary["policy_gap_variance"], 0.0)
        self.assertGreater(summary["reference_gap_variance"], 0.0)
        self.assertEqual({"positive": 1, "negative": 1, "zero": 0}, summary["implicit_reward_gap_sign_counts"])

    def test_audit_rejects_missing_logprob_fields(self) -> None:
        rows = [
            {
                "sample_id": "p1",
                "policy_logprob_response_1": -1.0,
                "policy_logprob_response_2": -2.0,
                "reference_logprob_response_1": -1.5,
            }
        ]

        with self.assertRaisesRegex(ValueError, "missing logprob fields"):
            audit_preference_logprobs(rows)

    def test_audit_rejects_non_finite_logprobs(self) -> None:
        rows = [
            {
                "sample_id": "p1",
                "policy_logprob_response_1": float("nan"),
                "policy_logprob_response_2": -2.0,
                "reference_logprob_response_1": -1.5,
                "reference_logprob_response_2": -1.1,
            }
        ]

        with self.assertRaisesRegex(ValueError, "non-finite logprob"):
            audit_preference_logprobs(rows)

    def test_audit_rejects_all_zero_implicit_margins_by_default(self) -> None:
        rows = [
            {
                "sample_id": "p1",
                "policy_logprob_response_1": -1.0,
                "policy_logprob_response_2": -2.0,
                "reference_logprob_response_1": -1.0,
                "reference_logprob_response_2": -2.0,
            }
        ]

        with self.assertRaisesRegex(ValueError, "implicit margin is zero for every row"):
            audit_preference_logprobs(rows)


if __name__ == "__main__":
    unittest.main()
