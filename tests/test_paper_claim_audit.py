from __future__ import annotations

import unittest

from mias_dcms.paper_claim_audit import (
    audit_paper_claim_evidence,
    audit_paper_text_claims,
)


class PaperClaimAuditTest(unittest.TestCase):
    def test_claims_with_required_evidence_are_ready_for_freeze(self) -> None:
        report = audit_paper_claim_evidence(
            [
                {
                    "claim_id": "dcms_frontier",
                    "claim_text": "DCMS improves the utility-coverage frontier in the specified setting.",
                    "claim_type": "dcms_algorithm",
                    "evidence_ids": ["fig2", "table2", "stats_dcms"],
                }
            ],
            evidence=[
                _evidence("fig2", artifact_type="figure", seed_count=5),
                _evidence("table2", artifact_type="table", seed_count=5),
                _evidence("stats_dcms", artifact_type="statistical_test", seed_count=5),
            ],
            required_evidence_by_claim_type={
                "dcms_algorithm": ["figure", "table", "statistical_test"],
            },
            minimum_seed_count=5,
        )

        self.assertTrue(report.is_ready)
        self.assertEqual([], report.issues)
        self.assertEqual(1, len(report.claims))

    def test_missing_required_evidence_is_reported_by_claim(self) -> None:
        report = audit_paper_claim_evidence(
            [
                {
                    "claim_id": "harmful_shift",
                    "claim_text": "Coverage shift causes downstream capability loss in this setting.",
                    "claim_type": "downstream_harm",
                    "evidence_ids": ["fig3"],
                }
            ],
            evidence=[_evidence("fig3", artifact_type="figure", seed_count=5)],
            required_evidence_by_claim_type={
                "downstream_harm": ["figure", "statistical_test", "controlled_intervention"],
            },
            minimum_seed_count=5,
        )

        self.assertFalse(report.is_ready)
        self.assertIn("missing_required_evidence_type", {issue["code"] for issue in report.issues})
        self.assertIn("statistical_test", {issue.get("required_type") for issue in report.issues})

    def test_low_seed_evidence_cannot_support_performance_claim(self) -> None:
        report = audit_paper_claim_evidence(
            [
                {
                    "claim_id": "performance_floor",
                    "claim_text": "DCMS improves worst-group performance.",
                    "claim_type": "performance",
                    "evidence_ids": ["table2", "stats"],
                }
            ],
            evidence=[
                _evidence("table2", artifact_type="table", seed_count=3),
                _evidence("stats", artifact_type="statistical_test", seed_count=3),
            ],
            required_evidence_by_claim_type={"performance": ["table", "statistical_test"]},
            minimum_seed_count=5,
        )

        self.assertFalse(report.is_ready)
        self.assertIn("insufficient_seed_count", {issue["code"] for issue in report.issues})

    def test_banned_or_overgeneralized_paper_text_is_flagged(self) -> None:
        issues = audit_paper_text_claims(
            "We are the first to discover active learning sampling bias. "
            "DCMS unconditionally improves downstream performance."
        )

        codes = {issue["code"] for issue in issues}
        self.assertIn("banned_claim_first_sampling_bias", codes)
        self.assertIn("banned_claim_unconditional_dcms_performance", codes)


def _evidence(evidence_id: str, *, artifact_type: str, seed_count: int) -> dict[str, object]:
    return {
        "evidence_id": evidence_id,
        "artifact_type": artifact_type,
        "path": f"experiments/reports/{evidence_id}.json",
        "seed_count": seed_count,
        "includes_failed_runs": True,
    }


if __name__ == "__main__":
    unittest.main()
