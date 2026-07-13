from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence


BANNED_CLAIM_PATTERNS = (
    (
        "banned_claim_first_sampling_bias",
        (
            "first to discover active learning sampling bias",
            "first discover active learning sampling bias",
            "首次发现 active learning sampling bias",
            "首次发现主动学习 sampling bias",
        ),
    ),
    (
        "banned_claim_all_active_shift_harmful",
        (
            "all active shift",
            "all active sampling shift",
            "所有 active shift 都有害",
            "所有非 iid active sampling 都有害",
        ),
    ),
    (
        "banned_claim_pool_matching_optimal",
        (
            "matching the pool is always optimal",
            "pool distribution is always optimal",
            "匹配 pool 分布必然最优",
        ),
    ),
    (
        "banned_claim_unconditional_dcms_performance",
        (
            "dcms unconditionally improves",
            "dcms always improves",
            "dcms has unconditional downstream guarantee",
            "dcms 无条件提高",
            "dcms 有下游无条件保证",
        ),
    ),
    (
        "banned_claim_uncertain_pairs_length_close",
        (
            "uncertain pairs are length-close",
            "uncertain pair are length-close",
            "不确定 pair 一定长度接近",
        ),
    ),
)


@dataclass(frozen=True)
class PaperClaimAuditReport:
    claims: list[dict[str, Any]]
    evidence: list[dict[str, Any]]
    issues: list[dict[str, Any]] = field(default_factory=list)

    @property
    def is_ready(self) -> bool:
        return not self.issues

    def as_dict(self) -> dict[str, Any]:
        return {
            "is_ready": self.is_ready,
            "claims": [dict(row) for row in self.claims],
            "evidence": [dict(row) for row in self.evidence],
            "issues": [dict(issue) for issue in self.issues],
        }


def audit_paper_claim_evidence(
    claims: Iterable[Mapping[str, Any]],
    *,
    evidence: Iterable[Mapping[str, Any]],
    required_evidence_by_claim_type: Mapping[str, Sequence[str]],
    minimum_seed_count: int = 1,
) -> PaperClaimAuditReport:
    claim_rows = [dict(row) for row in claims]
    evidence_rows = [dict(row) for row in evidence]
    evidence_by_id = {str(row.get("evidence_id", "")): row for row in evidence_rows}
    issues: list[dict[str, Any]] = []

    for claim_index, claim in enumerate(claim_rows):
        claim_id = str(claim.get("claim_id", f"claim_{claim_index}"))
        claim_type = str(claim.get("claim_type", ""))
        evidence_ids = [str(value) for value in claim.get("evidence_ids", [])]
        if not evidence_ids:
            issues.append({"code": "claim_missing_evidence", "claim_id": claim_id, "claim_type": claim_type})

        linked_evidence = []
        for evidence_id in evidence_ids:
            evidence_row = evidence_by_id.get(evidence_id)
            if evidence_row is None:
                issues.append(
                    {
                        "code": "unknown_evidence_id",
                        "claim_id": claim_id,
                        "claim_type": claim_type,
                        "evidence_id": evidence_id,
                    }
                )
                continue
            linked_evidence.append(evidence_row)
            seed_count = int(evidence_row.get("seed_count", 0))
            if seed_count < int(minimum_seed_count):
                issues.append(
                    {
                        "code": "insufficient_seed_count",
                        "claim_id": claim_id,
                        "claim_type": claim_type,
                        "evidence_id": evidence_id,
                        "seed_count": seed_count,
                        "minimum_seed_count": int(minimum_seed_count),
                    }
                )
            if "includes_failed_runs" not in evidence_row:
                issues.append(
                    {
                        "code": "evidence_missing_failed_run_policy",
                        "claim_id": claim_id,
                        "claim_type": claim_type,
                        "evidence_id": evidence_id,
                    }
                )

        observed_types = {str(row.get("artifact_type", "")) for row in linked_evidence}
        for required_type in required_evidence_by_claim_type.get(claim_type, ()):
            if str(required_type) not in observed_types:
                issues.append(
                    {
                        "code": "missing_required_evidence_type",
                        "claim_id": claim_id,
                        "claim_type": claim_type,
                        "required_type": str(required_type),
                        "observed_types": sorted(observed_types),
                    }
                )

        for text_issue in audit_paper_text_claims(str(claim.get("claim_text", ""))):
            issues.append(
                {
                    "code": text_issue["code"],
                    "claim_id": claim_id,
                    "claim_type": claim_type,
                    "matched_text": text_issue["matched_text"],
                }
            )

    return PaperClaimAuditReport(claims=claim_rows, evidence=evidence_rows, issues=issues)


def audit_paper_text_claims(text: str) -> list[dict[str, Any]]:
    lowered = str(text).lower()
    issues: list[dict[str, Any]] = []
    for code, patterns in BANNED_CLAIM_PATTERNS:
        for pattern in patterns:
            if pattern.lower() in lowered:
                issues.append({"code": code, "matched_text": pattern})
                break
    if "correlation causes" in lowered or "correlation proves causation" in lowered:
        issues.append({"code": "banned_claim_correlation_as_causation", "matched_text": "correlation"})
    return issues
