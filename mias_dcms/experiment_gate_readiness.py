from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ExperimentGateSpec:
    gate_id: str
    title: str
    required_evidence: tuple[str, ...]


@dataclass(frozen=True)
class ExperimentGateReadinessReport:
    gates: dict[str, dict[str, Any]]
    ready_gates: list[str] = field(default_factory=list)
    blocked_gates: list[str] = field(default_factory=list)
    issues: list[dict[str, Any]] = field(default_factory=list)

    @property
    def is_ready(self) -> bool:
        return not self.blocked_gates and not self.issues

    def as_dict(self) -> dict[str, Any]:
        return {
            "is_ready": self.is_ready,
            "gate_count": len(self.gates),
            "ready_gate_count": len(self.ready_gates),
            "blocked_gate_count": len(self.blocked_gates),
            "missing_evidence_count": sum(
                len(gate["missing_evidence"]) for gate in self.gates.values()
            ),
            "ready_gates": list(self.ready_gates),
            "blocked_gates": list(self.blocked_gates),
            "gates": {gate_id: dict(gate) for gate_id, gate in self.gates.items()},
            "issues": [dict(issue) for issue in self.issues],
        }


GATE_SPECS: tuple[ExperimentGateSpec, ...] = (
    ExperimentGateSpec(
        gate_id="gate_0_protocol_freeze",
        title="Gate 0: task and protocol freeze",
        required_evidence=("protocol.freeze",),
    ),
    ExperimentGateSpec(
        gate_id="gate_1_binary_reaudit",
        title="Gate 1: legacy binary reaudit",
        required_evidence=(
            "binary.sample_level_records",
            "binary.budget_report",
            "binary.mechanism_statistics",
            "binary.downstream_metrics",
        ),
    ),
    ExperimentGateSpec(
        gate_id="gate_2_multiclass_environment",
        title="Gate 2: multiclass fixed environment",
        required_evidence=(
            "multiclass.ag_news_split",
            "multiclass.trec_split",
            "multiclass.initial_logits",
            "multiclass.baseline_selection_audits",
        ),
    ),
    ExperimentGateSpec(
        gate_id="gate_3_multiclass_mias",
        title="Gate 3: multiclass MIAS causal identification",
        required_evidence=(
            "multiclass.intervention_curves",
            "multiclass.intervention_statistics",
            "multiclass.propensity_identity_audit",
            "multiclass.representation_interventions",
        ),
    ),
    ExperimentGateSpec(
        gate_id="gate_4_preference_fixed_pool",
        title="Gate 4: preference fixed pool and label isolation",
        required_evidence=(
            "preference.active_pool",
            "preference.oracle_store",
            "preference.logprobs",
            "preference.split_manifest",
            "dpo.initial_policy_checkpoint",
        ),
    ),
    ExperimentGateSpec(
        gate_id="gate_5_preference_baselines",
        title="Gate 5: active preference baselines",
        required_evidence=(
            "preference.baseline_scores",
            "preference.selector_sanity_audits",
            "preference.acquisition_audits",
            "preference.random_reference",
        ),
    ),
    ExperimentGateSpec(
        gate_id="gate_6_dpo_mias",
        title="Gate 6: DPO-side MIAS causal identification",
        required_evidence=(
            "dpo.length_gamma_interventions",
            "dpo.selector_replacement_interventions",
            "dpo.ab_position_interventions",
            "dpo.intervention_statistics",
        ),
    ),
    ExperimentGateSpec(
        gate_id="gate_7_dcms_correctness",
        title="Gate 7: DCMS algorithm correctness",
        required_evidence=(
            "dcms.synthetic_correctness",
            "dcms.soft_group_calibration",
            "dcms.soft_group_error_audit",
            "dcms.frontier_audit",
            "dcms.matched_utility_audit",
        ),
    ),
    ExperimentGateSpec(
        gate_id="gate_8_main_results",
        title="Gate 8: downstream causal and main results",
        required_evidence=(
            "main.multiclass_run_records",
            "main.dpo_run_records",
            "main.matched_utility_results",
            "main.composition_intervention_results",
        ),
    ),
    ExperimentGateSpec(
        gate_id="gate_9_statistics_fairness",
        title="Gate 9: statistics and fairness",
        required_evidence=(
            "statistics.run_metric_comparison",
            "statistics.budget_report",
            "statistics.intervention_statistics",
        ),
    ),
    ExperimentGateSpec(
        gate_id="gate_10_paper_claim_freeze",
        title="Gate 10: paper figures, tables, and claims",
        required_evidence=(
            "paper.freeze_pack",
            "paper.claim_audit",
            "paper.artifact_manifest",
        ),
    ),
)


def audit_experiment_gate_readiness(
    evidence_paths: Mapping[str, Any],
    *,
    gate_specs: Sequence[ExperimentGateSpec] = GATE_SPECS,
    require_existing_paths: bool = False,
    base_dir: str | Path | None = None,
) -> ExperimentGateReadinessReport:
    present_evidence, issues = _present_evidence(
        evidence_paths,
        require_existing_paths=require_existing_paths,
        base_dir=base_dir,
    )
    gates: dict[str, dict[str, Any]] = {}
    ready_gates: list[str] = []
    blocked_gates: list[str] = []

    for spec in gate_specs:
        missing = [
            evidence_key
            for evidence_key in spec.required_evidence
            if evidence_key not in present_evidence
        ]
        is_ready = not missing
        status = "ready" if is_ready else "blocked"
        gates[spec.gate_id] = {
            "gate_id": spec.gate_id,
            "title": spec.title,
            "status": status,
            "is_ready": is_ready,
            "required_evidence": list(spec.required_evidence),
            "present_evidence": [
                evidence_key
                for evidence_key in spec.required_evidence
                if evidence_key in present_evidence
            ],
            "missing_evidence": missing,
        }
        if is_ready:
            ready_gates.append(spec.gate_id)
        else:
            blocked_gates.append(spec.gate_id)

    return ExperimentGateReadinessReport(
        gates=gates,
        ready_gates=ready_gates,
        blocked_gates=blocked_gates,
        issues=issues,
    )


def _present_evidence(
    evidence_paths: Mapping[str, Any],
    *,
    require_existing_paths: bool,
    base_dir: str | Path | None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    present: dict[str, Any] = {}
    issues: list[dict[str, Any]] = []
    for key, value in evidence_paths.items():
        evidence_key = str(key)
        if not _evidence_value_is_present(value):
            continue
        if require_existing_paths and isinstance(value, str):
            if not _evidence_path_exists(value, base_dir=base_dir):
                issues.append(
                    {
                        "code": "missing_evidence_path",
                        "evidence_key": evidence_key,
                        "path": value,
                    }
                )
                continue
        present[evidence_key] = value
    return present, issues


def _evidence_value_is_present(value: Any) -> bool:
    if value is None or value is False:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    return True


def _evidence_path_exists(value: str, *, base_dir: str | Path | None) -> bool:
    path = Path(value)
    if path.is_absolute():
        return path.exists()
    if base_dir is not None:
        return (Path(base_dir) / path).exists()
    return path.exists()
