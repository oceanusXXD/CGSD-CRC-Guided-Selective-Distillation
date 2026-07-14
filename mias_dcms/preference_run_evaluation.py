from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

from mias_dcms.preference_evaluation import build_preference_evaluation_metrics, preference_accuracy


FIXED_HUMAN_PAIRWISE_EVALUATION_SOURCE = "heldout_fixed_human_pairwise_labels"
PREFERRED_RESPONSE_LOGPROB_PROXY = "preferred_response_logprob_proxy"


@dataclass(frozen=True)
class PreferenceRunEvaluationArtifacts:
    preference_rows: list[dict[str, Any]]
    judge_rows: list[dict[str, Any]]
    capability_rows: list[dict[str, Any]]
    aulc_rows: list[dict[str, Any]]
    metrics: dict[str, float | int]
    initial_metrics: dict[str, float | int]
    metadata: dict[str, Any]


def build_preference_run_evaluation_artifacts(
    heldout_rows: Iterable[Mapping[str, Any]],
    *,
    oracle_store: Mapping[str, Mapping[str, Any]],
    logprob_rows: Iterable[Mapping[str, Any]],
    seed_budget: int,
    active_budget: int,
    id_field: str = "sample_id",
    group_field: str = "length_gap_bin",
) -> PreferenceRunEvaluationArtifacts:
    """Materialize post-training held-out pairwise evaluation from auditable log-probs.

    The resulting win-rate rows are explicitly tied to held-out human preference
    labels; they are not a substitute for an external generation judge.
    """
    if int(seed_budget) < 0 or int(active_budget) <= 0:
        raise ValueError("seed_budget must be non-negative and active_budget must be positive")

    pool_by_id = _rows_by_id(heldout_rows, id_field=id_field, source_name="heldout pool")
    logprobs_by_id = _rows_by_id(logprob_rows, id_field=id_field, source_name="heldout logprobs")
    missing_logprobs = sorted(set(pool_by_id) - set(logprobs_by_id))
    extra_logprobs = sorted(set(logprobs_by_id) - set(pool_by_id))
    if missing_logprobs or extra_logprobs:
        details: list[str] = []
        if missing_logprobs:
            details.append(f"missing={len(missing_logprobs)} example={missing_logprobs[0]!r}")
        if extra_logprobs:
            details.append(f"extra={len(extra_logprobs)} example={extra_logprobs[0]!r}")
        raise ValueError("heldout logprobs must exactly cover the heldout pool: " + "; ".join(details))

    preference_rows: list[dict[str, Any]] = []
    capability_rows: list[dict[str, Any]] = []
    excluded_tie_ids: list[str] = []
    for sample_id, pool_row in pool_by_id.items():
        if sample_id not in oracle_store:
            raise ValueError(f"heldout oracle store is missing sample id {sample_id!r}")
        oracle_label = _preference_label(oracle_store[sample_id].get("preference_label"))
        if oracle_label == "tie":
            excluded_tie_ids.append(sample_id)
            continue
        logprob_row = logprobs_by_id[sample_id]
        policy_margin = _margin(logprob_row, "policy_logprob_response_1", "policy_logprob_response_2")
        reference_margin = _margin(logprob_row, "reference_logprob_response_1", "reference_logprob_response_2")
        if group_field not in pool_row:
            raise ValueError(f"heldout row {sample_id!r} is missing group field {group_field!r}")
        preference_rows.append(
            {
                "sample_id": sample_id,
                "oracle_preference": oracle_label,
                "predicted_preference": "A" if policy_margin >= 0.0 else "B",
                "reference_predicted_preference": "A" if reference_margin >= 0.0 else "B",
                "observable_group": str(pool_row[group_field]),
                "length_gap_bin": str(pool_row.get("length_gap_bin", pool_row[group_field])),
                "policy_preference_margin": policy_margin,
                "reference_preference_margin": reference_margin,
                "evaluation_source": FIXED_HUMAN_PAIRWISE_EVALUATION_SOURCE,
            }
        )
        preferred_suffix = "1" if oracle_label == "A" else "2"
        capability_rows.append(
            {
                "sample_id": sample_id,
                "baseline_score": float(logprob_row[f"reference_logprob_response_{preferred_suffix}"]),
                "policy_score": float(logprob_row[f"policy_logprob_response_{preferred_suffix}"]),
                "capability_definition": PREFERRED_RESPONSE_LOGPROB_PROXY,
            }
        )

    if not preference_rows:
        raise ValueError("heldout evaluation has no non-tie preference rows")
    judge_rows = [
        {
            "sample_id": row["sample_id"],
            "length_gap_bin": row["length_gap_bin"],
            "judge_win": int(row["predicted_preference"] == row["oracle_preference"]),
            "judge_source": FIXED_HUMAN_PAIRWISE_EVALUATION_SOURCE,
            "is_generation_judge": False,
        }
        for row in preference_rows
    ]
    initial_rows = [
        {
            **row,
            "predicted_preference": row["reference_predicted_preference"],
        }
        for row in preference_rows
    ]
    initial_accuracy = preference_accuracy(initial_rows)
    final_accuracy = preference_accuracy(preference_rows)
    aulc_rows = [
        {"budget": int(seed_budget), "performance": initial_accuracy},
        {"budget": int(seed_budget) + int(active_budget), "performance": final_accuracy},
    ]
    metrics = build_preference_evaluation_metrics(
        preference_rows=preference_rows,
        judge_rows=judge_rows,
        capability_rows=capability_rows,
        aulc_rows=aulc_rows,
        group_field="observable_group",
        length_bin_field="length_gap_bin",
    )
    return PreferenceRunEvaluationArtifacts(
        preference_rows=preference_rows,
        judge_rows=judge_rows,
        capability_rows=capability_rows,
        aulc_rows=aulc_rows,
        metrics=metrics,
        initial_metrics={
            "preference_accuracy": initial_accuracy,
            "preference_eval_count": len(initial_rows),
        },
        metadata={
            "evaluation_source": FIXED_HUMAN_PAIRWISE_EVALUATION_SOURCE,
            "generation_judge_available": False,
            "capability_evaluation_available": True,
            "capability_definition": PREFERRED_RESPONSE_LOGPROB_PROXY,
            "excluded_tie_count": len(excluded_tie_ids),
            "excluded_tie_ids": excluded_tie_ids,
            "capability_proxy_definition": PREFERRED_RESPONSE_LOGPROB_PROXY,
        },
    )


def _rows_by_id(
    rows: Iterable[Mapping[str, Any]],
    *,
    id_field: str,
    source_name: str,
) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for row in rows:
        payload = dict(row)
        value = payload.get(id_field, payload.get("id"))
        if value is None:
            raise ValueError(f"{source_name} row is missing {id_field!r}")
        sample_id = str(value)
        if sample_id in indexed:
            raise ValueError(f"{source_name} contains duplicate sample id {sample_id!r}")
        indexed[sample_id] = payload
    if not indexed:
        raise ValueError(f"{source_name} must not be empty")
    return indexed


def _preference_label(value: Any) -> str:
    if value in {"A", "a", 1, "1", "response_a", "response_1"}:
        return "A"
    if value in {"B", "b", 2, "2", "response_b", "response_2"}:
        return "B"
    if value in {"tie", "Tie", "TIE", 0, "0", None}:
        return "tie"
    raise ValueError(f"unsupported preference label: {value!r}")


def _margin(row: Mapping[str, Any], first_field: str, second_field: str) -> float:
    try:
        return float(row[first_field]) - float(row[second_field])
    except KeyError as exc:
        raise ValueError(f"heldout logprob row is missing {exc.args[0]!r}") from exc
