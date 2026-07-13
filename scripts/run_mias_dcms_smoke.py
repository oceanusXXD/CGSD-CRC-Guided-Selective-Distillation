from __future__ import annotations

import argparse
import math
import random
from pathlib import Path
import sys
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mias_dcms.preference_acquisition_audit import audit_preference_acquisition
from mias_dcms.preference_evaluation import build_preference_evaluation_metrics
from mias_dcms.preference_intervention_audit import audit_ab_position_intervention
from mias_dcms.preference_intervention_inputs import build_preference_intervention_rows
from mias_dcms.preference_pool import build_preference_fixed_pool
from mias_dcms.preference_reveal import reveal_selected_preference_labels
from mias_dcms.preference_scoring import build_preference_baseline_score_rows
from mias_dcms.preference_split_manifest import build_preference_split_manifest
from mias_dcms.selectors import random_without_replacement, select_top_budget
from mias_dcms.selection.dcms import solve_dcms_with_slack
from mias_dcms.utils import write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a deterministic CPU MIAS/DCMS synthetic smoke experiment.")
    parser.add_argument(
        "--output_path",
        type=Path,
        default=PROJECT_ROOT / "experiments/reports/smoke_mias_dcms.current.json",
    )
    parser.add_argument("--seed", type=int, default=20260713)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rng = random.Random(int(args.seed))
    raw_rows = _synthetic_preference_rows(rng, count=60)
    fixed_pool = build_preference_fixed_pool(raw_rows, seed=int(args.seed))
    split_manifest = build_preference_split_manifest(
        fixed_pool.active_pool,
        seed=int(args.seed),
        seed_size=10,
        active_size=40,
        heldout_size=5,
        test_size=5,
    )

    logprob_rows = [_synthetic_logprob_row(row, index=index) for index, row in enumerate(fixed_pool.active_pool)]
    scored_input = [
        {
            **row,
            **logprob,
            "prompt_cluster_probabilities": [0.8, 0.2] if index % 3 else [0.5, 0.5],
            "prompt_cluster": "c0" if index % 2 else "c1",
            "length_gap_bin": _length_gap_bin(float(row["length_gap"])),
        }
        for index, (row, logprob) in enumerate(zip(fixed_pool.active_pool, logprob_rows, strict=True))
    ]
    scored_rows = build_preference_baseline_score_rows(
        scored_input,
        methods=("reward_margin", "apl", "active_dpo"),
        active_dpo_length_normalize=True,
        active_dpo_novelty_weight=0.25,
    )
    budget = 10
    sample_ids = [str(row["sample_id"]) for row in scored_rows]
    random_ids = random_without_replacement(sample_ids, budget=budget, seed=int(args.seed))
    active_dpo_ids = select_top_budget(
        sample_ids=sample_ids,
        scores=[float(row["active_dpo_score"]) for row in scored_rows],
        budget=budget,
    )
    candidate_rows = [
        {
            "sample_id": row["sample_id"],
            "score": row["active_dpo_score"],
            "groups": {
                f"length_gap_bin={row['length_gap_bin']}": 1.0,
                f"ab_position={row['ab_position']}": 1.0,
            },
        }
        for row in scored_rows
    ]
    groups = [row["groups"] for row in candidate_rows]
    target_moments = {
        group: sum(float(membership.get(group, 0.0)) for membership in groups) / len(groups)
        for group in sorted({group for membership in groups for group in membership})
    }
    dcms_result = solve_dcms_with_slack(
        sample_ids=sample_ids,
        utilities=[float(row["score"]) for row in candidate_rows],
        group_membership=groups,
        budget=budget,
        target_moments=target_moments,
        slack_grid=[0.0, 0.05, 0.1, 0.2, 0.5],
        kappa=0.2,
        rounding_seed=int(args.seed),
    )

    random_membership = [
        {**row, "selected": int(row["sample_id"] in set(random_ids))}
        for row in scored_rows
    ]
    dcms_membership = [
        {**row, "selected": int(row["sample_id"] in set(dcms_result.selected_ids))}
        for row in scored_rows
    ]
    random_audit = audit_preference_acquisition(
        random_membership,
        method="Random",
        group_fields=("length_gap_bin", "ab_position"),
    )
    dcms_audit = audit_preference_acquisition(
        dcms_membership,
        method="ActiveDPO+DCMS",
        group_fields=("length_gap_bin", "ab_position"),
        random_reference_rows=random_membership,
    )

    reveal = reveal_selected_preference_labels(
        fixed_pool.active_pool,
        oracle_store=fixed_pool.oracle_store,
        selected_ids=dcms_result.selected_ids,
        round_index=0,
        method="active_dpo_dcms",
    )
    heldout_rows = [
        row
        for row in fixed_pool.active_pool
        if row["sample_id"] in set(split_manifest["heldout_ids"])
    ]
    preference_rows = [
        {
            "oracle_preference": fixed_pool.oracle_store[row["sample_id"]]["preference_label"],
            "predicted_preference": "A" if index % 4 else "B",
            "observable_group": _length_gap_bin(float(row["length_gap"])),
        }
        for index, row in enumerate(heldout_rows)
    ]
    judge_rows = [
        {
            "judge_win": 1.0 if index % 3 else 0.0,
            "length_gap_bin": _length_gap_bin(float(row["length_gap"])),
        }
        for index, row in enumerate(heldout_rows)
    ]
    capability_rows = [
        {"baseline_score": 0.70 + index * 0.01, "policy_score": 0.68 + index * 0.01}
        for index in range(3)
    ]
    aulc_rows = [
        {"budget": 0, "performance": 0.50},
        {"budget": budget, "performance": 0.62},
        {"budget": budget * 2, "performance": 0.70},
    ]
    evaluation = build_preference_evaluation_metrics(
        preference_rows=preference_rows,
        judge_rows=judge_rows,
        capability_rows=capability_rows,
        aulc_rows=aulc_rows,
        group_field="observable_group",
    )

    paired_pool = build_preference_fixed_pool(
        raw_rows[:10],
        seed=int(args.seed),
        include_both_positions=True,
    )
    paired_rows = [
        {
            **row,
            "score": 0.5 + (0.1 if row["ab_position"] == "original" else 0.0) + index * 0.001,
        }
        for index, row in enumerate(paired_pool.active_pool)
    ]
    ab_audit = audit_ab_position_intervention(paired_rows, score_field="score", budget=4)
    intervention_rows = build_preference_intervention_rows(
        active_pool_rows=fixed_pool.active_pool,
        logprob_rows=logprob_rows,
        score_rows=scored_rows,
    )

    payload = {
        "smoke_type": "synthetic_cpu",
        "seed": int(args.seed),
        "input_rows": len(raw_rows),
        "active_pool_rows": len(fixed_pool.active_pool),
        "split_sizes": {
            key: len(value)
            for key, value in split_manifest.items()
            if key.endswith("_ids")
        },
        "score_methods": ["reward_margin", "apl", "active_dpo"],
        "selected_counts": {
            "random": len(random_ids),
            "active_dpo": len(active_dpo_ids),
            "active_dpo_dcms": len(dcms_result.selected_ids),
        },
        "revealed_count": len(reveal.revealed_ids),
        "dpo_train_row_count": len(reveal.training_rows),
        "dcms": {
            "solver_status": dcms_result.solver_status,
            "selected_slack": dcms_result.selected_slack,
            "utility_retained": dcms_result.utility_retained,
            "max_constraint_violation": dcms_result.max_constraint_violation,
            "non_binary_propensity_count": sum(0 < value < 1 for value in dcms_result.q_propensity.values()),
            "rounded_moments": dcms_result.rounded_moments,
        },
        "acquisition_audit": {
            "random_max_tv": random_audit["max_acquisition_tv"],
            "dcms_max_tv": dcms_audit["max_acquisition_tv"],
            "dcms_random_reference_present": dcms_audit["random_reference_present"],
        },
        "evaluation": evaluation,
        "ab_position": {
            "pair_count": ab_audit["pair_count"],
            "selected_count": len(ab_audit["selected_ids"]),
            "position_propensity": ab_audit["position_propensity"],
        },
        "intervention_input_rows": len(intervention_rows),
    }
    write_json(payload, args.output_path)
    print(payload)


def _synthetic_preference_rows(rng: random.Random, *, count: int) -> list[dict[str, Any]]:
    rows = []
    for index in range(count):
        rows.append(
            {
                "id": f"pair:{index}",
                "prompt": f"Synthetic prompt {index}",
                "response_a": "good " * (3 + index % 5),
                "response_b": "alternative " * (2 + (index * 3) % 7),
                "chosen": "A" if rng.random() >= 0.45 else "B",
                "source_a": "model_a" if index % 2 else "model_b",
                "source_b": "model_c" if index % 3 else "model_a",
            }
        )
    return rows


def _synthetic_logprob_row(row: dict[str, Any], *, index: int) -> dict[str, Any]:
    policy_gap = ((index % 9) - 4) / 3.0
    reference_gap = ((index % 5) - 2) / 5.0
    implicit_gap = policy_gap - reference_gap
    return {
        "sample_id": row["sample_id"],
        "policy_logprob_response_1": policy_gap,
        "policy_logprob_response_2": 0.0,
        "reference_logprob_response_1": reference_gap,
        "reference_logprob_response_2": 0.0,
        "policy_logprob_gap": policy_gap,
        "reference_logprob_gap": reference_gap,
        "implicit_reward_gap": implicit_gap,
        "probability_response_1": 1.0 / (1.0 + math.exp(-policy_gap)),
        "response_1_token_count": 8 + index % 5,
        "response_2_token_count": 7 + (index * 2) % 6,
    }


def _length_gap_bin(gap: float) -> str:
    if gap < -0.2:
        return "b_longer"
    if gap > 0.2:
        return "a_longer"
    return "balanced"


if __name__ == "__main__":
    main()
