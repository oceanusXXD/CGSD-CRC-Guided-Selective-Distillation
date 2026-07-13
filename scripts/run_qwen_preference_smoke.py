from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mias_dcms.benchmark_training import LoraTrainingConfig, train_preference_dpo
from mias_dcms.checkpoint_registry import register_initial_policy_checkpoint
from mias_dcms.preference_dcms_inputs import build_preference_dcms_candidate_rows
from mias_dcms.preference_evaluation import build_preference_evaluation_metrics, preference_accuracy
from mias_dcms.preference_logprob_audit import audit_preference_logprobs
from mias_dcms.preference_logprob_generation import (
    build_preference_logprob_rows,
    load_causal_lm_for_logprobs,
    load_tokenizer_for_logprobs,
)
from mias_dcms.preference_pool import build_preference_fixed_pool
from mias_dcms.preference_reveal import reveal_selected_preference_labels
from mias_dcms.preference_scoring import build_preference_baseline_score_rows
from mias_dcms.preference_selection_metrics import (
    build_preference_selection_metrics,
    materialize_preference_group_fields,
    utility_retained_from_scores,
)
from mias_dcms.preference_split_manifest import build_preference_split_manifest
from mias_dcms.prompt_clusters import build_prompt_cluster_assignments
from mias_dcms.selection import rank_normalize_utilities, solve_dcms_with_slack
from mias_dcms.selectors import random_without_replacement, select_top_budget
from mias_dcms.utils import (
    configure_torch_performance,
    get_device,
    read_json,
    resolve_model_reference,
    write_json,
    write_jsonl,
)


GROUP_FIELDS = (
    "length_gap_bin",
    "source_pair",
    "prompt_cluster",
    "ab_position",
    "length_by_prompt_cluster",
)
TOPICS = (
    ("water", "Water freezes at zero degrees Celsius under standard pressure."),
    ("photosynthesis", "Plants use light to convert water and carbon dioxide into sugars."),
    ("gravity", "Gravity attracts masses toward one another."),
    ("oceans", "The Pacific Ocean is the largest ocean on Earth."),
    ("sound", "Sound travels through a medium as a mechanical wave."),
    ("maps", "A map scale relates a drawn distance to a real-world distance."),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a small real-model preference pipeline on a synthetic, auditable pool."
    )
    parser.add_argument("--model_path", default="qwen3-0.6b")
    parser.add_argument("--raw_input_path", type=Path)
    parser.add_argument("--output_dir", type=Path, default=PROJECT_ROOT / "experiments/runs/qwen06b_preference_smoke")
    parser.add_argument("--report_path", type=Path, default=PROJECT_ROOT / "experiments/reports/real_smoke_qwen06b.json")
    parser.add_argument("--pair_count", type=int, default=24)
    parser.add_argument("--budget", type=int, default=6)
    parser.add_argument("--seed", type=int, default=20260713)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--train_method", default="ActiveDPO+DCMS")
    parser.add_argument("--update_steps", type=int, default=1)
    parser.add_argument("--initial_update_steps", type=int, default=2)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_response_words", type=int, default=120)
    parser.add_argument("--dcms_kappa", type=float, default=0.05)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if int(args.pair_count) < 6 or int(args.pair_count) % 6:
        raise ValueError("pair_count must be at least 6 and divisible by 6")
    if int(args.budget) <= 0:
        raise ValueError("budget must be positive")

    model_path = resolve_model_reference(str(args.model_path), PROJECT_ROOT)
    if not Path(model_path).is_dir():
        raise FileNotFoundError(
            f"model path {args.model_path!r} was not found; set MIAS_DCMS_MODEL_ROOT or pass --model_path"
        )
    output_dir = _resolve_output_path(args.output_dir)
    report_path = _resolve_output_path(args.report_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    configure_torch_performance(enable_tf32=True)
    device = get_device(str(args.device))

    raw_rows = _load_or_build_raw_rows(
        pair_count=int(args.pair_count),
        raw_input_path=args.raw_input_path,
        seed=int(args.seed),
        max_response_words=int(args.max_response_words),
    )
    pair_count = len(raw_rows)
    if pair_count < 6 or pair_count % 6:
        raise ValueError("the selected raw pool must contain a pair count divisible by 6")
    write_jsonl(raw_rows, output_dir / "raw_preference.jsonl")
    fixed_pool = build_preference_fixed_pool(
        raw_rows,
        seed=int(args.seed),
        include_both_positions=True,
    )
    write_jsonl(fixed_pool.active_pool, output_dir / "active_pool.jsonl")
    write_json(fixed_pool.oracle_store, output_dir / "oracle_store.json")
    write_json(fixed_pool.swap_manifest, output_dir / "swap_manifest.json")

    cluster_result = build_prompt_cluster_assignments(
        rows=fixed_pool.active_pool,
        embeddings_by_id=_deterministic_embeddings(fixed_pool.active_pool),
        cluster_count=3,
        id_field="sample_id",
        softmax_temperature=0.5,
    )
    write_jsonl(cluster_result.rows, output_dir / "prompt_clusters.jsonl")
    write_json(cluster_result.summary, output_dir / "prompt_clusters.summary.json")

    split_manifest = build_preference_split_manifest(
        fixed_pool.active_pool,
        seed=int(args.seed),
        seed_size=pair_count // 3,
        active_size=pair_count,
        heldout_size=pair_count // 3,
        test_size=pair_count // 3,
        id_field="sample_id",
        prompt_field="prompt",
        group_field="swap_pair_id",
    )
    write_json(split_manifest, output_dir / "split_manifest.json")

    initial_seed_dir = output_dir / "initial_seed"
    seed_ids = list(split_manifest["seed_ids"])
    _write_selected_ids(initial_seed_dir, seed_ids, method="initial_seed", budget=len(seed_ids))
    initial_reveal = reveal_selected_preference_labels(
        fixed_pool.active_pool,
        oracle_store=fixed_pool.oracle_store,
        selected_ids=seed_ids,
        round_index=0,
        method="initial_seed",
    )
    initial_reveal_summary = _write_reveal(initial_seed_dir, initial_reveal, seed_ids, "initial_seed")

    dtype = "float16" if device.type == "cuda" else "float32"
    initial_adapter_dir = output_dir / "initial_policy_adapter"
    training_config = _training_config(
        model_path=model_path,
        output_dir=initial_adapter_dir,
        seed=int(args.seed),
        dtype=dtype,
        update_steps=int(args.initial_update_steps),
        initial_policy_adapter_path=None,
        max_length=int(args.max_length),
        batch_size=int(args.batch_size),
    )
    initial_training_summary = train_preference_dpo(
        initial_reveal.training_rows,
        config=training_config,
    )
    write_json(initial_training_summary, initial_seed_dir / "training_summary.json")
    write_json(
        {
            "seed_label_count": len(seed_ids),
            "active_label_count": len(seed_ids),
            "evaluation_label_count": 0,
            "judge_calls": 0,
            "selector_compute_seconds": 0.0,
            "oracle_label_calls": len(seed_ids),
        },
        initial_seed_dir / "cost_report.json",
    )
    checkpoint_manifest_path = output_dir / "initial_policy_checkpoint_manifest.json"
    checkpoint_report = register_initial_policy_checkpoint(
        checkpoint_path=initial_adapter_dir,
        output_manifest_path=checkpoint_manifest_path,
        base_dir=PROJECT_ROOT,
        model_name_or_path=model_path,
        training_config=_serializable_training_config(training_config),
    )
    if not checkpoint_report.is_ready:
        raise RuntimeError(f"initial policy checkpoint registration failed: {checkpoint_report.issues}")

    logprob_rows, logprob_summary = _generate_logprobs(
        fixed_pool.active_pool,
        model_path=model_path,
        adapter_path=initial_adapter_dir,
        device=device,
        dtype=dtype,
        max_length=int(args.max_length),
        batch_size=int(args.batch_size),
    )
    write_jsonl(logprob_rows, output_dir / "logprobs.jsonl")
    write_json(logprob_summary, output_dir / "logprobs.summary.json")
    scored_rows = _build_scored_rows(
        logprob_rows,
        active_pool=fixed_pool.active_pool,
        cluster_rows=cluster_result.rows,
    )
    write_jsonl(scored_rows, output_dir / "scored_pool.jsonl")

    method_selection = _run_selections(
        scored_rows,
        output_dir=output_dir,
        budget=int(args.budget),
        seed=int(args.seed),
        dcms_kappa=float(args.dcms_kappa),
    )
    train_method = str(args.train_method)
    if train_method not in method_selection:
        raise ValueError(f"train_method {train_method!r} is not one of {sorted(method_selection)}")

    train_dir = output_dir / _method_dir_name(train_method)
    train_payload = method_selection[train_method]
    train_reveal = reveal_selected_preference_labels(
        fixed_pool.active_pool,
        oracle_store=fixed_pool.oracle_store,
        selected_ids=train_payload["selected_ids"],
        round_index=0,
        method=train_method,
    )
    reveal_summary = _write_reveal(
        train_dir,
        train_reveal,
        train_payload["selected_ids"],
        train_method,
    )
    final_adapter_dir = train_dir / "policy_adapter"
    final_config = _training_config(
        model_path=model_path,
        output_dir=final_adapter_dir,
        seed=int(args.seed),
        dtype=dtype,
        update_steps=int(args.update_steps),
        initial_policy_adapter_path=initial_adapter_dir,
        max_length=int(args.max_length),
        batch_size=int(args.batch_size),
    )
    final_training_summary = train_preference_dpo(
        train_reveal.training_rows,
        config=final_config,
    )
    write_json(final_training_summary, train_dir / "training_summary.json")

    heldout_ids = set(split_manifest["heldout_ids"])
    heldout_rows = [row for row in fixed_pool.active_pool if str(row["sample_id"]) in heldout_ids]
    initial_eval_logprobs = [row for row in logprob_rows if str(row["sample_id"]) in heldout_ids]
    final_eval_logprobs, final_eval_summary = _generate_logprobs(
        heldout_rows,
        model_path=model_path,
        adapter_path=final_adapter_dir,
        device=device,
        dtype=dtype,
        max_length=int(args.max_length),
        batch_size=int(args.batch_size),
    )
    evaluation = _write_evaluation(
        train_dir,
        heldout_rows=heldout_rows,
        oracle_store=fixed_pool.oracle_store,
        initial_logprobs=initial_eval_logprobs,
        final_logprobs=final_eval_logprobs,
        seed_budget=len(seed_ids),
        active_budget=int(args.budget),
        final_eval_summary=final_eval_summary,
        judge_definition=(
            "fixed_heldout_human_preference_label"
            if args.raw_input_path is not None
            else "fixed_synthetic_oracle_label"
        ),
    )
    final_cost_report = {
        "seed_label_count": len(seed_ids),
        "active_label_count": int(args.budget),
        "evaluation_label_count": len(heldout_rows),
        "judge_calls": 0,
        "selector_compute_seconds": 0.0,
        "oracle_label_calls": int(args.budget),
    }
    write_json(final_cost_report, train_dir / "cost_report.json")
    run_record = _build_run_record(
        train_dir,
        selection_summary=read_json(train_dir / "selection_summary.json"),
        reveal_summary=reveal_summary,
        training_summary=final_training_summary,
        evaluation_metrics=evaluation["evaluation_metrics"],
        seed_label_count=len(seed_ids),
        evaluation_label_count=len(heldout_rows),
        budget=int(args.budget),
        method=train_method,
        seed=int(args.seed),
    )
    write_json(run_record, train_dir / "run_record.json")

    report = {
        "experiment_type": (
            "real_model_real_helpsteer2_subset_smoke"
            if args.raw_input_path is not None
            else "real_model_synthetic_pool_smoke"
        ),
        "paper_evidence": False,
        "model_path": model_path,
        "raw_input_path": str(args.raw_input_path) if args.raw_input_path is not None else None,
        "device": str(device),
        "seed": int(args.seed),
        "raw_pair_count": len(raw_rows),
        "active_pool_count": len(fixed_pool.active_pool),
        "split_sizes": {
            key: len(value)
            for key, value in split_manifest.items()
            if key.endswith("_ids")
        },
        "budget": int(args.budget),
        "dcms_kappa": float(args.dcms_kappa),
        "initial_training": initial_training_summary,
        "checkpoint_registration": checkpoint_report.as_dict(),
        "selection_methods": {
            method: {
                "selected_count": payload["selection_summary"]["selected_count"],
                "selected_ids": payload["selected_ids"],
                "selection_summary": payload["selection_summary"],
            }
            for method, payload in method_selection.items()
        },
        "trained_method": train_method,
        "training": final_training_summary,
        "evaluation": evaluation,
        "report_path": str(report_path),
        "artifacts": {
            "active_pool": str(output_dir / "active_pool.jsonl"),
            "oracle_store": str(output_dir / "oracle_store.json"),
            "split_manifest": str(output_dir / "split_manifest.json"),
            "logprobs": str(output_dir / "logprobs.jsonl"),
            "initial_adapter": str(initial_adapter_dir),
            "final_adapter": str(final_adapter_dir),
            "run_record": str(train_dir / "run_record.json"),
        },
        "limitations": [
            (
                "The pool is a deterministic subset of HelpSteer2-Preference; it is not the full main experiment."
                if args.raw_input_path is not None
                else "The pool and oracle labels are synthetic; no external human judge was used."
            ),
            "Capability regression is a preferred-response log-probability proxy, not a separate capability benchmark.",
            "Only the configured train_method is trained downstream; all listed methods receive selector outputs.",
        ],
    }
    write_json(report, report_path)
    print(json.dumps(_compact_report(report), ensure_ascii=False, sort_keys=True))


def _load_or_build_raw_rows(
    *,
    pair_count: int,
    raw_input_path: Path | None,
    seed: int,
    max_response_words: int,
) -> list[dict[str, Any]]:
    if raw_input_path is not None:
        from random import Random

        from mias_dcms.utils import read_jsonl

        source_rows = read_jsonl(_resolve_output_path(raw_input_path))
        by_prompt: dict[str, dict[str, Any]] = {}
        for row in source_rows:
            if int(row.get("preferred_response", 0)) not in (1, 2):
                continue
            prompt = str(row.get("prompt", ""))
            if prompt and prompt not in by_prompt:
                by_prompt[prompt] = dict(row)
        candidates = list(by_prompt.values())
        candidates = [
            row
            for row in candidates
            if len(str(row.get("response_1", "")).split()) <= max_response_words
            and len(str(row.get("response_2", "")).split()) <= max_response_words
        ]
        Random(int(seed)).shuffle(candidates)
        if len(candidates) < pair_count:
            raise ValueError(
                f"raw input contains only {len(candidates)} unique non-tie prompts; requested {pair_count}"
            )
        selected = candidates[:pair_count]
        return [
            {
                "id": str(row.get("id", f"helpsteer2:{index}")),
                "prompt": str(row["prompt"]),
                "response_1": str(row["response_1"]),
                "response_2": str(row["response_2"]),
                "preferred_response": int(row["preferred_response"]),
                "preference_strength": row.get("preference_strength"),
                "source_a": row.get("source_a"),
                "source_b": row.get("source_b"),
            }
            for index, row in enumerate(selected)
        ]
    return _build_raw_rows(pair_count)


def _build_raw_rows(pair_count: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index in range(pair_count):
        topic, fact = TOPICS[index % len(TOPICS)]
        detail = " " + "This statement is stated directly and avoids unnecessary speculation." * (index % 3)
        response_a = f"{fact}{detail}"
        response_b = f"A brief answer about {topic} is that it is an ordinary factual concept."
        rows.append(
            {
                "id": f"synthetic-pair-{index:03d}",
                "prompt": f"Give a concise factual answer about {topic}; use instance {index}.",
                "response_a": response_a,
                "response_b": response_b,
                "preference_label": "A" if index % 2 == 0 else "B",
                "preference_strength": 1.0,
                "source_a": "expert_reference",
                "source_b": "short_baseline",
            }
        )
    return rows


def _deterministic_embeddings(rows: Sequence[Mapping[str, Any]]) -> dict[str, list[float]]:
    embeddings: dict[str, list[float]] = {}
    for row in rows:
        pair_id = str(row["swap_pair_id"])
        index = int.from_bytes(hashlib.sha256(pair_id.encode("utf-8")).digest()[:4], "big")
        angle = 2.0 * math.pi * (index % 3) / 3.0
        embeddings[str(row["sample_id"])] = [math.cos(angle), math.sin(angle), 0.1 * (index % 2)]
    return embeddings


def _training_config(
    *,
    model_path: str,
    output_dir: Path,
    seed: int,
    dtype: str,
    update_steps: int,
    initial_policy_adapter_path: Path | None,
    max_length: int,
    batch_size: int,
) -> LoraTrainingConfig:
    return LoraTrainingConfig(
        model_name_or_path=str(model_path),
        output_dir=output_dir,
        epochs=1,
        learning_rate=1e-5,
        batch_size=batch_size,
        gradient_accumulation_steps=1,
        max_length=max_length,
        lora_r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules=("q_proj", "k_proj", "v_proj", "o_proj"),
        mixed_precision="no",
        dtype=dtype,
        gradient_checkpointing=False,
        seed=seed,
        beta=0.1,
        update_steps=update_steps,
        initial_policy_adapter_path=(
            str(initial_policy_adapter_path) if initial_policy_adapter_path else None
        ),
    )


def _generate_logprobs(
    rows: Sequence[Mapping[str, Any]],
    *,
    model_path: str,
    adapter_path: Path,
    device: torch.device,
    dtype: str,
    max_length: int,
    batch_size: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    torch_dtype: Any = torch.float16 if dtype == "float16" else torch.float32
    tokenizer = load_tokenizer_for_logprobs(model_path, local_files_only=True)
    policy = load_causal_lm_for_logprobs(
        model_path,
        device=device,
        torch_dtype=torch_dtype,
        local_files_only=True,
        adapter_path=str(adapter_path),
    )
    reference = load_causal_lm_for_logprobs(
        model_path,
        device=device,
        torch_dtype=torch_dtype,
        local_files_only=True,
    )
    generated, summary = build_preference_logprob_rows(
        rows,
        tokenizer=tokenizer,
        policy_model=policy,
        reference_model=reference,
        device=device,
        batch_size=batch_size,
        max_length=max_length,
        prompt_format="chatml_pairwise_v1",
        truncation_strategy="truncate_prompt_left",
    )
    audited, audit_summary = audit_preference_logprobs(
        generated,
        id_field="sample_id",
        require_nonzero_implicit_margin=True,
    )
    summary = {**summary, "audit": audit_summary, "adapter_path": str(adapter_path)}
    del policy, reference, tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return audited, summary


def _build_scored_rows(
    logprob_rows: Sequence[Mapping[str, Any]],
    *,
    active_pool: Sequence[Mapping[str, Any]],
    cluster_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    pool_by_id = {str(row["sample_id"]): dict(row) for row in active_pool}
    cluster_by_id = {str(row["sample_id"]): dict(row) for row in cluster_rows}
    merged: list[dict[str, Any]] = []
    for row in logprob_rows:
        sample_id = str(row["sample_id"])
        payload = dict(row)
        payload.update(pool_by_id[sample_id])
        payload.update(cluster_by_id[sample_id])
        merged.append(payload)
    return build_preference_baseline_score_rows(
        merged,
        methods=("reward_margin", "apl", "active_dpo"),
        active_dpo_length_normalize=True,
        active_dpo_novelty_weight=0.25,
        id_field="sample_id",
    )


def _run_selections(
    scored_rows: Sequence[Mapping[str, Any]],
    *,
    output_dir: Path,
    budget: int,
    seed: int,
    dcms_kappa: float,
) -> dict[str, dict[str, Any]]:
    rows = [dict(row) for row in scored_rows]
    sample_ids = [str(row["sample_id"]) for row in rows]
    method_specs = {
        "Random": (None, "random"),
        "Reward Margin": ("reward_margin_score", "reward_margin"),
        "APL": ("apl_score", "apl"),
        "ActiveDPO": ("active_dpo_score", "active_dpo"),
    }
    results: dict[str, dict[str, Any]] = {}
    for method, (score_field, normalized_method) in method_specs.items():
        method_dir = output_dir / _method_dir_name(method)
        if score_field is None:
            selected_ids = random_without_replacement(sample_ids, budget=budget, seed=seed)
            membership_rows = [
                {
                    **materialize_preference_group_fields(row),
                    "method": "random",
                    "selected": int(str(row["sample_id"]) in set(selected_ids)),
                }
                for row in rows
            ]
            selection_summary = {
                "method": "random",
                "budget": budget,
                "pool_size": len(rows),
                "selected_count": len(selected_ids),
                "selection_metrics": build_preference_selection_metrics(membership_rows, method="random"),
            }
            write_jsonl(membership_rows, method_dir / "membership.jsonl")
        else:
            scores = [float(row[score_field]) for row in rows]
            selected_ids = select_top_budget(sample_ids=sample_ids, scores=scores, budget=budget)
            selected_set = set(selected_ids)
            membership_rows = [
                {
                    **materialize_preference_group_fields(row),
                    "method": normalized_method,
                    "score_field": score_field,
                    "score": score,
                    "selected": int(str(row["sample_id"]) in selected_set),
                }
                for row, score in zip(rows, scores, strict=True)
            ]
            selection_summary = {
                "method": normalized_method,
                "budget": budget,
                "pool_size": len(rows),
                "selected_count": len(selected_ids),
                "selection_metrics": build_preference_selection_metrics(
                    membership_rows,
                    method=normalized_method,
                    score_field=score_field,
                    utility_retained=utility_retained_from_scores(
                        membership_rows,
                        selected_ids=selected_ids,
                        score_field="score",
                    ),
                ),
            }
            write_jsonl(membership_rows, method_dir / "membership.jsonl")
        selected_payload = {
            "selected_ids": selected_ids,
            "budget": budget,
            "selected_count": len(selected_ids),
            "method": method,
        }
        write_json(selected_payload, method_dir / "selected_ids.json")
        write_json(selection_summary, method_dir / "selection_summary.json")
        results[method] = {
            "selected_ids": selected_ids,
            "selection_summary": selection_summary,
            "directory": str(method_dir),
        }

    for method, base_method in (("APL+DCMS", "apl"), ("ActiveDPO+DCMS", "active_dpo")):
        method_dir = output_dir / _method_dir_name(method)
        candidates = build_preference_dcms_candidate_rows(
            rows,
            method=base_method,
            group_fields=GROUP_FIELDS,
            id_field="sample_id",
        )
        candidate_by_id = {str(row["sample_id"]): row for row in candidates}
        raw_scores = [float(candidate["score"]) for candidate in candidates]
        memberships = [candidate["groups"] for candidate in candidates]
        groups = sorted({group for membership in memberships for group in membership})
        targets = {
            group: sum(float(membership.get(group, 0.0)) for membership in memberships) / len(memberships)
            for group in groups
        }
        result = solve_dcms_with_slack(
            sample_ids=sample_ids,
            utilities=rank_normalize_utilities(raw_scores),
            group_membership=memberships,
            budget=budget,
            target_moments=targets,
            slack_grid=(0.0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5),
            kappa=float(dcms_kappa),
            rounding_seed=seed,
        )
        selected_ids = list(result.selected_ids)
        propensity_rows = []
        for row in rows:
            sample_id = str(row["sample_id"])
            candidate = candidate_by_id[sample_id]
            propensity_rows.append(
                {
                    **materialize_preference_group_fields(row, group_fields=GROUP_FIELDS),
                    "base_score": float(candidate["score"]),
                    "q_propensity": float(result.q_propensity[sample_id]),
                    "selected": int(result.selection_indicator[sample_id]),
                }
            )
        selection_summary = {
            "method": method,
            "budget": budget,
            "pool_size": len(rows),
            "selected_count": len(selected_ids),
            "selected_slack": result.selected_slack,
            "utility_retained": result.utility_retained,
            "max_constraint_violation": result.max_constraint_violation,
            "solver_status": result.solver_status,
            "rounded_moments": result.rounded_moments,
            "selection_metrics": build_preference_selection_metrics(
                propensity_rows,
                method=method,
                group_fields=GROUP_FIELDS,
                constraint_violation=result.max_constraint_violation,
                utility_retained=result.utility_retained,
            ),
        }
        write_json(
            {"selected_ids": selected_ids, "budget": budget, "selected_count": len(selected_ids), "method": method},
            method_dir / "selected_ids.json",
        )
        write_jsonl(propensity_rows, method_dir / "propensity.jsonl")
        write_json(selection_summary, method_dir / "selection_summary.json")
        results[method] = {
            "selected_ids": selected_ids,
            "selection_summary": selection_summary,
            "directory": str(method_dir),
        }
    return results


def _write_selected_ids(output_dir: Path, selected_ids: Sequence[str], *, method: str, budget: int) -> None:
    write_json(
        {"selected_ids": list(selected_ids), "budget": int(budget), "selected_count": len(selected_ids), "method": method},
        output_dir / "selected_ids.json",
    )


def _write_reveal(output_dir: Path, result: Any, selected_ids: Sequence[str], method: str) -> dict[str, Any]:
    write_jsonl(result.revealed_rows, output_dir / "revealed_rows.jsonl")
    write_jsonl(result.training_rows, output_dir / "dpo_train_rows.jsonl")
    summary = {
        "round": 0,
        "method": method,
        "selected_count": len(selected_ids),
        "revealed_count": len(result.revealed_rows),
        "dpo_train_row_count": len(result.training_rows),
        "unrevealed_count": len(result.unrevealed_ids),
    }
    write_json(summary, output_dir / "summary.json")
    return summary


def _write_evaluation(
    output_dir: Path,
    *,
    heldout_rows: Sequence[Mapping[str, Any]],
    oracle_store: Mapping[str, Mapping[str, Any]],
    initial_logprobs: Sequence[Mapping[str, Any]],
    final_logprobs: Sequence[Mapping[str, Any]],
    seed_budget: int,
    active_budget: int,
    final_eval_summary: Mapping[str, Any],
    judge_definition: str,
) -> dict[str, Any]:
    pool_by_id = {str(row["sample_id"]): row for row in heldout_rows}
    initial_eval = _evaluation_rows(initial_logprobs, pool_by_id=pool_by_id, oracle_store=oracle_store)
    final_eval = _evaluation_rows(final_logprobs, pool_by_id=pool_by_id, oracle_store=oracle_store)
    judge_rows = [
        {
            **row,
            "judge_win": int(row["predicted_preference"] == row["oracle_preference"]),
            "judge_source": str(judge_definition),
        }
        for row in final_eval
    ]
    capability_rows = [
        {
            "sample_id": final_row["sample_id"],
            "baseline_score": initial_row["preferred_response_logprob"],
            "policy_score": final_row["preferred_response_logprob"],
            "capability_source": "preferred_response_logprob_proxy",
        }
        for initial_row, final_row in zip(initial_eval, final_eval, strict=True)
    ]
    aulc_rows = [
        {"budget": int(seed_budget), "performance": preference_accuracy(initial_eval)},
        {"budget": int(seed_budget + active_budget), "performance": preference_accuracy(final_eval)},
    ]
    write_jsonl(final_eval, output_dir / "heldout_preference_predictions.jsonl")
    write_jsonl(judge_rows, output_dir / "judge_rows.jsonl")
    write_jsonl(capability_rows, output_dir / "capability_rows.jsonl")
    write_jsonl(aulc_rows, output_dir / "aulc_rows.jsonl")
    metrics = build_preference_evaluation_metrics(
        preference_rows=final_eval,
        judge_rows=judge_rows,
        capability_rows=capability_rows,
        aulc_rows=aulc_rows,
        group_field="observable_group",
        length_bin_field="length_gap_bin",
        label_field="oracle_preference",
        prediction_field="predicted_preference",
    )
    payload = {
        "input_paths": {
            "preference_predictions_path": str(output_dir / "heldout_preference_predictions.jsonl"),
            "judge_rows_path": str(output_dir / "judge_rows.jsonl"),
            "capability_rows_path": str(output_dir / "capability_rows.jsonl"),
            "aulc_rows_path": str(output_dir / "aulc_rows.jsonl"),
        },
        "evaluation_metrics": metrics,
        "initial_evaluation_metrics": {
            "preference_accuracy": preference_accuracy(initial_eval),
            "evaluation_count": len(initial_eval),
        },
        "final_logprob_generation": dict(final_eval_summary),
        "judge_definition": str(judge_definition),
        "capability_definition": "preferred_response_logprob_proxy",
    }
    write_json(payload, output_dir / "evaluation_metrics.json")
    return payload


def _evaluation_rows(
    logprob_rows: Sequence[Mapping[str, Any]],
    *,
    pool_by_id: Mapping[str, Mapping[str, Any]],
    oracle_store: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for logprob in logprob_rows:
        sample_id = str(logprob["sample_id"])
        pool_row = dict(pool_by_id[sample_id])
        oracle_label = str(oracle_store[sample_id]["preference_label"])
        predicted = "A" if float(logprob["implicit_reward_gap"]) >= 0.0 else "B"
        preferred_key = "policy_logprob_response_1" if oracle_label == "A" else "policy_logprob_response_2"
        rows.append(
            {
                **pool_row,
                "oracle_preference": oracle_label,
                "predicted_preference": predicted,
                "observable_group": str(pool_row["length_gap_bin"]),
                "preferred_response_logprob": float(logprob[preferred_key]),
            }
        )
    return rows


def _build_run_record(
    output_dir: Path,
    *,
    selection_summary: Mapping[str, Any],
    reveal_summary: Mapping[str, Any],
    training_summary: Mapping[str, Any],
    evaluation_metrics: Mapping[str, Any],
    seed_label_count: int,
    evaluation_label_count: int,
    budget: int,
    method: str,
    seed: int,
) -> dict[str, Any]:
    selected_count = int(selection_summary["selected_count"])
    return {
        "dataset": "synthetic_preference_pool",
        "model": "qwen3-0.6b",
        "method": method,
        "budget": budget,
        "seed": seed,
        "run_status": "completed",
        "selected_count": selected_count,
        "selection_metrics": dict(selection_summary.get("selection_metrics") or {}),
        "training_metrics": dict(training_summary),
        "evaluation_metrics": dict(evaluation_metrics),
        "cost_metrics": {
            "seed_label_count": seed_label_count,
            "active_label_count": selected_count,
            "evaluation_label_count": evaluation_label_count,
            "judge_calls": 0,
            "oracle_label_calls": selected_count,
        },
        "artifacts": {
            "selection_summary_path": str(output_dir / "selection_summary.json"),
            "reveal_summary_path": str(output_dir / "summary.json"),
            "training_summary_path": str(output_dir / "training_summary.json"),
            "evaluation_metrics_path": str(output_dir / "evaluation_metrics.json"),
        },
    }


def _serializable_training_config(config: LoraTrainingConfig) -> dict[str, Any]:
    return {
        key: (str(value) if isinstance(value, Path) else value)
        for key, value in config.__dict__.items()
    }


def _resolve_output_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def _method_dir_name(method: str) -> str:
    return str(method).replace("/", "_").replace(" ", "_")


def _compact_report(report: Mapping[str, Any]) -> dict[str, Any]:
    evaluation = dict(report.get("evaluation") or {})
    return {
        "experiment_type": report.get("experiment_type"),
        "paper_evidence": report.get("paper_evidence"),
        "model_path": report.get("model_path"),
        "active_pool_count": report.get("active_pool_count"),
        "trained_method": report.get("trained_method"),
        "selection_methods": sorted(dict(report.get("selection_methods") or {})),
        "evaluation_metrics": evaluation.get("evaluation_metrics"),
        "report_path": report.get("report_path"),
    }


if __name__ == "__main__":
    main()
