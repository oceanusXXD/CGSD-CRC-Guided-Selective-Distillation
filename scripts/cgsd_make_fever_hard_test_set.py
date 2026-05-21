#!/usr/bin/env python
"""Build a hard-skewed FEVER test set from round0 base-error signals.

The default selector deliberately uses only the pre-LoRA base model's actual
round0 correctness. CRC defer and NS difficulty are still reported in the
summary, but are not used by the default test selector.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from transformers import AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from algorithms.cgsd import apply_crc_decisions, calibrate_crc, score_ns_difficulty_global, summarize_crc_decisions  # noqa: E402
from scripts.cgsd_calibrate import compute_crc_sampling_statistics  # noqa: E402
from scripts.cgsd_cli_common import binary_to_int, read_jsonl  # noqa: E402
from scripts.run_cgsd import assert_embedding_coverage, load_embeddings  # noqa: E402
from src.data import format_cgsd_chat_answer, format_cgsd_chat_prompt  # noqa: E402
from src.utils import disable_tokenizer_thinking, ensure_tokenizer_padding, read_json, write_json, write_jsonl  # noqa: E402


DEFAULT_OUTPUT_ROOT = (
    "experiments/inputs/fever/"
    "qwen06b_round0_t1_alpha010_budget_sweep_1500_9000_seed1/seed1/test_sets"
)
DEFAULT_COMPARE_SPLIT = (
    "experiments/inputs/fever/"
    "qwen06b_round0_t1_alpha010_budget_sweep_1500_9000_seed1/seed1/test_sets/"
    "balanced_test_10000_ml4096safe_exclude_guide_final_train12_seed20260521.vllm_pool_split_ids.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--name", default="hard70err_test_10000_ml4096safe_exclude_guide_final_train12_seed20260521")
    parser.add_argument("--split_ids_path", default="experiments/inputs/fever/cgsd_split_ids.json")
    parser.add_argument(
        "--calibration_predictions_path",
        default="experiments/inputs/fever/round_0/calibration_student_predictions.jsonl",
    )
    parser.add_argument(
        "--pool_predictions_path",
        default="experiments/inputs/fever/round_0/pool_student_predictions.jsonl",
    )
    parser.add_argument("--embeddings_path", default="experiments/inputs/fever/embeddings.npy")
    parser.add_argument("--embedding_dim", type=int, default=2560)
    parser.add_argument("--model_path", default="/teamspace/studios/this_studio/model/qwen3-0.6b")
    parser.add_argument("--target_size", type=int, default=10_000)
    parser.add_argument("--target_base_error_rate", type=float, default=0.70)
    parser.add_argument("--label_balance", choices=("none", "exact"), default="none")
    parser.add_argument("--error_allocation", choices=("natural", "equal_per_label"), default="natural")
    parser.add_argument("--within_bucket_sampling", choices=("random", "ns_weighted"), default="random")
    parser.add_argument("--max_length", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=20260521)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--weight_power", type=float, default=2.0)
    parser.add_argument(
        "--train_ids_glob",
        default=(
            "experiments/inputs/fever/"
            "qwen06b_round0_t1_alpha010_budget_sweep_1500_9000_seed1/seed1/"
            "*_seed1.ids.json"
        ),
    )
    parser.add_argument("--compare_split_ids_path", default=DEFAULT_COMPARE_SPLIT)
    return parser.parse_args()


def resolve_path(path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute() or candidate.exists():
        return candidate
    return PROJECT_ROOT / candidate


def display_path(path: str | Path) -> str:
    candidate = resolve_path(path).resolve()
    try:
        return str(candidate.relative_to(PROJECT_ROOT.resolve()))
    except ValueError:
        return str(candidate)


def row_id(row: Mapping[str, Any]) -> str:
    sample_id = row.get("id", row.get("sample_id"))
    if sample_id is None or str(sample_id) == "":
        raise ValueError(f"row is missing id/sample_id: {row!r}")
    return str(sample_id)


def label_value(row: Mapping[str, Any]) -> int:
    return binary_to_int(row.get("groundtruth", row.get("label")), field_name=f"row {row_id(row)!r} label")


def prediction_value(row: Mapping[str, Any]) -> int:
    value = row.get("prediction")
    if value is None:
        value = int(float(row.get("score", 0.0) or 0.0) > 0.0)
    return binary_to_int(value, field_name=f"row {row_id(row)!r} prediction")


def is_base_error(row: Mapping[str, Any]) -> bool:
    return prediction_value(row) != label_value(row)


def ns_p_error(row: Mapping[str, Any]) -> float:
    return float(row.get("ns_p_error", row.get("ns_difficulty", 0.0)) or 0.0)


def quantiles(values: Sequence[float]) -> dict[str, float | None]:
    if not values:
        return {"mean": None, "p50": None, "p75": None, "p90": None, "p95": None, "max": None}
    arr = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(arr)),
        "p50": float(np.quantile(arr, 0.50)),
        "p75": float(np.quantile(arr, 0.75)),
        "p90": float(np.quantile(arr, 0.90)),
        "p95": float(np.quantile(arr, 0.95)),
        "max": float(np.max(arr)),
    }


def percentile_summary(values: Sequence[int]) -> dict[str, int | None]:
    if not values:
        return {"min": None, "p50": None, "p90": None, "p95": None, "p99": None, "max": None}
    arr = np.asarray(values, dtype=np.int64)
    return {
        "min": int(np.min(arr)),
        "p50": int(np.quantile(arr, 0.50)),
        "p90": int(np.quantile(arr, 0.90)),
        "p95": int(np.quantile(arr, 0.95)),
        "p99": int(np.quantile(arr, 0.99)),
        "max": int(np.max(arr)),
    }


def load_formal_train_ids(pattern: str) -> tuple[set[str], list[str]]:
    paths = sorted(resolve_path(path) for path in Path(PROJECT_ROOT).glob(pattern) if not Path(path).name.startswith("hard"))
    if not paths:
        paths = sorted(resolve_path(path) for path in Path(".").glob(pattern))
    ids: set[str] = set()
    used_paths: list[str] = []
    for path in paths:
        if not path.exists() or path.parent.name == "test_sets":
            continue
        payload = read_json(path)
        row_ids = [str(sample_id) for sample_id in payload.get("ids", [])]
        if not row_ids:
            continue
        ids.update(row_ids)
        used_paths.append(str(path.relative_to(PROJECT_ROOT) if path.is_relative_to(PROJECT_ROOT) else path))
    return ids, used_paths


def load_blocked_ids(split_payload: Mapping[str, Any], train_ids_glob: str) -> tuple[set[str], dict[str, Any]]:
    calibration_ids = {str(sample_id) for sample_id in split_payload.get("calibration_ids", [])}
    final_calibration_ids = {str(sample_id) for sample_id in split_payload.get("final_calibration_ids", [])}
    train_ids, train_paths = load_formal_train_ids(train_ids_glob)
    blocked = calibration_ids | final_calibration_ids | train_ids
    return blocked, {
        "calibration": len(calibration_ids),
        "final_calibration": len(final_calibration_ids),
        "formal_train_union": len(train_ids),
        "blocked_union": len(blocked),
        "excluded_train_id_files": train_paths,
    }


def answer_token_lengths(tokenizer: Any) -> dict[str, int]:
    return {
        str(label): len(tokenizer(format_cgsd_chat_answer(label), add_special_tokens=False).input_ids)
        for label in (0, 1)
    }


def add_length_metadata(rows: Iterable[dict[str, Any]], tokenizer: Any, answer_lengths: Mapping[str, int]) -> None:
    for row in rows:
        prompt = format_cgsd_chat_prompt(str(row.get("query", "")), str(row.get("document", "")))
        prompt_tokens = len(tokenizer(prompt, add_special_tokens=False).input_ids)
        label = label_value(row)
        row["prompt_tokens"] = int(prompt_tokens)
        row["train_seq_tokens"] = int(prompt_tokens + int(answer_lengths[str(label)]))


def row_stats(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    n = len(rows)
    labels = Counter(str(label_value(row)) for row in rows)
    errors = [row for row in rows if is_base_error(row)]
    deferred = [row for row in rows if bool(row.get("defer", False))]
    base_error_by_label = {
        str(label): sum(1 for row in rows if label_value(row) == label and is_base_error(row))
        for label in (0, 1)
    }
    return {
        "n": n,
        "label_counts": {"0": int(labels.get("0", 0)), "1": int(labels.get("1", 0))},
        "base_error_count": len(errors),
        "base_error_rate": float(len(errors) / n) if n else 0.0,
        "base_error_by_label": base_error_by_label,
        "defer_count": len(deferred),
        "defer_rate": float(len(deferred) / n) if n else 0.0,
        "ns_p_error": quantiles([ns_p_error(row) for row in rows]),
        "train_seq_tokens": percentile_summary([int(row.get("train_seq_tokens", 0) or 0) for row in rows]),
    }


def weighted_sample_without_replacement(
    rows: Sequence[dict[str, Any]],
    *,
    k: int,
    rng: np.random.Generator,
    weight_power: float,
) -> list[dict[str, Any]]:
    requested = int(k)
    if requested <= 0:
        return []
    if requested > len(rows):
        raise ValueError(f"requested {requested} rows from only {len(rows)} candidates")
    weights = np.asarray([(max(ns_p_error(row), 1e-6) ** float(weight_power)) for row in rows], dtype=np.float64)
    if not np.isfinite(weights).all() or float(weights.sum()) <= 0.0:
        probabilities = None
    else:
        probabilities = weights / float(weights.sum())
    indices = rng.choice(len(rows), size=requested, replace=False, p=probabilities)
    return [rows[int(index)] for index in indices]


def sample_without_replacement(
    rows: Sequence[dict[str, Any]],
    *,
    k: int,
    rng: random.Random,
) -> list[dict[str, Any]]:
    requested = int(k)
    if requested <= 0:
        return []
    if requested > len(rows):
        raise ValueError(f"requested {requested} rows from only {len(rows)} candidates")
    selected = list(rows)
    rng.shuffle(selected)
    return selected[:requested]


def sample_bucket(
    rows: Sequence[dict[str, Any]],
    *,
    k: int,
    seed: int,
    weight_power: float,
    within_bucket_sampling: str,
) -> list[dict[str, Any]]:
    if within_bucket_sampling == "ns_weighted":
        return weighted_sample_without_replacement(
            rows,
            k=int(k),
            rng=np.random.default_rng(int(seed)),
            weight_power=float(weight_power),
        )
    return sample_without_replacement(rows, k=int(k), rng=random.Random(int(seed)))


def allocate_base_errors(
    candidates: Sequence[dict[str, Any]],
    *,
    per_label: int,
    target_error_count: int,
    mode: str,
) -> dict[int, int]:
    if mode == "equal_per_label":
        return {0: int(round(target_error_count / 2)), 1: int(target_error_count) - int(round(target_error_count / 2))}

    budgets = {0: 0, 1: 0}
    label_summaries: list[tuple[float, int, int]] = []
    for label in (0, 1):
        label_rows = [row for row in candidates if label_value(row) == label]
        error_rows = [row for row in label_rows if is_base_error(row)]
        correct_rows = [row for row in label_rows if not is_base_error(row)]
        max_errors = min(len(error_rows), int(per_label))
        min_errors = max(0, int(per_label) - len(correct_rows))
        budgets[label] = min_errors
        mean_error = float(np.mean([ns_p_error(row) for row in error_rows])) if error_rows else 0.0
        mean_correct = float(np.mean([ns_p_error(row) for row in correct_rows])) if correct_rows else 0.0
        label_summaries.append((mean_error - mean_correct, label, max_errors))

    remaining = int(target_error_count) - sum(budgets.values())
    if remaining < 0:
        raise ValueError(f"target_error_count {target_error_count} is below required minimum {sum(budgets.values())}")
    for _delta, label, max_errors in sorted(label_summaries, reverse=True):
        capacity = max_errors - budgets[label]
        take = min(capacity, remaining)
        budgets[label] += take
        remaining -= take
        if remaining == 0:
            break
    if remaining != 0:
        raise ValueError(f"not enough base-error rows to allocate {target_error_count} errors; short by {remaining}")
    return budgets


def choose_hard_test(
    candidates: Sequence[dict[str, Any]],
    *,
    target_size: int,
    target_base_error_rate: float,
    seed: int,
    weight_power: float,
    label_balance: str,
    error_allocation: str,
    within_bucket_sampling: str,
) -> list[dict[str, Any]]:
    if str(label_balance) == "exact":
        return choose_hard_balanced_test(
            candidates,
            target_size=int(target_size),
            target_base_error_rate=float(target_base_error_rate),
            seed=int(seed),
            weight_power=float(weight_power),
            error_allocation=str(error_allocation),
            within_bucket_sampling=str(within_bucket_sampling),
        )
    if int(target_size) <= 0:
        raise ValueError("--target_size must be a positive integer")
    if not 0.0 < float(target_base_error_rate) < 1.0:
        raise ValueError("--target_base_error_rate must be between 0 and 1")

    target_error_count = int(round(int(target_size) * float(target_base_error_rate)))
    target_correct_count = int(target_size) - target_error_count
    error_rows = [dict(row) for row in candidates if is_base_error(row)]
    correct_rows = [dict(row) for row in candidates if not is_base_error(row)]
    if len(error_rows) < target_error_count:
        raise ValueError(f"only {len(error_rows)} base-error candidates; need {target_error_count}")
    if len(correct_rows) < target_correct_count:
        raise ValueError(f"only {len(correct_rows)} base-correct candidates; need {target_correct_count}")
    selected = [
        *sample_bucket(
            error_rows,
            k=target_error_count,
            seed=int(seed) + 100,
            weight_power=float(weight_power),
            within_bucket_sampling=str(within_bucket_sampling),
        ),
        *sample_bucket(
            correct_rows,
            k=target_correct_count,
            seed=int(seed) + 200,
            weight_power=float(weight_power),
            within_bucket_sampling=str(within_bucket_sampling),
        ),
    ]
    random.Random(int(seed) + 300).shuffle(selected)
    if len({row_id(row) for row in selected}) != len(selected):
        raise ValueError("selected test rows contain duplicate ids")
    return selected


def choose_hard_balanced_test(
    candidates: Sequence[dict[str, Any]],
    *,
    target_size: int,
    target_base_error_rate: float,
    seed: int,
    weight_power: float,
    error_allocation: str,
    within_bucket_sampling: str,
) -> list[dict[str, Any]]:
    if int(target_size) <= 0 or int(target_size) % 2 != 0:
        raise ValueError("--target_size must be a positive even integer")
    if not 0.0 < float(target_base_error_rate) < 1.0:
        raise ValueError("--target_base_error_rate must be between 0 and 1")

    per_label = int(target_size) // 2
    target_error_count = int(round(int(target_size) * float(target_base_error_rate)))
    error_budget_by_label = allocate_base_errors(
        candidates,
        per_label=per_label,
        target_error_count=target_error_count,
        mode=str(error_allocation),
    )
    selected: list[dict[str, Any]] = []
    for label in (0, 1):
        label_rows = [dict(row) for row in candidates if label_value(row) == label]
        error_rows = [row for row in label_rows if is_base_error(row)]
        correct_rows = [row for row in label_rows if not is_base_error(row)]
        error_per_label = int(error_budget_by_label[label])
        correct_per_label = per_label - error_per_label
        if len(error_rows) < error_per_label:
            raise ValueError(f"label {label} has only {len(error_rows)} base-error rows; need {error_per_label}")
        if len(correct_rows) < correct_per_label:
            raise ValueError(f"label {label} has only {len(correct_rows)} base-correct rows; need {correct_per_label}")
        selected.extend(
            sample_bucket(
                error_rows,
                k=error_per_label,
                seed=int(seed) + 100 + label,
                weight_power=float(weight_power),
                within_bucket_sampling=str(within_bucket_sampling),
            )
        )
        selected.extend(
            sample_bucket(
                correct_rows,
                k=correct_per_label,
                seed=int(seed) + 200 + label,
                weight_power=float(weight_power),
                within_bucket_sampling=str(within_bucket_sampling),
            )
        )

    random.Random(int(seed) + 300).shuffle(selected)
    if len({row_id(row) for row in selected}) != len(selected):
        raise ValueError("selected test rows contain duplicate ids")
    return selected


def compute_scored_pool_rows(args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    calibration_rows = read_jsonl(resolve_path(args.calibration_predictions_path))
    pool_rows = read_jsonl(resolve_path(args.pool_predictions_path))
    embeddings_path = resolve_path(args.embeddings_path)
    embeddings_by_id = load_embeddings(embeddings_path)
    assert_embedding_coverage(
        embeddings_by_id,
        [*calibration_rows, *pool_rows],
        expected_dim=int(args.embedding_dim),
    )
    crc = calibrate_crc(
        calibration_rows,
        alpha=float(args.alpha),
        temperature=float(args.temperature),
        embeddings_by_id=embeddings_by_id,
    )
    calibration_decisions = apply_crc_decisions(
        calibration_rows,
        lambda_hat=float(crc.lambda_hat),
        temperature=float(args.temperature),
        embeddings_by_id=embeddings_by_id,
        support_rows=calibration_rows,
        crc_result=crc,
        neighbor_exclude_self=True,
    )
    pool_decisions = apply_crc_decisions(
        pool_rows,
        lambda_hat=float(crc.lambda_hat),
        temperature=float(args.temperature),
        embeddings_by_id=embeddings_by_id,
        support_rows=calibration_rows,
        crc_result=crc,
    )
    sampling_statistics = compute_crc_sampling_statistics(
        calibration_decisions,
        pool_decisions,
        temperature=float(args.temperature),
        lambda_hat=float(crc.lambda_hat),
    )
    ns_scoring = score_ns_difficulty_global(
        pool_decisions,
        calibration_decisions,
        embeddings_by_id=embeddings_by_id,
        e_all=float(sampling_statistics["e_all"]),
    )
    metadata = {
        "temperature": float(args.temperature),
        "alpha": float(args.alpha),
        "lambda_hat": float(crc.lambda_hat),
        "pool_summary": summarize_crc_decisions(pool_decisions),
        "guide_summary": summarize_crc_decisions(calibration_decisions),
        "sampling_statistics": sampling_statistics,
        "ns_scoring": ns_scoring.to_dict(),
    }
    return [dict(row) for row in ns_scoring.pool_rows], metadata


def compare_split_stats(
    *,
    compare_split_ids_path: str,
    rows_by_id: Mapping[str, dict[str, Any]],
) -> dict[str, Any] | None:
    path = resolve_path(compare_split_ids_path)
    if not path.exists():
        return None
    payload = read_json(path)
    ids = [str(sample_id) for sample_id in payload.get("pool_ids", payload.get("test_ids", payload.get("ids", [])))]
    rows = [rows_by_id[sample_id] for sample_id in ids if sample_id in rows_by_id]
    if len(rows) != len(ids):
        missing = [sample_id for sample_id in ids if sample_id not in rows_by_id]
        raise ValueError(f"compare split has missing scored ids: {missing[:5]}")
    return {
        "path": str(path.relative_to(PROJECT_ROOT) if path.is_relative_to(PROJECT_ROOT) else path),
        "stats": row_stats(rows),
    }


def main() -> None:
    args = parse_args()
    started = time.time()
    output_dir = resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    split_payload = read_json(resolve_path(args.split_ids_path))
    scored_pool_rows, scoring_metadata = compute_scored_pool_rows(args)
    rows_by_id = {row_id(row): dict(row) for row in scored_pool_rows}
    blocked_ids, blocked_metadata = load_blocked_ids(split_payload, str(args.train_ids_glob))

    tokenizer = AutoTokenizer.from_pretrained(resolve_path(args.model_path), trust_remote_code=True)
    ensure_tokenizer_padding(tokenizer)
    disable_tokenizer_thinking(tokenizer)
    answer_lengths = answer_token_lengths(tokenizer)
    add_length_metadata(scored_pool_rows, tokenizer, answer_lengths)
    rows_by_id = {row_id(row): dict(row) for row in scored_pool_rows}

    safe_candidates = [
        row
        for row in scored_pool_rows
        if row_id(row) not in blocked_ids and int(row.get("train_seq_tokens", 0) or 0) <= int(args.max_length)
    ]
    selected_rows = choose_hard_test(
        safe_candidates,
        target_size=int(args.target_size),
        target_base_error_rate=float(args.target_base_error_rate),
        seed=int(args.seed),
        weight_power=float(args.weight_power),
        label_balance=str(args.label_balance),
        error_allocation=str(args.error_allocation),
        within_bucket_sampling=str(args.within_bucket_sampling),
    )
    selected_ids = [row_id(row) for row in selected_rows]
    selected_stats = row_stats(selected_rows)
    compare_stats = compare_split_stats(compare_split_ids_path=str(args.compare_split_ids_path), rows_by_id=rows_by_id)

    name = str(args.name)
    id_payload = {"name": name, "count": len(selected_ids), "ids": selected_ids}
    split_test_payload = {
        "name": name,
        "split_algorithm": "label_balanced_target_base_error_weighted_by_round0_ns_p_error",
        "seed": int(args.seed),
        "target_size": int(args.target_size),
        "target_base_error_rate": float(args.target_base_error_rate),
        "label_balance": str(args.label_balance),
        "error_allocation": str(args.error_allocation),
        "within_bucket_sampling": str(args.within_bucket_sampling),
        "weight_power": float(args.weight_power),
        "max_length": int(args.max_length),
        "calibration_ids": [],
        "final_calibration_ids": [],
        "pool_ids": selected_ids,
        "test_ids": selected_ids,
        "ids": selected_ids,
    }
    summary = {
        "stage_name": "cgsd_make_fever_hard_test_set",
        "name": name,
        "elapsed_seconds": round(time.time() - started, 2),
        "selection_policy": {
            "uses_lora_outputs": False,
            "selection_signals": ["round0_base_error"],
            "reported_signals": ["crc_defer", "ns_p_error"],
            "label_balance": str(args.label_balance),
            "target_base_error_rate": float(args.target_base_error_rate),
            "error_allocation": str(args.error_allocation),
            "within_bucket_sampling": (
                f"weighted_without_replacement_ns_p_error_power_{float(args.weight_power):g}"
                if str(args.within_bucket_sampling) == "ns_weighted"
                else "uniform_random_without_replacement"
            ),
        },
        "blocked_counts": blocked_metadata,
        "candidate_counts_after_block_and_length": {
            "total": len(safe_candidates),
            "label_0": sum(1 for row in safe_candidates if label_value(row) == 0),
            "label_1": sum(1 for row in safe_candidates if label_value(row) == 1),
            "base_error": sum(1 for row in safe_candidates if is_base_error(row)),
            "base_correct": sum(1 for row in safe_candidates if not is_base_error(row)),
        },
        "candidate_stats_after_block_and_length": row_stats(safe_candidates),
        "selected_stats": selected_stats,
        "comparison": {"balanced_test": compare_stats},
        "length": {
            "max_length": int(args.max_length),
            "answer_token_lengths": answer_lengths,
        },
        "source_paths": {
            "split_ids_path": display_path(args.split_ids_path),
            "calibration_predictions_path": display_path(args.calibration_predictions_path),
            "pool_predictions_path": display_path(args.pool_predictions_path),
            "embeddings_path": display_path(args.embeddings_path),
            "model_path": str(resolve_path(args.model_path)),
        },
        "scoring": scoring_metadata,
        "ids_path": str(output_dir / f"{name}.ids.json"),
        "jsonl_path": str(output_dir / f"{name}.jsonl"),
        "split_ids_path": str(output_dir / f"{name}.split_ids.json"),
        "vllm_pool_split_ids_path": str(output_dir / f"{name}.vllm_pool_split_ids.json"),
    }

    write_json(id_payload, output_dir / f"{name}.ids.json")
    write_jsonl(selected_rows, output_dir / f"{name}.jsonl")
    write_json(split_test_payload, output_dir / f"{name}.split_ids.json")
    write_json(split_test_payload, output_dir / f"{name}.vllm_pool_split_ids.json")
    write_json(summary, output_dir / f"{name}.summary.json")

    print(
        json.dumps(
            {
                "name": name,
                "selected": selected_stats,
                "balanced_test": compare_stats["stats"] if compare_stats else None,
                "summary_path": str(output_dir / f"{name}.summary.json"),
                "vllm_split": str(output_dir / f"{name}.vllm_pool_split_ids.json"),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
