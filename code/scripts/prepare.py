
from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.cli_common import (
    add_stage_cache_args,
    embedding_usage_payload,
    output_dir_from_arg,
    output_artifact_path,
    print_existing_stage_result,
    stage_cache_decision,
    write_stage_usage,
)
from src.binary_protocol import normalize_binary_label
from src.data import examples_to_rows, load_examples
from src.embeddings import assert_embedding_coverage, load_embeddings
from src.utils import resolve_input_path, set_seed, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_path", default="datasets")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--split_ids_path", default=None)
    parser.add_argument("--usage_path", default=None)
    parser.add_argument("--query_field", default="query")
    parser.add_argument("--document_field", default="document")
    parser.add_argument("--label_field", default="groundtruth")
    parser.add_argument("--embeddings_path", default=None)
    parser.add_argument("--embedding_dim", type=int, default=0)
    parser.add_argument("--n_guide", type=int, default=None)
    parser.add_argument("--n_final", type=int, default=None)
    parser.add_argument("--split_strategy", choices=["stratified", "random"], default="stratified")
    parser.add_argument("--seed", type=int, default=42)
    add_stage_cache_args(parser)
    return parser.parse_args()


def _round_half_up(value: float) -> int:
    return int(math.floor(float(value) + 0.5))


def _row_id(row: dict[str, object]) -> str:
    sample_id = row.get("id")
    if sample_id is None or str(sample_id) == "":
        raise ValueError(f"row missing id: {row!r}")
    return str(sample_id)


def _row_label(row: dict[str, object]) -> int:
    return normalize_binary_label(row.get("label", row.get("groundtruth")), field_name="row label")


def _label_distribution(rows_by_id: dict[str, dict[str, object]], ids: list[str]) -> dict[str, object]:
    label1_count = sum(1 for sample_id in ids if _row_label(rows_by_id[sample_id]) == 1)
    size = len(ids)
    return {
        "label0_count": int(size - label1_count),
        "label1_count": int(label1_count),
        "size": int(size),
        "label1_rate": float(label1_count / size) if size else 0.0,
    }


def _allocate_label1_count(*, requested_size: int, remaining_label0: int, remaining_label1: int) -> int:
    remaining_total = remaining_label0 + remaining_label1
    if requested_size <= 0 or remaining_total <= 0:
        return 0
    target_label1 = _round_half_up(requested_size * (remaining_label1 / remaining_total))
    min_label1 = max(0, requested_size - remaining_label0)
    max_label1 = min(requested_size, remaining_label1)
    return max(min_label1, min(max_label1, target_label1))


def _take_stratified_ids(
    *,
    remaining_by_label: dict[int, list[str]],
    requested_size: int,
) -> list[str]:
    if requested_size <= 0:
        return []
    label1_count = _allocate_label1_count(
        requested_size=requested_size,
        remaining_label0=len(remaining_by_label[0]),
        remaining_label1=len(remaining_by_label[1]),
    )
    label0_count = requested_size - label1_count
    selected_label0 = remaining_by_label[0][:label0_count]
    selected_label1 = remaining_by_label[1][:label1_count]
    del remaining_by_label[0][:label0_count]
    del remaining_by_label[1][:label1_count]
    return [*selected_label0, *selected_label1]


def _split_ids_stratified(
    rows: list[dict[str, object]],
    *,
    n_guide: int,
    n_final: int,
    rng: random.Random,
) -> tuple[list[str], list[str], list[str]]:
    remaining_by_label = {0: [], 1: []}
    for row in rows:
        remaining_by_label[_row_label(row)].append(_row_id(row))
    rng.shuffle(remaining_by_label[0])
    rng.shuffle(remaining_by_label[1])

    guide_ids = _take_stratified_ids(remaining_by_label=remaining_by_label, requested_size=n_guide)
    final_ids = _take_stratified_ids(remaining_by_label=remaining_by_label, requested_size=n_final)
    pool_ids = [*remaining_by_label[0], *remaining_by_label[1]]
    rng.shuffle(guide_ids)
    rng.shuffle(final_ids)
    rng.shuffle(pool_ids)
    return guide_ids, final_ids, pool_ids


def _split_ids_random(
    rows: list[dict[str, object]],
    *,
    n_guide: int,
    n_final: int,
    rng: random.Random,
) -> tuple[list[str], list[str], list[str]]:
    all_ids = [_row_id(row) for row in rows]
    rng.shuffle(all_ids)
    guide_ids = all_ids[:n_guide]
    final_ids = all_ids[n_guide : n_guide + n_final]
    pool_ids = all_ids[n_guide + n_final :]
    return guide_ids, final_ids, pool_ids


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    output_dir = output_dir_from_arg(args.output_dir)
    split_ids_path = output_artifact_path(args.split_ids_path, output_dir / "split_ids.json")
    usage_path = output_artifact_path(args.usage_path, output_dir / "prepare_usage.json")
    if args.show_result:
        print_existing_stage_result(stage_name="prepare", summary_path=split_ids_path)
        return
    cache_decision = stage_cache_decision(
        stage_name="prepare",
        required_outputs=[split_ids_path, usage_path],
        cache_policy=args.cache_policy,
    )
    if cache_decision.cache_hit:
        print_existing_stage_result(stage_name="prepare", summary_path=split_ids_path)
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    data_path = resolve_input_path(args.data_path, PROJECT_ROOT)

    examples = load_examples(
        data_path,
        query_field=args.query_field,
        document_field=args.document_field,
        label_field=args.label_field,
    )
    rows = examples_to_rows(examples)
    n_guide = 1000 if args.n_guide is None else int(args.n_guide)
    n_final = 0 if args.n_final is None else int(args.n_final)
    if n_guide <= 0:
        raise ValueError("--n_guide must be positive")
    if n_final < 0:
        raise ValueError("--n_final must be non-negative")
    if n_guide + n_final >= len(rows):
        raise ValueError("--n_guide + --n_final must be smaller than the dataset size")
    rows_by_id = {_row_id(row): row for row in rows}
    rng = random.Random(int(args.seed))
    if args.split_strategy == "stratified":
        guide_ids, final_ids, pool_ids = _split_ids_stratified(rows, n_guide=n_guide, n_final=n_final, rng=rng)
        split_algorithm = "stratified_label_guide_final_pool_v1"
    else:
        guide_ids, final_ids, pool_ids = _split_ids_random(rows, n_guide=n_guide, n_final=n_final, rng=rng)
        split_algorithm = "fixed_random_guide_final_pool_v1"
    label_distribution = {
        "all": _label_distribution(rows_by_id, [_row_id(row) for row in rows]),
        "guide": _label_distribution(rows_by_id, guide_ids),
        "final": _label_distribution(rows_by_id, final_ids),
        "pool": _label_distribution(rows_by_id, pool_ids),
    }
    embedding_usage = None
    if args.embeddings_path:
        embeddings_path = resolve_input_path(args.embeddings_path, PROJECT_ROOT)
        embeddings_by_id = load_embeddings(embeddings_path)
        assert_embedding_coverage(embeddings_by_id, rows, expected_dim=int(args.embedding_dim))
        embedding_usage = embedding_usage_payload(
            embedding_source=embeddings_path,
            row_count=len(rows),
            embedding_dim=int(args.embedding_dim),
            purpose="prepare_validate_full_embedding_coverage",
        )

    split_payload = {
        "guide_ids": guide_ids,
        "final_ids": final_ids,
        "pool_ids": pool_ids,
        "n_guide": n_guide,
        "n_final": n_final,
        "seed": args.seed,
        "split_algorithm": split_algorithm,
        "split_strategy": str(args.split_strategy),
        "label_distribution": label_distribution,
    }
    write_json(split_payload, split_ids_path)
    usage_payload = {
        "stage_name": "prepare",
        "cache": cache_decision.to_dict(),
        "data_path": str(data_path),
        "split_ids_path": str(split_ids_path),
        "data_rows": len(rows),
        "guide_size": len(guide_ids),
        "final_size": len(final_ids),
        "pool_size": len(pool_ids),
        "split_strategy": str(args.split_strategy),
        "label_distribution": label_distribution,
        "student_model_calls": 0,
        "teacher_calls": 0,
        "groundtruth_substitute_calls": 0,
    }
    if embedding_usage is not None:
        usage_payload["embedding"] = embedding_usage
    write_stage_usage(usage_path, usage_payload)
    print(json.dumps({"prepared": True, "output_dir": str(output_dir)}, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
