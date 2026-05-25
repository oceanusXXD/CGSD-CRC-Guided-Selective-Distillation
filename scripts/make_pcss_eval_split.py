#!/usr/bin/env python
"""Build the post-training PCSS eval split.

The output split keeps the locked certification rows in `final_ids` and places
the reusable common test set in `pool_ids`. `guide_ids` is empty because guide
participated in selection diagnostics and must not be used for post-training
calibration or evaluation.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.cli_common import CACHE_POLICIES, input_artifact_path, output_artifact_path


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _row_id(row: dict[str, Any]) -> str:
    sample_id = row.get("id", row.get("sample_id"))
    if sample_id is None or str(sample_id) == "":
        raise ValueError(f"row missing id/sample_id: {row!r}")
    return str(sample_id)


def _label(row: dict[str, Any]) -> str:
    value = row.get("label", row.get("groundtruth"))
    if value is None:
        raise ValueError(f"row missing label/groundtruth: {row!r}")
    return str(int(value))


def _ensure_unique(ids: list[str], *, name: str) -> None:
    counts = Counter(ids)
    duplicates = sorted(sample_id for sample_id, count in counts.items() if count > 1)
    if duplicates:
        raise ValueError(f"{name} contains duplicate ids; first={duplicates[0]!r}")


def _label_counts(ids: list[str], rows_by_id: dict[str, dict[str, Any]]) -> dict[str, int]:
    counts = Counter(_label(rows_by_id[sample_id]) for sample_id in ids)
    return {str(key): int(counts[key]) for key in sorted(counts)}


def build_pcss_eval_split(
    *,
    data_path: str | Path,
    split_ids_path: str | Path,
    train_rows_path: str | Path,
    output_path: str | Path,
    summary_path: str | Path,
) -> dict[str, Any]:
    data_source = Path(data_path)
    split_source = Path(split_ids_path)
    train_source = Path(train_rows_path)
    output = Path(output_path)
    summary_output = Path(summary_path)

    rows = _read_jsonl(data_source)
    data_ids = [_row_id(row) for row in rows]
    _ensure_unique(data_ids, name=str(data_source))
    rows_by_id = {_row_id(row): row for row in rows}

    split_payload = _read_json(split_source)
    guide_ids = [str(sample_id) for sample_id in split_payload["guide_ids"]]
    final_ids = [str(sample_id) for sample_id in split_payload["final_ids"]]
    pool_ids = [str(sample_id) for sample_id in split_payload["pool_ids"]]
    for name, ids in (("guide_ids", guide_ids), ("final_ids", final_ids), ("pool_ids", pool_ids)):
        _ensure_unique(ids, name=name)
        missing = [sample_id for sample_id in ids if sample_id not in rows_by_id]
        if missing:
            raise ValueError(f"{name} contains ids missing from data; first={missing[0]!r}")

    split_all_ids = set(guide_ids) | set(final_ids) | set(pool_ids)
    missing_from_split = [sample_id for sample_id in data_ids if sample_id not in split_all_ids]
    if missing_from_split:
        raise ValueError(f"split ids do not cover data; first missing id={missing_from_split[0]!r}")

    locked_ids = set(guide_ids) | set(final_ids)
    train_rows = _read_jsonl(train_source)
    train_ids = [_row_id(row) for row in train_rows]
    _ensure_unique(train_ids, name=str(train_source))
    train_id_set = set(train_ids)
    train_locked_overlap = sorted(train_id_set & locked_ids)
    if train_locked_overlap:
        raise ValueError(
            "selected training rows overlaps locked guide/final ids; "
            f"first={train_locked_overlap[0]!r}, overlap_count={len(train_locked_overlap)}"
        )
    pool_id_set = set(pool_ids)
    train_not_pool = sorted(train_id_set - pool_id_set)
    if train_not_pool:
        raise ValueError(
            "selected training rows must come from the PCSS pool; "
            f"first={train_not_pool[0]!r}, count={len(train_not_pool)}"
        )

    pool_test_ids = [sample_id for sample_id in pool_ids if sample_id not in train_id_set]
    eval_split = {
        "guide_ids": [],
        "final_ids": final_ids,
        "pool_ids": pool_test_ids,
    }
    summary = {
        "stage_name": "make_pcss_eval_split",
        "data_path": str(data_source),
        "split_ids_path": str(split_source),
        "train_rows_path": str(train_source),
        "output_path": str(output),
        "summary_path": str(summary_output),
        "data_count": len(data_ids),
        "guide_count": len(guide_ids),
        "cert_count": len(final_ids),
        "pool_candidate_count": len(pool_ids),
        "selected_train_count": len(train_ids),
        "pool_test_count": len(pool_test_ids),
        "pool_test_label_counts": _label_counts(pool_test_ids, rows_by_id),
        "cert_label_counts": _label_counts(final_ids, rows_by_id),
        "excluded_from_test_count": len(guide_ids) + len(final_ids) + len(train_ids),
        "uses_full_dataset_as_test": len(pool_test_ids) == len(data_ids),
        "guide_ids_used_for_eval": False,
        "cert_ids_used_as_final_split": True,
    }
    _write_json(output, eval_split)
    _write_json(summary_output, summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--split_ids_path", required=True)
    parser.add_argument("--train_rows_path", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--summary_path", default=None)
    parser.add_argument("--cache_policy", choices=CACHE_POLICIES, default="reuse")
    parser.add_argument("--show_result", action="store_true", default=False)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_path = input_artifact_path(args.data_path, PROJECT_ROOT / str(args.data_path))
    split_ids_path = input_artifact_path(args.split_ids_path, PROJECT_ROOT / str(args.split_ids_path))
    train_rows_path = input_artifact_path(args.train_rows_path, PROJECT_ROOT / str(args.train_rows_path))
    output_path = output_artifact_path(args.output_path, PROJECT_ROOT / str(args.output_path))
    summary_path = output_artifact_path(
        args.summary_path,
        output_path.with_suffix(".summary.json"),
    )
    outputs_exist = output_path.exists() and summary_path.exists()
    if args.show_result:
        if not summary_path.exists():
            raise FileNotFoundError(f"summary does not exist: {summary_path}")
        print(summary_path.read_text(encoding="utf-8").strip())
        return
    if outputs_exist and args.cache_policy == "fail":
        raise FileExistsError(f"PCSS eval split outputs already exist: {output_path}, {summary_path}")
    if outputs_exist and args.cache_policy == "reuse":
        print(summary_path.read_text(encoding="utf-8").strip())
        return

    summary = build_pcss_eval_split(
        data_path=data_path,
        split_ids_path=split_ids_path,
        train_rows_path=train_rows_path,
        output_path=output_path,
        summary_path=summary_path,
    )
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
