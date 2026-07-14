from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mias_dcms.binary_benchmark_data import (  # noqa: E402
    BINARY_BENCHMARK_SPECS,
    BinaryBenchmarkSpec,
    EmptyBinaryBenchmarkTextError,
    normalize_binary_benchmark_row,
    validate_normalized_binary_rows,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download and normalize native-label binary benchmarks for MIAS re-audits."
    )
    parser.add_argument(
        "--datasets",
        default=",".join(BINARY_BENCHMARK_SPECS),
        help="Comma-separated names: " + ", ".join(BINARY_BENCHMARK_SPECS),
    )
    parser.add_argument(
        "--output_root",
        type=Path,
        default=Path("experiments/inputs/binary"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    requested = _parse_datasets(str(args.datasets))
    summaries = [
        _download_or_reuse(spec=BINARY_BENCHMARK_SPECS[name], output_root=args.output_root)
        for name in requested
    ]
    print(json.dumps({"datasets": summaries}, ensure_ascii=False, indent=2, sort_keys=True))


def _parse_datasets(value: str) -> list[str]:
    names = [item.strip() for item in value.split(",") if item.strip()]
    unknown = sorted(set(names) - set(BINARY_BENCHMARK_SPECS))
    if unknown:
        raise ValueError(f"unsupported binary benchmark names: {unknown}")
    if not names:
        raise ValueError("at least one binary benchmark is required")
    return names


def _download_or_reuse(*, spec: BinaryBenchmarkSpec, output_root: Path) -> dict[str, Any]:
    output_dir = output_root / spec.name
    manifest_path = output_dir / "source_manifest.json"
    if manifest_path.exists():
        payload = _read_json(manifest_path)
        _validate_existing_manifest(payload, spec=spec, output_dir=output_dir)
        return {"dataset": spec.name, "status": "reused", "manifest_path": str(manifest_path)}
    if output_dir.exists():
        raise FileExistsError(
            f"{output_dir} exists without source_manifest.json; remove it after inspection before retrying"
        )

    try:
        from datasets import load_dataset
    except ImportError as exc:  # pragma: no cover - depends on local runtime setup
        raise ImportError(
            "datasets is required. Install the pinned requirements into the Python environment running this script."
        ) from exc

    loaded = load_dataset(spec.repo_id, spec.config, revision=spec.revision)
    observed_splits = tuple(sorted(str(name) for name in loaded.keys()))
    missing_splits = sorted(set(spec.expected_splits) - set(observed_splits))
    if missing_splits:
        raise ValueError(
            f"{spec.name} is missing expected splits {missing_splits}; got {observed_splits}"
        )

    split_summary: dict[str, dict[str, Any]] = {}
    for split in spec.expected_splits:
        normalized: list[dict[str, object]] = []
        dropped_rows: list[dict[str, object]] = []
        for index, row in enumerate(loaded[split]):
            try:
                normalized.append(normalize_binary_benchmark_row(spec.name, row, split=split, index=index))
            except EmptyBinaryBenchmarkTextError as exc:
                dropped_rows.append(
                    {
                        "source_index": index,
                        "native_label": _source_binary_label(row, dataset_name=spec.name, index=index),
                        "reason": str(exc),
                    }
                )
        label_counts = validate_normalized_binary_rows(
            normalized,
            dataset_name=spec.name,
            split=split,
        )
        output_path = output_dir / f"{split}.jsonl"
        _write_jsonl_atomic(normalized, output_path)
        split_summary[split] = {
            "path": str(output_path),
            "source_row_count": len(loaded[split]),
            "row_count": len(normalized),
            "label_counts": label_counts,
            "dropped_empty_input_rows": dropped_rows,
            "sha256": _sha256_file(output_path),
        }

    manifest = {
        "schema_version": "binary-benchmark-input-v1",
        "dataset": spec.name,
        "source": {
            "repo_id": spec.repo_id,
            "config": spec.config,
            "revision": spec.revision,
            "dataset_card_url": spec.dataset_card_url,
            "label_policy": "native_binary_label_copied_without_remapping",
            "label_names": {"0": spec.label_names[0], "1": spec.label_names[1]},
            "ignored_source_splits": sorted(set(observed_splits) - set(spec.expected_splits)),
        },
        "record_schema": {
            "id": "stable source-derived identifier",
            "query": "task instruction or first sentence",
            "document": "source text or second sentence",
            "groundtruth": "native source binary label",
        },
        "splits": split_summary,
    }
    _write_json_atomic(manifest, manifest_path)
    return {"dataset": spec.name, "status": "downloaded", "manifest_path": str(manifest_path)}


def _validate_existing_manifest(
    payload: dict[str, Any],
    *,
    spec: BinaryBenchmarkSpec,
    output_dir: Path,
) -> None:
    source = payload.get("source", {})
    if not isinstance(source, dict) or source.get("revision") != spec.revision:
        raise ValueError(f"existing {spec.name} manifest does not match the pinned source revision")
    splits = payload.get("splits", {})
    if not isinstance(splits, dict) or set(splits) != set(spec.expected_splits):
        raise ValueError(f"existing {spec.name} manifest has unexpected split metadata")
    for split in spec.expected_splits:
        details = splits[split]
        path = output_dir / f"{split}.jsonl"
        if not path.exists() or not isinstance(details, dict):
            raise ValueError(f"existing {spec.name} is missing {split}.jsonl")
        if _sha256_file(path) != details.get("sha256"):
            raise ValueError(f"existing {spec.name}/{split} hash does not match its manifest")


def _write_jsonl_atomic(rows: Iterable[dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
            handle.write("\n")
    temporary.replace(path)


def _write_json_atomic(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected an object in {path}")
    return payload


def _source_binary_label(row: Any, *, dataset_name: str, index: int) -> int:
    try:
        label = int(row["label"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"{dataset_name} row {index} has an invalid native label") from exc
    if label not in (0, 1):
        raise ValueError(f"{dataset_name} row {index} must have a native binary label")
    return label


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    main()
