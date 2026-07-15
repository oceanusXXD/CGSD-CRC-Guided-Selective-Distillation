from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mias_dcms.multiclass_training_rows import build_multiclass_training_rows  # noqa: E402
from mias_dcms.utils import read_jsonl, write_json, write_jsonl  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build one auditable multiclass LoRA training set from fixed seed and selected rows."
    )
    parser.add_argument("--seed_rows_path", type=Path, required=True)
    parser.add_argument("--selected_rows_path", type=Path, required=True)
    parser.add_argument("--method", required=True)
    parser.add_argument("--output_path", type=Path, required=True)
    parser.add_argument("--manifest_path", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows, summary = build_multiclass_training_rows(
        read_jsonl(args.seed_rows_path),
        read_jsonl(args.selected_rows_path),
    )
    write_jsonl(rows, args.output_path)
    manifest = {
        "schema_version": "multiclass-training-rows-v1",
        "method": str(args.method),
        "seed_rows_path": str(args.seed_rows_path),
        "seed_rows_sha256": _sha256_file(args.seed_rows_path),
        "selected_rows_path": str(args.selected_rows_path),
        "selected_rows_sha256": _sha256_file(args.selected_rows_path),
        "output_path": str(args.output_path),
        **summary,
    }
    write_json(manifest, args.manifest_path)
    print(json.dumps(manifest, ensure_ascii=False, sort_keys=True))


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    main()
