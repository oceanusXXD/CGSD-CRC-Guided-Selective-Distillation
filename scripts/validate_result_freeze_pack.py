from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mias_dcms.result_freeze_pack import validate_result_freeze_pack
from mias_dcms.utils import read_json, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate a Gate 10 / D8 result freeze pack before paper claim freeze."
    )
    parser.add_argument("--freeze_pack_path", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--expected_main_tables", required=True)
    parser.add_argument("--expected_figures", required=True)
    parser.add_argument("--expected_metrics", required=True)
    parser.add_argument("--expected_baselines", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pack = read_json(Path(args.freeze_pack_path))
    report = validate_result_freeze_pack(
        pack,
        expected_main_tables=_parse_csv(args.expected_main_tables),
        expected_figures=_parse_csv(args.expected_figures),
        expected_metrics=_parse_csv(args.expected_metrics),
        expected_baselines=_parse_csv(args.expected_baselines),
    )
    payload = report.as_dict()
    payload["freeze_pack_path"] = str(args.freeze_pack_path)
    write_json(payload, Path(args.output_path))
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    raise SystemExit(0 if payload["is_ready"] else 1)


def _parse_csv(value: str) -> list[str]:
    return [item.strip() for item in str(value).split(",") if item.strip()]


if __name__ == "__main__":
    main()
