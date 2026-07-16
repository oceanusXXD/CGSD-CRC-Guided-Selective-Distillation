from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from mias_dcms.selectors import assert_selector_rows_are_label_safe


def merge_feature_rows(
    rows: Iterable[Mapping[str, Any]],
    feature_rows: Iterable[Mapping[str, Any]],
    *,
    source_name: str,
) -> list[dict[str, Any]]:
    """Join a label-safe feature artifact with exact ID coverage."""
    materialized = [dict(row) for row in rows]
    features = [dict(row) for row in feature_rows]
    assert_selector_rows_are_label_safe(features)
    feature_by_id = _unique_rows_by_id(features, source_name=source_name)
    row_ids = [_row_id(row, source_name="rows") for row in materialized]
    if len(set(row_ids)) != len(row_ids):
        raise ValueError("rows contain duplicate sample ids")
    missing = sorted(set(row_ids) - set(feature_by_id))
    extra = sorted(set(feature_by_id) - set(row_ids))
    if missing or extra:
        raise ValueError(
            f"{source_name} must exactly cover rows: "
            f"missing={len(missing)}, extra={len(extra)}"
        )
    return [
        {
            **row,
            **{
                key: value
                for key, value in feature_by_id[_row_id(row, source_name="rows")].items()
                if key not in {"id", "sample_id"}
            },
        }
        for row in materialized
    ]


def _unique_rows_by_id(
    rows: list[dict[str, Any]],
    *,
    source_name: str,
) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for row in rows:
        sample_id = _row_id(row, source_name=source_name)
        if sample_id in output:
            raise ValueError(f"{source_name} contains duplicate id {sample_id!r}")
        output[sample_id] = row
    return output


def _row_id(row: Mapping[str, Any], *, source_name: str) -> str:
    value = row.get("sample_id", row.get("id"))
    if value is None or not str(value):
        raise ValueError(f"{source_name} row is missing a non-empty sample_id/id")
    return str(value)
