from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.cgsd_make_fever_balanced_subsets import (  # noqa: E402
    build_accept_defer_balanced_ids,
    build_label_balanced_ids,
)


def _row(sample_id: str, label: int, *, defer: bool = False) -> dict:
    return {"id": sample_id, "groundtruth": label, "label": label, "defer": defer}


def test_build_label_balanced_ids_selects_exact_half_per_label():
    rows = [_row(f"pos_{i}", 1) for i in range(20)] + [_row(f"neg_{i}", 0) for i in range(20)]

    selected = build_label_balanced_ids(rows, size=10, seed=7)

    selected_rows = {row["id"]: row for row in rows}
    labels = [selected_rows[sample_id]["groundtruth"] for sample_id in selected]
    assert len(selected) == 10
    assert len(set(selected)) == 10
    assert labels.count(1) == 5
    assert labels.count(0) == 5


def test_build_accept_defer_balanced_ids_preserves_ratio_and_label_balance():
    rows = []
    for label in (0, 1):
        rows.extend(_row(f"accept_{label}_{i}", label, defer=False) for i in range(20))
        rows.extend(_row(f"defer_{label}_{i}", label, defer=True) for i in range(40))

    selected = build_accept_defer_balanced_ids(rows, size=20, accept_fraction=0.15, seed=3)

    selected_rows = {row["id"]: row for row in rows}
    labels = [selected_rows[sample_id]["groundtruth"] for sample_id in selected]
    accepts = [sample_id for sample_id in selected if not selected_rows[sample_id]["defer"]]
    defers = [sample_id for sample_id in selected if selected_rows[sample_id]["defer"]]
    assert len(selected) == 20
    assert len(set(selected)) == 20
    assert len(accepts) == 3
    assert len(defers) == 17
    assert labels.count(1) == 10
    assert labels.count(0) == 10


if __name__ == "__main__":
    test_build_label_balanced_ids_selects_exact_half_per_label()
    test_build_accept_defer_balanced_ids_preserves_ratio_and_label_balance()
