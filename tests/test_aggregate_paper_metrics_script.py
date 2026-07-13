from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path
import unittest

from mias_dcms.utils import write_jsonl


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = PROJECT_ROOT / "scripts" / "aggregate_paper_metrics.py"


def _run_row(method: str, seed: int, macro_f1: float, acquisition_tv: float) -> dict[str, object]:
    return {
        "dataset": "toy",
        "model": "model-a",
        "method": method,
        "budget": 4,
        "seed": seed,
        "selected_count": 4,
        "config_hash": f"cfg-{method}-{seed}",
        "selection_metrics": {"acquisition_tv": acquisition_tv},
        "training_metrics": {},
        "evaluation_metrics": {"macro_f1": macro_f1, "worst_group": macro_f1 - 0.1},
        "cost_metrics": {
            "seed_label_count": 2,
            "active_label_count": 4,
            "evaluation_label_count": 6,
            "judge_calls": 0,
            "train_tokens": 1000 + seed,
            "selector_compute_seconds": 0.25,
        },
    }


class AggregatePaperMetricsScriptTest(unittest.TestCase):
    def test_script_aggregates_run_level_jsonl_to_paper_level_table(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            input_path = tmp / "runs.jsonl"
            output_path = tmp / "paper_table.json"
            write_jsonl(
                [
                    _run_row("Random", 1, 0.60, 0.20),
                    _run_row("Random", 2, 0.70, 0.25),
                    _run_row("Entropy+DCMS", 1, 0.68, 0.06),
                    _run_row("Entropy+DCMS", 2, 0.72, 0.08),
                ],
                input_path,
            )

            subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "--input_path",
                    str(input_path),
                    "--output_path",
                    str(output_path),
                    "--evaluation_metrics",
                    "macro_f1,worst_group",
                    "--selection_metrics",
                    "acquisition_tv",
                    "--cost_metrics",
                    "train_tokens",
                    "--resamples",
                    "200",
                    "--seed",
                    "9",
                ],
                check=True,
                cwd=PROJECT_ROOT,
            )

            payload = json.loads(output_path.read_text(encoding="utf-8"))
            by_method = {row["method"]: row for row in payload["paper_metric_table"]}

            self.assertEqual(4, payload["run_count"])
            self.assertEqual({"Entropy+DCMS", "Random"}, set(by_method))
            self.assertAlmostEqual(0.65, by_method["Random"]["evaluation_metrics"]["macro_f1"]["mean"])
            self.assertAlmostEqual(0.07, by_method["Entropy+DCMS"]["selection_metrics"]["acquisition_tv"]["mean"])
            self.assertEqual(["macro_f1", "worst_group"], payload["evaluation_metrics"])


if __name__ == "__main__":
    unittest.main()
