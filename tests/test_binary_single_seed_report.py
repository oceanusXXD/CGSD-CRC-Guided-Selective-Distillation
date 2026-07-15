from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from mias_dcms.binary_single_seed_report import build_binary_single_seed_gate_summary
from mias_dcms.utils import write_json, write_jsonl


class BinarySingleSeedReportTest(unittest.TestCase):
    def test_collects_completed_shared_entropy_margin_run(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            config_path = root / "config.json"
            protocol_path = root / "protocol.json"
            source_path = root / "source.json"
            write_json({"frozen": True}, config_path)
            write_json({"dataset": "toy"}, protocol_path)
            write_json({"dataset": "toy"}, source_path)
            import hashlib

            config_hash = hashlib.sha256(config_path.read_bytes()).hexdigest()
            for method, selected_ids in {
                "random": ["a", "b"],
                "entropy": ["c", "d"],
                "margin": ["c", "d"],
            }.items():
                write_json(
                    {
                        "method": method,
                        "dataset": "toy",
                        "seed": 17,
                        "budget": 2,
                        "config_hash": config_hash,
                        "selected_ids": selected_ids,
                    },
                    root / "selection" / "seed_17" / method / "selection_summary.json",
                )
            _write_completed_run(root, "random", metrics={"accuracy": 0.8, "macro_F1": 0.7})
            _write_completed_run(root, "entropy_margin", metrics={"accuracy": 0.6, "macro_F1": 0.5})

            summary = build_binary_single_seed_gate_summary(
                run_root=root,
                dataset="toy",
                config_snapshot_path=config_path,
                selection_config_snapshot_path=None,
                protocol_manifest_path=protocol_path,
                source_manifest_path=source_path,
                seed=17,
                expected_test_size=2,
            )

            self.assertEqual("completed_single_seed_feasibility_gate", summary["status"])
            self.assertEqual(1.0, summary["entropy_margin_equivalence"]["jaccard"])
            self.assertAlmostEqual(0.2, summary["random_minus_entropy_margin_fixed_test_delta"]["accuracy"])
            self.assertEqual(2, summary["selection_methods"]["random"]["fixed_test_prediction_count"])

    def test_rejects_nonidentical_entropy_margin_selection(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            config_path = root / "config.json"
            protocol_path = root / "protocol.json"
            write_json({}, config_path)
            write_json({}, protocol_path)
            import hashlib

            config_hash = hashlib.sha256(config_path.read_bytes()).hexdigest()
            for method, selected_ids in {
                "random": ["a"],
                "entropy": ["b"],
                "margin": ["c"],
            }.items():
                write_json(
                    {
                        "method": method,
                        "dataset": "toy",
                        "seed": 17,
                        "budget": 1,
                        "config_hash": config_hash,
                        "selected_ids": selected_ids,
                    },
                    root / "selection" / "seed_17" / method / "selection_summary.json",
                )

            with self.assertRaisesRegex(ValueError, "different ids"):
                build_binary_single_seed_gate_summary(
                    run_root=root,
                    dataset="toy",
                    config_snapshot_path=config_path,
                    selection_config_snapshot_path=None,
                    protocol_manifest_path=protocol_path,
                    source_manifest_path=None,
                    seed=17,
                    expected_test_size=2,
                )

    def test_marks_different_selection_and_execution_config_as_exception(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            config_path = root / "execution_config.json"
            selection_config_path = root / "selection_config.json"
            protocol_path = root / "protocol.json"
            write_json({"phase": "execution"}, config_path)
            write_json({"phase": "selection"}, selection_config_path)
            write_json({"dataset": "toy"}, protocol_path)
            import hashlib

            selection_config_hash = hashlib.sha256(selection_config_path.read_bytes()).hexdigest()
            for method, selected_ids in {
                "random": ["a", "b"],
                "entropy": ["c", "d"],
                "margin": ["c", "d"],
            }.items():
                write_json(
                    {
                        "method": method,
                        "dataset": "toy",
                        "seed": 17,
                        "budget": 2,
                        "config_hash": selection_config_hash,
                        "selected_ids": selected_ids,
                    },
                    root / "selection" / "seed_17" / method / "selection_summary.json",
                )
            _write_completed_run(root, "random", metrics={"accuracy": 0.8})
            _write_completed_run(root, "entropy_margin", metrics={"accuracy": 0.6})

            summary = build_binary_single_seed_gate_summary(
                run_root=root,
                dataset="toy",
                config_snapshot_path=config_path,
                selection_config_snapshot_path=selection_config_path,
                protocol_manifest_path=protocol_path,
                source_manifest_path=None,
                seed=17,
                expected_test_size=2,
            )

            self.assertEqual("completed_with_config_provenance_exception", summary["status"])
            self.assertFalse(summary["config_provenance_aligned"])


def _write_completed_run(root: Path, method: str, *, metrics: dict[str, float]) -> None:
    round_dir = root / "method_runs" / "seed_17" / method / "round_1"
    model_dir = round_dir / "model"
    write_json({"run_complete": True}, model_dir / "training_state.json")
    write_json({"input_format": "chat_binary"}, model_dir / "model_config.json")
    write_json({"input_format": "chat_binary"}, round_dir / "training_round_summary.json")
    write_json(metrics, round_dir / "fixed_test_metrics.json")
    write_jsonl([{"id": "x"}, {"id": "y"}], round_dir / "fixed_test_predictions.jsonl")


if __name__ == "__main__":
    unittest.main()
