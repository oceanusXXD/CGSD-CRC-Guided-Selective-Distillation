from pathlib import Path
import unittest

from scripts.check_ast_integrity import analyze_paths


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class CodebaseCleanlinessTest(unittest.TestCase):
    def test_python_sources_have_clean_ast_import_graph(self) -> None:
        report = analyze_paths(
            [
                PROJECT_ROOT / "algorithms",
                PROJECT_ROOT / "src",
                PROJECT_ROOT / "scripts",
                PROJECT_ROOT / "experiments" / "bin",
            ]
        )

        self.assertEqual([], report.syntax_errors)
        self.assertEqual([], report.missing_internal_imports)
        self.assertEqual([], report.import_cycles)

    def test_obsolete_one_off_entrypoints_are_removed(self) -> None:
        obsolete_paths = [
            "experiments/bin/cgsd_average_lora_adapters.py",
            "experiments/bin/cgsd_collect_results.py",
            "experiments/bin/cgsd_lrobench_ordered_split.py",
            "experiments/bin/cgsd_split_lrobench_inputs.py",
            "experiments/bin/run_fever_formula_ratio_lora_500.py",
            "experiments/bin/run_fever_unbalanced_lora_curve.py",
            "experiments/bin/run_lrobench_transfer_one.py",
            "experiments/bin/run_lrobench_transfer_one_overwrite.py",
            "scripts/cgsd_defer_only_real_run.py",
            "scripts/cgsd_predict_defer_only_round.py",
        ]

        existing = [path for path in obsolete_paths if (PROJECT_ROOT / path).exists()]
        self.assertEqual([], existing)


if __name__ == "__main__":
    unittest.main()
