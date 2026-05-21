from __future__ import annotations

from pathlib import Path
import unittest

from scripts.check_ast_integrity import analyze_paths


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class CodebaseCleanlinessTest(unittest.TestCase):
    def test_python_sources_have_clean_ast_import_graph(self) -> None:
        report = analyze_paths(
            [
                PROJECT_ROOT / "src",
                PROJECT_ROOT / "scripts",
            ]
        )

        self.assertEqual([], report.syntax_errors)
        self.assertEqual([], report.missing_internal_imports)
        self.assertEqual([], report.import_cycles)

    def test_scripts_directory_only_contains_public_entrypoints(self) -> None:
        expected = {
            "cgsd_build_embeddings.py",
            "cgsd_cli_common.py",
            "cgsd_compute_crc.py",
            "cgsd_convert_jsonl.py",
            "cgsd_predict_local.py",
            "cgsd_predict_vllm_openai.py",
            "cgsd_prepare.py",
            "cgsd_select_data.py",
            "cgsd_train_round.py",
            "check_ast_integrity.py",
        }
        actual = {path.name for path in (PROJECT_ROOT / "scripts").glob("*.py")}

        self.assertEqual(expected, actual)


if __name__ == "__main__":
    unittest.main()
