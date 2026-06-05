from __future__ import annotations

import ast
from pathlib import Path
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class CodebaseCleanlinessTest(unittest.TestCase):
    def test_python_sources_parse(self) -> None:
        for path in sorted((PROJECT_ROOT / "src").glob("*.py")) + sorted((PROJECT_ROOT / "scripts").glob("*.py")):
            with self.subTest(path=path.relative_to(PROJECT_ROOT).as_posix()):
                ast.parse(path.read_text(encoding="utf-8"), filename=str(path))

    def test_scripts_directory_only_contains_public_entrypoints(self) -> None:
        expected = {
            "build_embeddings.py",
            "cli_common.py",
            "compute_crc.py",
            "convert_jsonl.py",
            "make_pcss_eval_split.py",
            "predict_local.py",
            "predict_vllm_openai.py",
            "prepare.py",
            "select_crc_error_mass.py",
            "select_pcss.py",
            "select_random.py",
            "selection_common.py",
            "train_round.py",
        }
        actual = {path.name for path in (PROJECT_ROOT / "scripts").glob("*.py")}

        self.assertEqual(expected, actual)


if __name__ == "__main__":
    unittest.main()
