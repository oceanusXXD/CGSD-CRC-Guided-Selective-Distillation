from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from mias_dcms.utils import resolve_model_reference


class UtilsTest(unittest.TestCase):
    def test_resolve_model_reference_finds_workspace_sibling_models_directory(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = Path(tmpdir)
            project_root = workspace / "project"
            model_path = workspace / "models" / "qwen3-0.6b"
            project_root.mkdir()
            model_path.mkdir(parents=True)

            resolved = resolve_model_reference("qwen3-0.6b", project_root)

        self.assertEqual(str(model_path), resolved)


if __name__ == "__main__":
    unittest.main()
