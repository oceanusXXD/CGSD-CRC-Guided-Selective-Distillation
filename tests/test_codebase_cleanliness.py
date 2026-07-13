from __future__ import annotations

import ast
from pathlib import Path
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class CodebaseCleanlinessTest(unittest.TestCase):
    def test_python_sources_parse(self) -> None:
        source_paths = sorted((PROJECT_ROOT / "mias_dcms").rglob("*.py")) + sorted(
            (PROJECT_ROOT / "scripts").glob("*.py")
        )
        for path in source_paths:
            with self.subTest(path=path.relative_to(PROJECT_ROOT).as_posix()):
                ast.parse(path.read_text(encoding="utf-8"), filename=str(path))

    def test_old_mainline_directories_are_removed(self) -> None:
        self.assertFalse((PROJECT_ROOT / "src").exists())
        self.assertFalse((PROJECT_ROOT / "code").exists())
        self.assertFalse((PROJECT_ROOT / "result").exists())

    def test_scripts_directory_only_contains_public_entrypoints(self) -> None:
        expected = {
            "aggregate_paper_metrics.py",
            "audit_experiment_gate_readiness.py",
            "audit_paper_claims.py",
            "audit_budget_report.py",
            "audit_dcms_frontier.py",
            "audit_dpo_execution_status.py",
            "audit_preference_acquisition.py",
            "audit_preference_experiment_preflight.py",
            "benchmark_pipeline.py",
            "build_experiment_run_matrix.py",
            "audit_matched_utility.py",
            "audit_mias_selection.py",
            "audit_preference_intervention.py",
            "audit_intervention_response.py",
            "audit_intervention_statistics.py",
            "audit_soft_group_error.py",
            "audit_preference_evaluation.py",
            "audit_preference_logprobs.py",
            "audit_preference_selector_scores.py",
            "build_embeddings.py",
            "build_dpo_execution_manifest.py",
            "build_dpo_run_record.py",
            "build_paper_artifacts.py",
            "build_preference_run_summary.py",
            "cli_common.py",
            "collect_dpo_run_records.py",
            "compare_run_metrics.py",
            "compute_crc.py",
            "convert_jsonl.py",
            "generate_preference_logprobs.py",
            "make_pcss_eval_split.py",
            "predict_local.py",
            "predict_vllm_openai.py",
            "prepare_preference_dcms_inputs.py",
            "prepare_preference_intervention_inputs.py",
            "prepare_preference_pool.py",
            "prepare_preference_splits.py",
            "score_preference_baselines.py",
            "prepare_multiclass_splits.py",
            "prepare_soft_group_intervals.py",
            "prepare.py",
            "prepare_prompt_clusters.py",
            "register_initial_policy_checkpoint.py",
            "reveal_preference_labels.py",
            "run_dpo_manifest_stage.py",
            "select_crc_error_mass.py",
            "select_dcms.py",
            "select_moment_matched_random.py",
            "select_preference_baseline.py",
            "select_preference_random.py",
            "select_pcss.py",
            "select_random.py",
            "selection_common.py",
            "train_preference_dpo_run.py",
            "train_round.py",
            "validate_dpo_run_pack.py",
            "validate_result_freeze_pack.py",
        }
        actual = {path.name for path in (PROJECT_ROOT / "scripts").glob("*.py")}

        self.assertEqual(expected, actual)

    def test_active_tree_does_not_import_legacy_code_package(self) -> None:
        active_paths = sorted((PROJECT_ROOT / "mias_dcms").rglob("*.py")) + sorted(
            (PROJECT_ROOT / "scripts").glob("*.py")
        )
        for path in active_paths:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            with self.subTest(path=path.relative_to(PROJECT_ROOT).as_posix()):
                for node in ast.walk(tree):
                    if isinstance(node, ast.Import):
                        imported = {alias.name for alias in node.names}
                        self.assertNotIn("code", imported)
                        self.assertNotIn("src", imported)
                    elif isinstance(node, ast.ImportFrom):
                        self.assertNotEqual("code", node.module)
                        self.assertNotEqual("src", node.module)

    def test_documented_script_paths_exist(self) -> None:
        documented_paths = [
            path
            for doc_path in [
                PROJECT_ROOT / "README.md",
                *sorted((PROJECT_ROOT / "docs").rglob("*.md")),
                *sorted((PROJECT_ROOT / "experiments").glob("*.md")),
                PROJECT_ROOT / "experiments" / "inputs" / "README.md",
                PROJECT_ROOT / "experiments" / "reports" / "README.md",
                PROJECT_ROOT / "resources" / "README.md",
            ]
            if doc_path.exists()
            for path in _script_paths_from_text(doc_path.read_text(encoding="utf-8"))
        ]

        missing = sorted(
            {
                path
                for path in documented_paths
                if not (PROJECT_ROOT / path).exists()
            }
        )

        self.assertEqual([], missing)


def _script_paths_from_text(text: str) -> set[str]:
    paths: set[str] = set()
    for raw_token in text.replace("`", " ").replace("'", " ").replace('"', " ").split():
        token = raw_token.strip(".,;:()[]{}<>")
        if token.startswith("scripts/") and token.endswith(".py"):
            paths.add(token)
    return paths


if __name__ == "__main__":
    unittest.main()
