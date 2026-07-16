from __future__ import annotations

import unittest

from mias_dcms.experiment_run_matrix import build_experiment_run_matrix
from mias_dcms.preference_experiment_preflight import (
    PreferenceExperimentPreflightInputs,
    audit_preference_experiment_preflight,
)


class PreferenceExperimentPreflightTest(unittest.TestCase):
    def test_ready_when_fixed_pool_logprobs_splits_and_run_matrix_align(self) -> None:
        report = audit_preference_experiment_preflight(
            PreferenceExperimentPreflightInputs(
                active_pool=_active_pool(),
                oracle_store=_oracle_store(),
                logprob_rows=_logprob_rows(),
                split_manifest=_split_manifest(),
                run_matrix=_run_matrix(),
                expected_methods=["Random", "APL"],
                expected_seeds=[1],
            )
        )

        self.assertTrue(report.is_ready)
        self.assertEqual(2, report.active_pool_count)
        self.assertEqual(2, report.oracle_label_count)
        self.assertEqual(2, report.logprob_count)
        self.assertEqual(2, report.planned_run_count)
        self.assertEqual(["APL", "Random"], report.covered_methods)
        self.assertEqual([], report.issues)
        self.assertTrue(report.logprob_summary["implicit_margin_not_all_zero"])

    def test_reports_label_leakage_missing_oracle_and_missing_logprobs(self) -> None:
        active_pool = _active_pool()
        active_pool[0]["preference_label"] = "A"
        oracle_store = {"p1": _oracle_store()["p1"]}
        logprob_rows = [_logprob_rows()[0]]

        report = audit_preference_experiment_preflight(
            PreferenceExperimentPreflightInputs(
                active_pool=active_pool,
                oracle_store=oracle_store,
                logprob_rows=logprob_rows,
                split_manifest=_split_manifest(),
                run_matrix=_run_matrix(),
                expected_methods=["Random", "APL"],
                expected_seeds=[1],
            )
        )

        self.assertFalse(report.is_ready)
        codes = {issue["code"] for issue in report.issues}
        self.assertIn("hidden_label_leakage", codes)
        self.assertIn("oracle_missing_active_id", codes)
        self.assertIn("logprob_missing_active_id", codes)

    def test_reports_split_overlap_and_run_matrix_input_path_mismatch(self) -> None:
        split_manifest = _split_manifest()
        split_manifest["seed_ids"] = ["p1"]
        split_manifest["active_pool_ids"] = ["p1", "p2"]
        run_matrix = _run_matrix()
        run_matrix[0]["data_config"]["active_pool_path"] = "wrong/active_pool.jsonl"

        report = audit_preference_experiment_preflight(
            PreferenceExperimentPreflightInputs(
                active_pool=_active_pool(),
                oracle_store=_oracle_store(),
                logprob_rows=_logprob_rows(),
                split_manifest=split_manifest,
                run_matrix=run_matrix,
                expected_active_pool_path="experiments/inputs/preference/active_pool.jsonl",
                expected_oracle_store_path="experiments/inputs/preference/oracle_store.jsonl",
                expected_logprobs_path="experiments/inputs/preference/logprobs.jsonl",
                expected_methods=["Random", "APL"],
                expected_seeds=[1],
            )
        )

        self.assertFalse(report.is_ready)
        codes = {issue["code"] for issue in report.issues}
        self.assertIn("split_overlap", codes)
        self.assertIn("run_matrix_active_pool_path_mismatch", codes)

    def test_mias_preflight_requires_label_safe_exact_feature_artifacts(self) -> None:
        report = audit_preference_experiment_preflight(
            PreferenceExperimentPreflightInputs(
                active_pool=_active_pool(),
                oracle_store=_oracle_store(),
                logprob_rows=_logprob_rows(),
                split_manifest=_split_manifest(),
                run_matrix=_run_matrix(),
                expected_methods=["MIAS"],
                mias_seed_rows=[
                    {"sample_id": "s1", "preferred_response": 1},
                    {"sample_id": "s2", "preferred_response": 2},
                ],
                mias_seed_features=[
                    {"sample_id": "s1", "response_a_embedding": [1.0]},
                ],
                mias_pool_features=[
                    {"sample_id": "p1", "response_a_embedding": [1.0]},
                    {
                        "sample_id": "p2",
                        "response_a_embedding": [2.0],
                        "oracle_label": "A",
                    },
                ],
            )
        )

        codes = {issue["code"] for issue in report.issues}
        self.assertIn("mias_seed_features_invalid", codes)
        self.assertIn("mias_pool_features_invalid", codes)
        self.assertIn("mias_dpo_seed_insufficient", codes)

    def test_mias_preflight_rejects_a_seed_without_both_preference_directions(self) -> None:
        report = audit_preference_experiment_preflight(
            PreferenceExperimentPreflightInputs(
                active_pool=_active_pool(),
                oracle_store=_oracle_store(),
                logprob_rows=_logprob_rows(),
                split_manifest=_split_manifest(),
                run_matrix=_run_matrix(),
                expected_methods=["MIAS"],
                mias_seed_rows=[
                    {"sample_id": f"s{index}", "preferred_response": 1}
                    for index in range(20)
                ],
                mias_seed_features=[
                    {"sample_id": f"s{index}", "response_a_embedding": [float(index)]}
                    for index in range(20)
                ],
                mias_pool_features=[
                    {"sample_id": "p1", "response_a_embedding": [1.0]},
                    {"sample_id": "p2", "response_a_embedding": [2.0]},
                ],
            )
        )

        self.assertIn("mias_dpo_seed_missing_direction", {issue["code"] for issue in report.issues})


def _active_pool() -> list[dict[str, object]]:
    return [
        {
            "sample_id": "p1",
            "prompt": "Prompt one",
            "response_a": "Good answer",
            "response_b": "Bad answer",
            "length_gap": 0.0,
        },
        {
            "sample_id": "p2",
            "prompt": "Prompt two",
            "response_a": "Short",
            "response_b": "Longer answer",
            "length_gap": -0.33,
        },
    ]


def _oracle_store() -> dict[str, dict[str, object]]:
    return {
        "p1": {"sample_id": "p1", "preference_label": "A"},
        "p2": {"sample_id": "p2", "preference_label": "B"},
    }


def _logprob_rows() -> list[dict[str, object]]:
    return [
        {
            "sample_id": "p1",
            "policy_logprob_response_1": -1.0,
            "policy_logprob_response_2": -2.0,
            "reference_logprob_response_1": -1.5,
            "reference_logprob_response_2": -1.6,
        },
        {
            "sample_id": "p2",
            "policy_logprob_response_1": -2.0,
            "policy_logprob_response_2": -1.0,
            "reference_logprob_response_1": -1.4,
            "reference_logprob_response_2": -1.8,
        },
    ]


def _split_manifest() -> dict[str, object]:
    return {
        "seed_ids": ["s1"],
        "active_pool_ids": ["p1", "p2"],
        "heldout_ids": ["h1"],
        "test_ids": ["t1"],
    }


def _run_matrix() -> list[dict[str, object]]:
    return build_experiment_run_matrix(
        datasets=["helpsteer2"],
        models=["qwen-0.6b"],
        budgets=[100],
        seeds=[1],
        methods=["Random", "APL"],
        artifact_root="experiments/runs/dpo_main",
        training_config={
            "initialization": "shared_seed_policy_v1",
            "optimizer": "adamw",
            "learning_rate": 5e-6,
            "batch_size": 8,
            "update_steps": 120,
            "train_token_budget": 240000,
            "data_accumulation": "cumulative",
            "prompt_format": "chatml_pairwise_v1",
            "generation_parameters": {"temperature": 0.0, "max_new_tokens": 256},
        },
        judge_config={
            "judge_version": "fixed-human-labels-primary",
            "judge_prompt_hash": "prompt-sha256",
            "evaluator": "held_out_preference_labels",
        },
        data_config={
            "active_pool_path": "experiments/inputs/preference/active_pool.jsonl",
            "oracle_store_path": "experiments/inputs/preference/oracle_store.jsonl",
            "logprobs_path": "experiments/inputs/preference/logprobs.jsonl",
            "split_manifest_path": "experiments/inputs/preference/split_manifest.json",
        },
    )


if __name__ == "__main__":
    unittest.main()
