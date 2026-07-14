from __future__ import annotations

import unittest

from mias_dcms.preference_run_evaluation import (
    FIXED_HUMAN_PAIRWISE_EVALUATION_SOURCE,
    build_preference_run_evaluation_artifacts,
)


class PreferenceRunEvaluationTest(unittest.TestCase):
    def test_materializes_heldout_pairwise_metrics_without_claiming_generation_judge(self) -> None:
        artifacts = build_preference_run_evaluation_artifacts(
            [
                {"sample_id": "one", "length_gap_bin": "balanced"},
                {"sample_id": "two", "length_gap_bin": "a_longer"},
                {"sample_id": "tie", "length_gap_bin": "b_longer"},
            ],
            oracle_store={
                "one": {"preference_label": "A"},
                "two": {"preference_label": "B"},
                "tie": {"preference_label": "tie"},
            },
            logprob_rows=[
                _logprobs("one", policy=(3.0, 1.0), reference=(0.0, 1.0)),
                _logprobs("two", policy=(0.0, 2.0), reference=(2.0, 0.0)),
                _logprobs("tie", policy=(0.0, 1.0), reference=(0.0, 1.0)),
            ],
            seed_budget=8,
            active_budget=4,
        )

        self.assertEqual(2, len(artifacts.preference_rows))
        self.assertEqual(1.0, artifacts.metrics["preference_accuracy"])
        self.assertEqual(1.0, artifacts.metrics["length_controlled_win_rate"])
        self.assertEqual(2, artifacts.metrics["preference_eval_count"])
        self.assertEqual(-2.5, artifacts.metrics["capability_regression"])
        self.assertEqual(2, artifacts.metrics["capability_eval_count"])
        self.assertEqual(0.0, artifacts.initial_metrics["preference_accuracy"])
        self.assertEqual(2, len(artifacts.aulc_rows))
        self.assertFalse(artifacts.metadata["generation_judge_available"])
        self.assertTrue(artifacts.metadata["capability_evaluation_available"])
        self.assertEqual(2, len(artifacts.capability_rows))
        self.assertEqual(FIXED_HUMAN_PAIRWISE_EVALUATION_SOURCE, artifacts.judge_rows[0]["judge_source"])

    def test_rejects_incomplete_logprob_coverage(self) -> None:
        with self.assertRaisesRegex(ValueError, "exactly cover"):
            build_preference_run_evaluation_artifacts(
                [{"sample_id": "one", "length_gap_bin": "balanced"}],
                oracle_store={"one": {"preference_label": "A"}},
                logprob_rows=[_logprobs("other", policy=(1.0, 0.0), reference=(1.0, 0.0))],
                seed_budget=8,
                active_budget=4,
            )


def _logprobs(
    sample_id: str,
    *,
    policy: tuple[float, float],
    reference: tuple[float, float],
) -> dict[str, float | str]:
    return {
        "sample_id": sample_id,
        "policy_logprob_response_1": policy[0],
        "policy_logprob_response_2": policy[1],
        "reference_logprob_response_1": reference[0],
        "reference_logprob_response_2": reference[1],
    }


if __name__ == "__main__":
    unittest.main()
