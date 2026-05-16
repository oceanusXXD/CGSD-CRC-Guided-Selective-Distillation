from __future__ import annotations

import contextlib
import importlib.util
import inspect
import io
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def load_python_script(path: Path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load script: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

from scripts import evaluate, train
from scripts import cgsd_make_baseline_rows
from scripts import cgsd_convert_fever
from scripts.cgsd_cli_common import apply_teacher_label, load_teacher_labels
from scripts.cgsd_cli_common import (
    stage_cache_decision,
    summarize_teacher_label_usage,
)
from scripts.run_cgsd import load_embeddings
from algorithms.cgsd import (
    apply_crc_decisions,
    build_deployment_rows,
    calibrate_crc,
    select_dbds_samples,
    split_calibration_pool_ids,
)
from src.data import (
    GenerationPairCollator,
    GenerationQueryDocumentDataset,
    PairExample,
    format_cgsd_chat_prompt,
    format_query_document,
    load_examples,
)
from src.metrics import compute_binary_metrics
from src.model import (
    ALL_MODES,
    LAST_LAYER_LORA_PATTERN,
    LORA_MODES,
    LORA_TARGET_GROUPS,
    LORA_TARGETS,
    QwenGenerativeModel,
    get_last_layer_index_from_config,
    resolve_lora_layers_to_transform,
    sparse_causal_lm_loss,
)
from src.trainer import evaluate_model


class LayerModeContractTests(unittest.TestCase):
    def test_cgsd_baseline_rows_random_excludes_calibration_and_writes_training_fields(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            split_path = root / "split.json"
            pool_predictions_path = root / "pool_student_predictions.jsonl"
            output_path = root / "cgsd_train_rows.jsonl"
            summary_path = root / "baseline_selection_summary.json"

            split_path.write_text(
                json.dumps({"calibration_ids": ["cal"], "pool_ids": ["a", "b", "c"]}),
                encoding="utf-8",
            )
            rows = [
                {"id": "cal", "query": "q", "document": "cal doc", "groundtruth": 0, "label": 0},
                {"id": "a", "query": "q", "document": "doc a", "groundtruth": 1, "teacher_label": 1, "teacher_confidence": 0.8, "teacher_source": "teacher_api_file"},
                {"id": "b", "query": "q", "document": "doc b", "groundtruth": 0, "teacher_label": 0, "teacher_confidence": 0.5, "teacher_source": "teacher_api_file"},
                {"id": "c", "query": "q", "document": "doc c", "groundtruth": 1, "teacher_label": 1, "teacher_confidence": 1.0, "teacher_source": "groundtruth_substitute_for_real_teacher_api"},
            ]
            pool_predictions_path.write_text(
                "".join(json.dumps(row) + "\n" for row in rows),
                encoding="utf-8",
            )

            selected = cgsd_make_baseline_rows.make_baseline_rows(
                strategy="random",
                budget=2,
                split_ids_path=split_path,
                pool_student_predictions_path=pool_predictions_path,
                pool_crc_predictions_path=None,
                embeddings_path=None,
                output_path=output_path,
                summary_path=summary_path,
                seed=3,
                teacher_beta=2.0,
            )

            self.assertEqual(len(selected), 2)
            self.assertNotIn("cal", {row["id"] for row in selected})
            for row in selected:
                self.assertEqual(row["label"], row["teacher_label"])
                self.assertEqual(row["selection_round"], 0)
                self.assertEqual(row["selection_role"], "baseline_random")
                self.assertIn("sample_weight", row)
            written = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(written, selected)
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            self.assertEqual(summary["strategy"], "random")
            self.assertEqual(summary["selected_rows"], 2)

    def test_cgsd_baseline_rows_uncertainty_and_defer_random_use_expected_sources(self) -> None:
        rows = [
            {"id": "a", "query": "q", "document": "doc a", "groundtruth": 1, "prediction": 1, "routing_score": 0.4, "defer": True},
            {"id": "b", "query": "q", "document": "doc b", "groundtruth": 0, "prediction": 0, "routing_score": 0.1, "defer": False},
            {"id": "c", "query": "q", "document": "doc c", "groundtruth": 1, "prediction": 1, "routing_score": 0.3, "defer": True},
        ]

        uncertainty_ids = [
            row["id"]
            for row in cgsd_make_baseline_rows.select_candidate_rows(
                strategy="uncertainty",
                candidate_rows=rows,
                budget=2,
                seed=1,
                embeddings_by_id=None,
            )
        ]
        defer_ids = {
            row["id"]
            for row in cgsd_make_baseline_rows.select_candidate_rows(
                strategy="defer-random",
                candidate_rows=rows,
                budget=2,
                seed=1,
                embeddings_by_id=None,
            )
        }

        self.assertEqual(uncertainty_ids, ["b", "c"])
        self.assertEqual(defer_ids, {"a", "c"})

    def test_load_embeddings_supports_npy_with_evidence_rows_sidecar(self) -> None:
        import numpy as np

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            embedding_path = root / "evidence_pair_embeddings.npy"
            rows_path = root / "evidence_rows.jsonl"
            np.save(embedding_path, np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32))
            rows_path.write_text(
                json.dumps({"id": "row-a"}) + "\n" + json.dumps({"sample_id": "row-b"}) + "\n",
                encoding="utf-8",
            )

            embeddings = load_embeddings(embedding_path)

            self.assertEqual(set(embeddings), {"row-a", "row-b"})
            self.assertEqual(embeddings["row-a"].tolist(), [1.0, 0.0])
            self.assertEqual(embeddings["row-b"].tolist(), [0.0, 1.0])

    def test_load_examples_uses_sample_id_when_id_is_absent(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            data_path = Path(tmp) / "pairs.jsonl"
            data_path.write_text(
                json.dumps(
                    {
                        "sample_id": "select100_row000:select100",
                        "query": "q",
                        "document": "d",
                        "label": "yes",
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            examples = load_examples(data_path, label_field="label")

            self.assertEqual(examples[0].sample_id, "select100_row000:select100")
            self.assertEqual(examples[0].label, 1)

    def test_convert_fever_rows_use_fixed_query_and_claim_evidence_document_by_default(self) -> None:
        rows = cgsd_convert_fever.convert_fever_documents(
            [
                {
                    "id": "fever_evidence_7",
                    "original_text": "Claim:\nA claim.\n\nEvidence:\nEvidence text.",
                    "label": 1,
                }
            ]
        )

        self.assertEqual(
            rows,
            [
                {
                    "id": "fever_evidence_7",
                    "query": "Does the evidence support the claim?",
                    "document": "Claim:\nA claim.\n\nEvidence:\nEvidence text.",
                    "groundtruth": 1,
                    "document_id": "fever_evidence_7",
                }
            ],
        )

    def test_cgsd_build_embeddings_uses_last_token_pooling(self) -> None:
        from scripts import cgsd_build_embeddings

        hidden_states = torch.tensor(
            [
                [[1.0, 0.0], [2.0, 0.0], [99.0, 99.0]],
                [[3.0, 0.0], [4.0, 0.0], [5.0, 0.0]],
            ]
        )
        attention_mask = torch.tensor(
            [
                [1, 1, 0],
                [1, 1, 1],
            ]
        )

        pooled = cgsd_build_embeddings.last_token_pool(hidden_states, attention_mask)

        self.assertEqual(pooled.tolist(), [[2.0, 0.0], [5.0, 0.0]])

    def test_prefix_tuning_script_contract(self) -> None:
        from scripts import train_prefix_tuning

        with patch.object(sys, "argv", ["train_prefix_tuning.py"]):
            args = train_prefix_tuning.parse_args()

        self.assertEqual(args.model_path, "model/qwen3-0.6b")
        self.assertEqual(args.output_dir, "outputs/prefix_tuning")
        self.assertEqual(args.prefix_num_virtual_tokens, 16)
        self.assertFalse(args.prefix_projection)
        self.assertEqual(args.max_length, 2048)
        self.assertEqual(args.threshold, 0.0)
        self.assertTrue(args.balance_train_classes)
        self.assertEqual(train_prefix_tuning.PREFIX_ADAPTER_DIRNAME, "prefix_adapter")

        examples = [
            PairExample(sample_id="n1", query="q", document="d", label=0),
            PairExample(sample_id="p1", query="q", document="d", label=1),
            PairExample(sample_id="p2", query="q", document="d", label=1),
        ]
        balanced = train_prefix_tuning.balance_train_examples(examples, seed=7)
        self.assertEqual(train_prefix_tuning.count_labels(balanced), {0: 2, 1: 2})

    def test_only_last_layer_lora_mode_is_registered(self) -> None:
        self.assertEqual(set(ALL_MODES), {"lora_attention_mlp"})
        self.assertEqual(LORA_MODES, {"lora_attention_mlp"})

        for removed_mode in ("frozen_linear", "frozen_mlp", "lora_attention", "lora_mlp", "lora_all_linear"):
            with self.subTest(mode=removed_mode):
                self.assertNotIn(removed_mode, ALL_MODES)
                self.assertNotIn(removed_mode, LORA_MODES)
                with patch.object(sys, "argv", ["train.py", "--mode", removed_mode]), contextlib.redirect_stderr(io.StringIO()):
                    with self.assertRaises(SystemExit):
                        train.parse_args()

    def test_lora_targets_last_transformer_layer_attention_and_mlp(self) -> None:
        self.assertEqual(
            LORA_TARGETS["lora_attention_mlp"],
            ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        )
        self.assertEqual(
            LORA_TARGET_GROUPS["attention"],
            ["q_proj", "k_proj", "v_proj", "o_proj"],
        )
        self.assertEqual(
            LORA_TARGET_GROUPS["qv"],
            ["q_proj", "v_proj"],
        )
        self.assertEqual(
            LORA_TARGET_GROUPS["mlp"],
            ["gate_proj", "up_proj", "down_proj"],
        )
        self.assertEqual(LAST_LAYER_LORA_PATTERN, "layers")

        class Config:
            num_hidden_layers = 24

        self.assertEqual(get_last_layer_index_from_config(Config()), 23)
        self.assertEqual(resolve_lora_layers_to_transform(Config(), "last1"), 23)
        self.assertEqual(resolve_lora_layers_to_transform(Config(), "last4"), [20, 21, 22, 23])
        self.assertIsNone(resolve_lora_layers_to_transform(Config(), "all"))

    def test_lora_modes_keep_2048_default_length_and_l4_batch_defaults(self) -> None:
        for mode in ("lora_attention_mlp",):
            with self.subTest(mode=mode):
                with patch.object(sys, "argv", ["train.py", "--mode", mode]):
                    args = train.parse_args()
                self.assertEqual(args.max_length, 2048)
                self.assertEqual(args.threshold, 0.0)
                self.assertEqual(args.lora_target_modules, "attention_mlp")
                self.assertEqual(args.lora_layer_scope, "last1")
                self.assertIsNone(args.split_ids_path)

                train.apply_mode_speed_defaults(args)
                self.assertEqual(args.batch_size, 8)
                self.assertEqual(args.eval_batch_size, 32)
                self.assertEqual(args.gradient_accumulation_steps, 2)

        with patch.object(sys, "argv", ["evaluate.py", "--checkpoint_dir", "checkpoint"]):
            eval_args = evaluate.parse_args()
        self.assertEqual(eval_args.max_length, 2048)
        self.assertEqual(eval_args.threshold, 0.0)

    def test_lora_class_balance_uses_class_weights_not_oversampling(self) -> None:
        examples = [
            PairExample(sample_id="n1", query="q", document="d", label=0),
            PairExample(sample_id="p1", query="q", document="d", label=1),
            PairExample(sample_id="p2", query="q", document="d", label=1),
        ]

        self.assertFalse(hasattr(train, "balance_train_examples"))
        self.assertEqual(train.compute_balanced_class_weights(examples), {0: 1.5, 1: 0.75})

    def test_precomputed_split_ids_are_reused_by_id(self) -> None:
        examples = [
            PairExample(sample_id="row-1", query="q", document="a", label=0),
            PairExample(sample_id="row-2", query="q", document="b", label=1),
            PairExample(sample_id="row-3", query="q", document="c", label=0),
        ]
        split_payload = {
            "train_ids": ["row-2"],
            "val_ids": [],
            "test_ids": ["row-1", "row-3"],
        }

        train_examples, eval_examples, test_examples = train.apply_precomputed_split_ids(
            examples,
            split_payload,
        )

        self.assertEqual([example.sample_id for example in train_examples], ["row-2"])
        self.assertEqual(eval_examples, [])
        self.assertEqual([example.sample_id for example in test_examples], ["row-1", "row-3"])

    def test_prompt_is_binary_generation_prompt_without_chat_template(self) -> None:
        text = format_query_document(
            "Is this document contains a link to social media?",
            "Profile includes https://twitter.com/example.",
        )

        self.assertEqual(
            text,
            "Query:\n"
            "Is this document contains a link to social media?\n\n"
            "Document:\n"
            "Profile includes https://twitter.com/example.\n\n"
            "Output exactly one character: 1 or 0. Yes is 1, no is 0.",
        )
        self.assertNotIn("<|im_start|>", text)
        self.assertNotIn("relevant to the query", text)
        self.assertIn("self.backbone.generate", inspect.getsource(QwenGenerativeModel.generate))

    def test_generation_dataset_masks_prompt_and_keeps_prompt_inputs(self) -> None:
        class TokenizerStub:
            padding_side = "right"

            def __call__(self, *texts: str, **_: object) -> dict[str, list[int]]:
                text = "\n".join(texts)
                return {"input_ids": [ord(character) for character in text]}

            def pad(
                self,
                features: list[dict[str, list[int]]],
                padding: bool,
                pad_to_multiple_of: int | None,
                return_tensors: str,
            ) -> dict[str, torch.Tensor]:
                del padding, return_tensors
                max_length = max(len(feature["input_ids"]) for feature in features)
                if pad_to_multiple_of:
                    max_length = ((max_length + pad_to_multiple_of - 1) // pad_to_multiple_of) * pad_to_multiple_of
                input_ids = []
                attention_mask = []
                for feature in features:
                    pad_length = max_length - len(feature["input_ids"])
                    input_ids.append(feature["input_ids"] + [0] * pad_length)
                    attention_mask.append(feature["attention_mask"] + [0] * pad_length)
                return {
                    "input_ids": torch.tensor(input_ids, dtype=torch.long),
                    "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
                }

        dataset = GenerationQueryDocumentDataset(
            [PairExample(sample_id="s1", query="q", document="d", label=1)],
            tokenizer=TokenizerStub(),
            max_length=128,
        )
        item = dataset[0]
        self.assertEqual(item["input_ids"], item["prompt_input_ids"] + [ord("1")])
        self.assertEqual(item["labels"], [-100] * len(item["prompt_input_ids"]) + [ord("1")])

        batch = GenerationPairCollator(TokenizerStub(), pad_to_multiple_of=None)([item])
        self.assertIn("prompt_input_ids", batch)
        self.assertIn("labels", batch)
        self.assertEqual(batch["target_labels"].tolist(), [1.0])
        self.assertEqual(batch["sample_weights"].tolist(), [1.0])

    def test_cgsd_dataset_uses_chat_prompt_and_yes_no_answer_tokens(self) -> None:
        class TokenizerStub:
            padding_side = "right"

            def __call__(self, text: str, **_: object) -> dict[str, list[int]]:
                return {"input_ids": [ord(character) for character in text]}

        dataset = GenerationQueryDocumentDataset(
            [PairExample(sample_id="s1", query="q", document="d", label=1)],
            tokenizer=TokenizerStub(),
            max_length=256,
            input_format="cgsd_chat_yes_no_v1",
        )
        item = dataset[0]
        prompt = format_cgsd_chat_prompt("q", "d")
        answer = "yes<|im_end|>"

        self.assertEqual(item["prompt_input_ids"], [ord(character) for character in prompt])
        self.assertEqual(item["input_ids"], [ord(character) for character in prompt + answer])
        self.assertEqual(
            item["labels"],
            [-100] * len(prompt) + [ord(character) for character in answer],
        )

    def test_sparse_causal_lm_loss_matches_full_shifted_cross_entropy(self) -> None:
        torch.manual_seed(7)
        logits = torch.randn(2, 5, 13, dtype=torch.bfloat16)
        labels = torch.tensor(
            [
                [-100, -100, -100, 3, 4],
                [-100, 2, -100, -100, 8],
            ],
            dtype=torch.long,
        )

        shifted_logits = logits[:, :-1, :].float()
        shifted_labels = labels[:, 1:]
        expected = torch.nn.functional.cross_entropy(
            shifted_logits.reshape(-1, shifted_logits.size(-1)),
            shifted_labels.reshape(-1),
            ignore_index=-100,
        )

        actual = sparse_causal_lm_loss(logits, labels)

        self.assertEqual(actual.dtype, torch.float32)
        self.assertTrue(torch.allclose(actual, expected, atol=1e-6))

    def test_sparse_causal_lm_loss_applies_binary_class_token_weights(self) -> None:
        torch.manual_seed(11)
        logits = torch.randn(2, 3, 5)
        labels = torch.tensor(
            [
                [-100, -100, 0],
                [-100, -100, 1],
            ],
            dtype=torch.long,
        )
        shifted_logits = logits[:, :-1, :].float()
        shifted_labels = labels[:, 1:]
        active_positions = shifted_labels.ne(-100)
        per_token_loss = torch.nn.functional.cross_entropy(
            shifted_logits[active_positions],
            shifted_labels[active_positions],
            reduction="none",
        )
        expected = (per_token_loss * torch.tensor([2.0, 0.5])).mean()

        actual = sparse_causal_lm_loss(logits, labels, class_token_weights={0: 2.0, 1: 0.5})

        self.assertTrue(torch.allclose(actual, expected, atol=1e-6))

    def test_evaluate_scores_binary_token_logits_without_generate(self) -> None:
        class TokenizerStub:
            pad_token_id = 0
            eos_token_id = 2

            def __call__(self, text: str, **_: object) -> dict[str, list[int]]:
                return {"input_ids": {"0": [10], "1": [11], "no": [10], "yes": [11]}[text]}

            def batch_decode(self, *_: object, **__: object) -> list[str]:
                return ["not-used"]

        class Output:
            def __init__(self, logits: torch.Tensor, loss: torch.Tensor | None = None) -> None:
                self.logits = logits
                self.loss = loss

        class ModelStub(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.generate_called = False

            def forward(
                self,
                input_ids: torch.Tensor,
                attention_mask: torch.Tensor,
                labels: torch.Tensor | None = None,
            ) -> Output:
                logits = torch.zeros(
                    input_ids.size(0),
                    input_ids.size(1),
                    12,
                    dtype=torch.float,
                    device=input_ids.device,
                )
                if labels is None:
                    logits[0, -1, 10] = -2.0
                    logits[0, -1, 11] = 2.0
                    logits[1, -1, 10] = 2.0
                    logits[1, -1, 11] = -2.0
                    logits[2, -1, 10] = 0.0
                    logits[2, -1, 11] = 0.2
                    return Output(logits=logits)
                return Output(logits=logits, loss=torch.tensor(0.25, device=input_ids.device))

            def generate(self, *_: object, **__: object) -> torch.Tensor:
                self.generate_called = True
                raise AssertionError("evaluation should score binary answer-token logits without generate")

        batch = {
            "input_ids": torch.tensor([[5, 6, 10], [7, 8, 11], [1, 2, 10]], dtype=torch.long),
            "attention_mask": torch.tensor([[1, 1, 1], [1, 1, 1], [1, 1, 1]], dtype=torch.long),
            "labels": torch.tensor([[-100, -100, 10], [-100, -100, 11], [-100, -100, 10]], dtype=torch.long),
            "prompt_input_ids": torch.tensor([[0, 5, 6], [7, 8, 9], [1, 2, 3]], dtype=torch.long),
            "prompt_attention_mask": torch.tensor([[0, 1, 1], [1, 1, 1], [1, 1, 1]], dtype=torch.long),
            "target_labels": torch.tensor([1.0, 0.0, 0.0]),
            "sample_ids": ["positive", "negative", "small-margin"],
        }
        model = ModelStub()

        with tempfile.TemporaryDirectory() as tmpdir:
            predictions_path = Path(tmpdir) / "predictions.jsonl"
            metrics = evaluate_model(
                model=model,
                dataloader=[batch],
                device=torch.device("cpu"),
                tokenizer=TokenizerStub(),
                predictions_path=predictions_path,
                threshold=0.5,
                negative_token_text="no",
                positive_token_text="yes",
            )
            rows = [
                json.loads(line)
                for line in predictions_path.read_text(encoding="utf-8").splitlines()
            ]

        self.assertFalse(model.generate_called)
        self.assertAlmostEqual(metrics["accuracy"], 1.0)
        self.assertAlmostEqual(metrics["loss"], 0.25)
        self.assertEqual([row["prediction"] for row in rows], [1, 0, 0])
        self.assertAlmostEqual(rows[0]["score"], 4.0)
        self.assertAlmostEqual(rows[1]["score"], -4.0)
        self.assertAlmostEqual(rows[2]["score"], 0.2)
        self.assertGreater(rows[0]["probability"], 0.98)
        self.assertLess(rows[1]["probability"], 0.02)
        self.assertEqual(rows[0]["generated_text"], "1")
        self.assertEqual(rows[1]["generated_text"], "0")
        self.assertNotIn("generated_token_text", rows[0])

    def test_binary_metrics_include_report_fields(self) -> None:
        metrics = compute_binary_metrics(
            labels=[1, 1, 1, 1, 0, 0],
            scores=[0.9, 0.8, 0.7, -0.4, 0.6, -0.1],
        )

        self.assertAlmostEqual(metrics["accuracy"], 4 / 6)
        self.assertAlmostEqual(metrics["balanced_accuracy"], 0.625)
        self.assertAlmostEqual(metrics["precision"], 0.75)
        self.assertAlmostEqual(metrics["recall"], 0.75)
        self.assertAlmostEqual(metrics["f1"], 0.75)
        self.assertAlmostEqual(metrics["positive_F1"], 0.75)
        self.assertAlmostEqual(metrics["macro_F1"], 0.625)


class CGSDAlgorithmContractTests(unittest.TestCase):
    def test_crc_calibration_uses_wrong_accept_bound(self) -> None:
        rows = [
            {"id": "a", "label": "yes", "prediction": 1, "score": 4.0},
            {"id": "b", "label": 0, "prediction": "no", "score": -4.0},
            {"id": "c", "label": "no", "prediction": "yes", "score": 0.2},
        ]

        result = calibrate_crc(rows, alpha=0.26, temperature=1.0, lambda_grid=[0.0, 0.8, 0.99])
        decisions = apply_crc_decisions(rows, lambda_hat=result.lambda_hat, temperature=1.0)

        self.assertEqual(result.lambda_hat, 0.8)
        self.assertEqual(result.wrong_accept_count, 0)
        self.assertLessEqual(result.risk_bound, 0.26)
        self.assertEqual([row["prediction"] for row in decisions], [1, 0, 1])
        self.assertEqual([row["crc_decision"] for row in decisions], ["accept", "accept", "defer"])

    def test_cgsd_split_keeps_calibration_disjoint_from_pool(self) -> None:
        rows = [
            {"id": f"n{i}", "label": 0}
            for i in range(8)
        ] + [
            {"id": f"p{i}", "label": 1}
            for i in range(2)
        ]

        calibration_ids, pool_ids = split_calibration_pool_ids(rows, n_calibration=4, seed=3)

        self.assertEqual(len(calibration_ids), 4)
        self.assertEqual(set(calibration_ids) & set(pool_ids), set())
        self.assertEqual(len(calibration_ids) + len(pool_ids), len(rows))

    def test_dbds_selects_defer_bands_and_easy_anchors(self) -> None:
        rows = [
            {"id": "b1", "label": 1, "prediction": 1, "routing_score": 0.79},
            {"id": "b2", "label": 0, "prediction": 0, "routing_score": 0.72},
            {"id": "m1", "label": 1, "prediction": 1, "routing_score": 0.65},
            {"id": "f1", "label": 0, "prediction": 1, "routing_score": 0.40},
            {"id": "f2", "label": 1, "prediction": 0, "routing_score": 0.30},
            {"id": "a1", "label": 1, "prediction": 1, "routing_score": 0.95},
        ]
        embeddings = {
            row["id"]: torch.tensor([float(index), 1.0]).numpy()
            for index, row in enumerate(rows, start=1)
        }

        selection = select_dbds_samples(
            rows,
            defer_ids=["b1", "b2", "m1", "f1", "f2"],
            already_selected_ids=[],
            budget=3,
            lambda_hat=0.8,
            embeddings_by_id=embeddings,
            easy_anchor_ratio=1 / 3,
            seed=9,
        )

        self.assertEqual(selection.selected_budget, 3)
        self.assertEqual(selection.anchor_ids, ["a1"])
        self.assertEqual(sum(selection.band_counts.values()), 3)
        self.assertEqual(set(selection.distillation_ids) & set(selection.anchor_ids), set())

    def test_dbds_can_reuse_cached_anchor_candidates(self) -> None:
        rows = [
            {"id": "d1", "label": 1, "prediction": 1, "routing_score": 0.71},
            {"id": "a1", "label": 1, "prediction": 1, "routing_score": 0.99},
            {"id": "a2", "label": 0, "prediction": 0, "routing_score": 0.98},
        ]
        embeddings = {row["id"]: torch.tensor([float(index), 1.0]).numpy() for index, row in enumerate(rows)}

        selection = select_dbds_samples(
            rows,
            defer_ids=["d1"],
            already_selected_ids=[],
            budget=1,
            lambda_hat=0.8,
            embeddings_by_id=embeddings,
            anchor_count=1,
            anchor_candidate_ids=["a2"],
            seed=9,
        )

        self.assertEqual(selection.anchor_ids, ["a2"])
        self.assertEqual(selection.anchor_candidate_source, "cached_anchor_ids")

    def test_select_cli_parses_band_ratios_for_ablation(self) -> None:
        from scripts.cgsd_select import parse_band_ratios

        self.assertEqual(parse_band_ratios("0.6,0.3,0.1"), {"B": 0.6, "M": 0.3, "F": 0.1})
        self.assertEqual(parse_band_ratios("B=1,M=0,F=0"), {"B": 1.0, "M": 0.0, "F": 0.0})
        with self.assertRaisesRegex(ValueError, "sum to 1"):
            parse_band_ratios("0.6,0.6,0.0")

    def test_teacher_labels_accept_api_output_or_groundtruth_substitute(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            teacher_path = Path(tmp_dir) / "teacher.jsonl"
            teacher_path.write_text(
                "\n".join(
                    [
                        json.dumps({"id": "x1", "teacher_logit_margin": -2.0}),
                        json.dumps({"id": "x2", "teacher_label": "yes", "teacher_confidence": 0.7}),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            labels = load_teacher_labels(teacher_path, teacher_temperature=1.0)
            api_row = apply_teacher_label({"id": "x1", "label": 1, "groundtruth": 1}, labels)
            fallback_row = apply_teacher_label({"id": "x3", "label": "no", "groundtruth": "no"}, labels)

        self.assertEqual(api_row["label"], 0)
        self.assertEqual(api_row["teacher_label_source"], "teacher_logit_margin_sign")
        self.assertGreater(api_row["teacher_confidence"], 0.5)
        self.assertEqual(fallback_row["label"], 0)
        self.assertEqual(fallback_row["teacher_confidence"], 1.0)
        self.assertEqual(fallback_row["teacher_source"], "groundtruth_substitute_for_real_teacher_api")

    def test_deployment_rows_reuse_train_labels_before_crc_defer(self) -> None:
        rows = [
            {"id": "trained", "label": 0, "prediction": 1, "score": 0.1},
            {"id": "accepted", "label": 1, "prediction": 1, "score": 4.0},
            {"id": "deferred", "label": 0, "prediction": 1, "score": 0.1},
        ]

        deployment = build_deployment_rows(
            rows,
            train_label_by_id={"trained": "no"},
            lambda_hat=0.8,
            temperature=1.0,
        )
        by_id = {row["id"]: row for row in deployment}

        self.assertEqual(by_id["trained"]["deployment_source"], "teacher_train_label")
        self.assertEqual(by_id["trained"]["output_label"], 0)
        self.assertEqual(by_id["accepted"]["deployment_source"], "student_accept")
        self.assertEqual(by_id["deferred"]["deployment_source"], "teacher_defer")

    def test_stage_cache_policy_reuses_complete_outputs_and_rejects_partials(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_a = Path(tmp_dir) / "a.json"
            output_b = Path(tmp_dir) / "b.jsonl"
            output_a.write_text("{}", encoding="utf-8")
            output_b.write_text("", encoding="utf-8")

            cached = stage_cache_decision(
                stage_name="example",
                required_outputs=[output_a, output_b],
                cache_policy="reuse",
            )

            self.assertTrue(cached.cache_hit)
            self.assertEqual(cached.action, "reuse")

            output_b.unlink()
            with self.assertRaisesRegex(RuntimeError, "partial cache"):
                stage_cache_decision(
                    stage_name="example",
                    required_outputs=[output_a, output_b],
                    cache_policy="reuse",
                )

            overwrite = stage_cache_decision(
                stage_name="example",
                required_outputs=[output_a, output_b],
                cache_policy="overwrite",
            )
            self.assertFalse(overwrite.cache_hit)
            self.assertEqual(overwrite.action, "run")

    def test_teacher_usage_counts_groundtruth_and_api_sources(self) -> None:
        rows = [
            {
                "id": "g1",
                "query": "q",
                "document": "short doc",
                "teacher_source": "groundtruth_substitute_for_real_teacher_api",
            },
            {
                "id": "api1",
                "query": "q",
                "document": "longer teacher doc",
                "teacher_source": "teacher_api_file",
            },
        ]

        usage = summarize_teacher_label_usage(rows, purpose="calibration")

        self.assertEqual(usage["purpose"], "calibration")
        self.assertEqual(usage["teacher_calls"], 2)
        self.assertEqual(usage["groundtruth_substitute_calls"], 1)
        self.assertEqual(usage["teacher_api_file_calls"], 1)
        self.assertGreater(usage["estimated_prompt_tokens"], 0)

    def test_run_cgsd_exposes_helpers_without_monolithic_pipeline_entry(self) -> None:
        from scripts import run_cgsd

        self.assertTrue(callable(run_cgsd.predict_examples))
        self.assertTrue(callable(run_cgsd.train_round_model))
        self.assertFalse(hasattr(run_cgsd, "parse_args"))
        self.assertFalse(hasattr(run_cgsd, "main"))

    def test_cgsd_stage_clis_do_not_depend_on_config_or_state_files(self) -> None:
        script_names = [
            "cgsd_prepare.py",
            "cgsd_predict.py",
            "cgsd_calibrate.py",
            "cgsd_select.py",
            "cgsd_train_round.py",
            "cgsd_finalize.py",
        ]
        forbidden = ("config_path", "cgsd_config", "cgsd_state", "load_state", "save_state")
        for script_name in script_names:
            source = (PROJECT_ROOT / "scripts" / script_name).read_text(encoding="utf-8")
            for token in forbidden:
                self.assertNotIn(token, source, msg=f"{script_name} should be CLI-driven, found {token}")

    def test_experiment_wrappers_keep_default_inputs_and_outputs_under_experiments(self) -> None:
        wrapper_names = [
            "cgsd_round0_eval.sh",
            "cgsd_round0_select.sh",
            "cgsd_select_round.sh",
            "cgsd_train_round.sh",
            "cgsd_eval_round.sh",
            "cgsd_finalize.sh",
            "cgsd_baseline_rows.sh",
            "cgsd_run_exp1_default_3rounds.sh",
        ]
        env_source = (PROJECT_ROOT / "experiments" / "bin" / "cgsd_env.sh").read_text(encoding="utf-8")
        self.assertIn('INPUT_DIR="${INPUT_DIR:-$EXPERIMENTS_DIR/inputs/$DATASET}"', env_source)
        self.assertIn('RUN_ROOT="${RUN_ROOT:-$EXPERIMENTS_DIR/runs/$DATASET}"', env_source)
        self.assertNotIn("outputs/", env_source)
        for script_name in wrapper_names:
            script_path = PROJECT_ROOT / "experiments" / "bin" / script_name
            self.assertTrue(script_path.exists(), msg=f"missing experiment wrapper {script_name}")
            source = script_path.read_text(encoding="utf-8")
            self.assertIn("cgsd_env.sh", source)
            self.assertNotIn("outputs/", source)

    def test_experiment_result_collector_reads_round_and_usage_summaries(self) -> None:
        collector = load_python_script(PROJECT_ROOT / "experiments" / "bin" / "cgsd_collect_results.py")
        with tempfile.TemporaryDirectory() as tmp_dir:
            run_dir = Path(tmp_dir) / "runs" / "lrobench" / "exp1_seed1"
            round_dir = run_dir / "round_1"
            round_dir.mkdir(parents=True)
            (round_dir / "round_summary.json").write_text(
                json.dumps(
                    {
                        "round_index": 1,
                        "temperature": 15,
                        "lambda_hat": 0.72,
                        "crc": {"alpha": 0.07, "empirical_risk": 0.01, "risk_bound": 0.02, "grid_feasible": True},
                        "pool_summary": {
                            "total": 100,
                            "defer_rate": 0.2,
                            "defer_count": 20,
                            "wrong_accept_count": 3,
                            "accept_error_rate": 0.0375,
                        },
                        "pool_metrics": {"accuracy": 0.83},
                    }
                ),
                encoding="utf-8",
            )
            (round_dir / "predict_usage.json").write_text(
                json.dumps({"student_model_calls": 300, "estimated_student_prompt_tokens": 1200}),
                encoding="utf-8",
            )
            (run_dir / "cgsd_summary.json").write_text(
                json.dumps({"best_round_index": 1, "teacher_train_calls": 50, "teacher_defer_calls": 20}),
                encoding="utf-8",
            )

            record = collector.collect_run(run_dir)

        self.assertEqual(record["dataset"], "lrobench")
        self.assertEqual(record["run_name"], "exp1_seed1")
        self.assertEqual(record["round_index"], 1)
        self.assertAlmostEqual(record["defer_rate"], 0.2)
        self.assertAlmostEqual(record["wrong_accept_rate"], 0.03)
        self.assertEqual(record["student_model_calls_total"], 300)
        self.assertEqual(record["teacher_defer_calls"], 20)

    def test_lrobench_splitter_writes_per_query_data_and_embeddings(self) -> None:
        import numpy as np

        splitter = load_python_script(PROJECT_ROOT / "experiments" / "bin" / "cgsd_split_lrobench_inputs.py")
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            data_path = root / "data.jsonl"
            embeddings_path = root / "embeddings.npy"
            data_path.write_text(
                "\n".join(
                    [
                        json.dumps({"id": "row0:q1", "query": "q one", "document": "d0", "groundtruth": 1}),
                        json.dumps({"id": "row1:q2", "query": "q two", "document": "d1", "groundtruth": 0}),
                        json.dumps({"id": "row2:q1", "query": "q one", "document": "d2", "groundtruth": 1}),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            np.save(embeddings_path, np.asarray([[1.0, 0.0], [0.0, 1.0], [2.0, 0.0]], dtype=np.float32))
            embeddings_path.with_suffix(".ids.jsonl").write_text(
                "\n".join(json.dumps({"id": sample_id}) for sample_id in ["row0:q1", "row1:q2", "row2:q1"]) + "\n",
                encoding="utf-8",
            )
            output_root = root / "inputs"

            summary = splitter.split_lrobench_inputs(
                data_path=data_path,
                embeddings_path=embeddings_path,
                output_root=output_root,
                prefix="lrobench",
            )

            q1_data = output_root / "lrobench_q1" / "data.jsonl"
            q1_emb = np.load(output_root / "lrobench_q1" / "embeddings.npy")
            q1_ids = [
                json.loads(line)["id"]
                for line in (output_root / "lrobench_q1" / "embeddings.ids.jsonl").read_text(encoding="utf-8").splitlines()
            ]
            q1_line_count = len(q1_data.read_text(encoding="utf-8").splitlines())

            self.assertEqual(summary["groups"], 2)
            self.assertEqual(q1_line_count, 2)
            self.assertEqual(q1_emb.tolist(), [[1.0, 0.0], [2.0, 0.0]])
            self.assertEqual(q1_ids, ["row0:q1", "row2:q1"])

    def test_show_result_does_not_create_default_stage_directories(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            usage_path = root / "predict_usage.json"
            output_dir = root / "new_output"
            usage_path.write_text(json.dumps({"ok": True}), encoding="utf-8")

            completed = subprocess.run(
                [
                    sys.executable,
                    str(PROJECT_ROOT / "scripts" / "cgsd_predict.py"),
                    "--output_dir",
                    str(output_dir),
                    "--round_index",
                    "0",
                    "--usage_path",
                    str(usage_path),
                    "--show_result",
                ],
                cwd=PROJECT_ROOT,
                text=True,
                capture_output=True,
                check=True,
            )

            self.assertIn('"ok": true', completed.stdout)
            self.assertFalse(output_dir.exists())

    def test_later_round_calibration_requires_fixed_temperature(self) -> None:
        from scripts.cgsd_calibrate import fixed_temperature_for_round

        self.assertEqual(
            fixed_temperature_for_round(
                round_index=2,
                explicit_temperature=10.0,
                previous_round_summary_path=None,
            ),
            (10.0, "cli_arg"),
        )
        with tempfile.TemporaryDirectory() as tmp_dir:
            summary_path = Path(tmp_dir) / "round_summary.json"
            summary_path.write_text(json.dumps({"temperature": 15.0}), encoding="utf-8")

            self.assertEqual(
                fixed_temperature_for_round(
                    round_index=2,
                    explicit_temperature=None,
                    previous_round_summary_path=summary_path,
                ),
                (15.0, "previous_round_summary"),
            )

        with self.assertRaisesRegex(RuntimeError, "fixed temperature"):
            fixed_temperature_for_round(
                round_index=2,
                explicit_temperature=None,
                previous_round_summary_path=None,
            )


if __name__ == "__main__":
    unittest.main()
