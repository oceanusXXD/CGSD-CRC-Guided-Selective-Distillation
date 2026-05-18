from argparse import Namespace
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.cgsd_cli_common import runtime_args_from_cli  # noqa: E402
from src.binary_protocol import BINARY_NEGATIVE_TEXT, BINARY_POSITIVE_TEXT  # noqa: E402
from src.data import format_cgsd_chat_answer, format_cgsd_chat_prompt  # noqa: E402


def test_cgsd_has_single_train_and_vllm_prediction_entrypoints():
    assert (PROJECT_ROOT / "scripts" / "cgsd_train_round.py").exists()
    assert (PROJECT_ROOT / "scripts" / "cgsd_predict_vllm_openai.py").exists()
    assert not (PROJECT_ROOT / "scripts" / "cgsd_predict.py").exists()
    assert not (PROJECT_ROOT / "scripts" / "train.py").exists()


def test_cgsd_runtime_defaults_are_fixed_for_binary_lora():
    args = runtime_args_from_cli(Namespace())

    assert args.max_length == 512
    assert args.batch_size == 4
    assert args.eval_batch_size == 32
    assert args.epochs == 3
    assert args.lr == 2e-4
    assert args.weight_decay == 0.01
    assert args.gradient_accumulation_steps == 4
    assert args.warmup_ratio == 0.1
    assert args.threshold == 0.0
    assert args.lora_r == 1
    assert args.lora_alpha == 16
    assert args.lora_dropout == 0.05
    assert args.lora_target_modules == "attention_mlp"
    assert args.lora_layer_scope == "all"
    assert args.balance_train_classes is False


def test_cgsd_training_and_inference_prompt_use_digit_protocol():
    prompt = format_cgsd_chat_prompt("query", "document")

    assert BINARY_POSITIVE_TEXT == "1"
    assert BINARY_NEGATIVE_TEXT == "0"
    assert 'Answer only "1" or "0"' in prompt
    assert format_cgsd_chat_answer(1) == "1<|im_end|>"
    assert format_cgsd_chat_answer(0) == "0<|im_end|>"


if __name__ == "__main__":
    test_cgsd_has_single_train_and_vllm_prediction_entrypoints()
    test_cgsd_runtime_defaults_are_fixed_for_binary_lora()
    test_cgsd_training_and_inference_prompt_use_digit_protocol()
