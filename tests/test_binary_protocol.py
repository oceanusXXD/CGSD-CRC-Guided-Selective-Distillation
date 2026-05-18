from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.binary_protocol import (  # noqa: E402
    BINARY_SCORE_SOURCE,
    canonical_binary_answer,
    normalize_binary_label,
    normalize_binary_token,
)
from src.data import format_cgsd_chat_answer, format_cgsd_chat_prompt, format_generation_answer  # noqa: E402


def test_binary_protocol_normalizes_labels_and_tokens():
    assert normalize_binary_label("1", field_name="label") == 1
    assert normalize_binary_label("0", field_name="label") == 0
    assert normalize_binary_token("1.") == "one"
    assert normalize_binary_token("0") == "zero"


def test_generation_and_cgsd_answers_use_canonical_digits():
    assert canonical_binary_answer(1) == "1"
    assert canonical_binary_answer(0) == "0"
    assert format_generation_answer(1) == "1"
    assert format_generation_answer(0) == "0"
    assert format_cgsd_chat_answer(1) == "1<|im_end|>"
    assert format_cgsd_chat_answer(0) == "0<|im_end|>"


def test_cgsd_chat_prompt_requests_digit_protocol():
    prompt = format_cgsd_chat_prompt("q", "d")

    assert "Answer only \"1\" or \"0\"" in prompt
    assert BINARY_SCORE_SOURCE == "1_minus_0_logprob_margin"


if __name__ == "__main__":
    test_binary_protocol_normalizes_labels_and_tokens()
    test_generation_and_cgsd_answers_use_canonical_digits()
    test_cgsd_chat_prompt_requests_digit_protocol()
