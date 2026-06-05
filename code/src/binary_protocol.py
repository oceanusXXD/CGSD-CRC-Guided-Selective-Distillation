"""Single source of truth for the binary classification protocol.

This module defines the shared convention used by training, inference, CRC, and
evaluation. The positive answer token is `1`, the negative answer token is `0`,
and the prediction score is `logit/logprob(1) - logit/logprob(0)`. Other modules
should not redefine label aliases or prompt text, because that can make the
training and inference protocols drift apart.
"""

from __future__ import annotations

from typing import Any

BINARY_POSITIVE_TEXT = "1"
BINARY_NEGATIVE_TEXT = "0"
BINARY_SCORE_SOURCE = "1_minus_0_logprob_margin"
BINARY_SYSTEM_PROMPT = 'You are a precise binary classifier. Answer only "1" or "0"./no_think'
BINARY_USER_PROMPT_TEMPLATE = (
    "Query: {query}\n"
    "Document: {document}\n"
    'Return "1" if the document satisfies the query; otherwise return "0".'
)


def normalize_binary_label(value: Any, *, field_name: str) -> int:
    """Normalize accepted binary labels to integer 1/0."""
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized == BINARY_POSITIVE_TEXT:
            return 1
        if normalized == BINARY_NEGATIVE_TEXT:
            return 0
    try:
        label = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be binary 0/1, got {value!r}") from exc
    if label not in {0, 1}:
        raise ValueError(f"{field_name} must be binary 0/1, got {value!r}")
    return label


def normalize_binary_token(token: Any) -> str:
    """Return `one`, `zero`, or empty string for a generated token."""
    raw = str(token or "")
    stripped = raw.strip().lower()
    if stripped == BINARY_POSITIVE_TEXT:
        return "one"
    if stripped == BINARY_NEGATIVE_TEXT:
        return "zero"
    edge = "".join(ch for ch in stripped if ch.isalpha() or ch.isdigit())
    if edge == BINARY_POSITIVE_TEXT:
        return "one"
    if edge == BINARY_NEGATIVE_TEXT:
        return "zero"
    return ""


def canonical_binary_answer(label: int) -> str:
    """Return the canonical supervised answer token for a binary label."""
    normalized = normalize_binary_label(label, field_name="label")
    return BINARY_POSITIVE_TEXT if normalized == 1 else BINARY_NEGATIVE_TEXT


def binary_user_prompt(query: str, document: str) -> str:
    """Return the shared query-document binary prompt body."""
    return BINARY_USER_PROMPT_TEMPLATE.format(query=query, document=document)
