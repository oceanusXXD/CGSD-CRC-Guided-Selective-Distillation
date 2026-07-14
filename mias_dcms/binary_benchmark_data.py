from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


@dataclass(frozen=True)
class BinaryBenchmarkSpec:
    name: str
    repo_id: str
    config: str
    revision: str
    label_names: tuple[str, str]
    expected_splits: tuple[str, ...]
    dataset_card_url: str


BINARY_BENCHMARK_SPECS: dict[str, BinaryBenchmarkSpec] = {
    "imdb": BinaryBenchmarkSpec(
        name="imdb",
        repo_id="stanfordnlp/imdb",
        config="plain_text",
        revision="e6281661ce1c48d982bc483cf8a173c1bbeb5d31",
        label_names=("negative", "positive"),
        expected_splits=("train", "test"),
        dataset_card_url="https://huggingface.co/datasets/stanfordnlp/imdb",
    ),
    "paws_labeled_final": BinaryBenchmarkSpec(
        name="paws_labeled_final",
        repo_id="google-research-datasets/paws",
        config="labeled_final",
        revision="161ece9501cf0a11f3e48bd356eaa82de46d6a09",
        label_names=("not_paraphrase", "paraphrase"),
        expected_splits=("train", "validation", "test"),
        dataset_card_url="https://huggingface.co/datasets/google-research-datasets/paws/viewer/labeled_final",
    ),
    "tweeteval_hate": BinaryBenchmarkSpec(
        name="tweeteval_hate",
        repo_id="cardiffnlp/tweet_eval",
        config="hate",
        revision="b3a375baf0f409c77e6bc7aa35102b7b3534f8be",
        label_names=("non-hate", "hate"),
        expected_splits=("train", "validation", "test"),
        dataset_card_url="https://huggingface.co/datasets/cardiffnlp/tweet_eval/viewer/hate",
    ),
}


class EmptyBinaryBenchmarkTextError(ValueError):
    pass


def normalize_binary_benchmark_row(
    dataset_name: str,
    row: Mapping[str, Any],
    *,
    split: str,
    index: int,
) -> dict[str, object]:
    """Convert a native binary benchmark record without changing its label."""
    spec = BINARY_BENCHMARK_SPECS[dataset_name]
    label = _binary_label(row.get("label"), dataset_name=dataset_name, index=index)

    if dataset_name == "imdb":
        source_id = str(index)
        query = "Classify the sentiment of this movie review as negative or positive."
        document = _required_text(row, "text", dataset_name=dataset_name, index=index)
    elif dataset_name == "paws_labeled_final":
        source_id = _required_text(row, "id", dataset_name=dataset_name, index=index)
        sentence_1 = _required_text(row, "sentence1", dataset_name=dataset_name, index=index)
        sentence_2 = _required_text(row, "sentence2", dataset_name=dataset_name, index=index)
        query = "Determine whether these two sentences are paraphrases.\n\nSentence A: " + sentence_1
        document = "Sentence B: " + sentence_2
    elif dataset_name == "tweeteval_hate":
        source_id = str(index)
        query = "Determine whether this post contains hate speech."
        document = _required_text(row, "text", dataset_name=dataset_name, index=index)
    else:  # pragma: no cover - protected by the lookup above
        raise ValueError(f"unsupported binary benchmark: {dataset_name!r}")

    return {
        "id": f"{spec.name}:{split}:{source_id}",
        "dataset": spec.name,
        "source_dataset": spec.repo_id,
        "source_config": spec.config,
        "source_revision": spec.revision,
        "source_split": str(split),
        "source_index": int(index),
        "query": query,
        "document": document,
        "groundtruth": label,
        "native_label": label,
        "label_name": spec.label_names[label],
    }


def validate_normalized_binary_rows(
    rows: list[Mapping[str, Any]],
    *,
    dataset_name: str,
    split: str,
) -> dict[str, int]:
    if not rows:
        raise ValueError(f"{dataset_name}/{split} contains no rows")
    ids = [str(row.get("id", "")) for row in rows]
    if not all(ids) or len(set(ids)) != len(ids):
        raise ValueError(f"{dataset_name}/{split} has missing or duplicate ids")

    counts = {"0": 0, "1": 0}
    for index, row in enumerate(rows):
        if str(row.get("dataset")) != dataset_name:
            raise ValueError(f"{dataset_name}/{split} row {index} has a mismatched dataset")
        if str(row.get("source_split")) != split:
            raise ValueError(f"{dataset_name}/{split} row {index} has a mismatched split")
        label = _binary_label(row.get("groundtruth"), dataset_name=dataset_name, index=index)
        if int(row.get("native_label", -1)) != label:
            raise ValueError(f"{dataset_name}/{split} row {index} changed its native label")
        if not str(row.get("query", "")).strip() or not str(row.get("document", "")).strip():
            raise ValueError(f"{dataset_name}/{split} row {index} has empty model input text")
        counts[str(label)] += 1
    return counts


def _binary_label(value: Any, *, dataset_name: str, index: int) -> int:
    try:
        label = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{dataset_name} row {index} has an invalid native label: {value!r}") from exc
    if label not in (0, 1):
        raise ValueError(f"{dataset_name} row {index} must have a native binary label, got {value!r}")
    return label


def _required_text(
    row: Mapping[str, Any],
    field: str,
    *,
    dataset_name: str,
    index: int,
) -> str:
    value = str(row.get(field, "")).strip()
    if not value:
        raise EmptyBinaryBenchmarkTextError(f"{dataset_name} row {index} has empty {field!r}")
    return value
