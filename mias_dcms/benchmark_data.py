from __future__ import annotations

from collections.abc import Iterable
import random
from typing import Any


AG_NEWS_LABELS = ("World", "Sports", "Business", "Sci/Tech")
TREC_LABELS = (
    "ABBREVIATION",
    "ENTITY",
    "DESCRIPTION",
    "HUMAN_BEING",
    "LOCATION",
    "NUMERIC_VALUE",
)
TREC_COARSE_LABEL_CODES = ("ABBR", "ENTY", "DESC", "HUM", "LOC", "NUM")
DBPEDIA_14_LABELS = (
    "Company",
    "EducationalInstitution",
    "Artist",
    "Athlete",
    "OfficeHolder",
    "MeanOfTransportation",
    "Building",
    "NaturalPlace",
    "Village",
    "Animal",
    "Plant",
    "Album",
    "Film",
    "WrittenWork",
)
HELPSTEER_ATTRIBUTES = ("helpfulness", "correctness", "coherence", "complexity", "verbosity")


def normalize_ag_news_row(row: dict[str, Any], *, split: str, index: int) -> dict[str, Any]:
    label = int(row["label"])
    return {
        "id": f"ag_news:{split}:{index}",
        "dataset": "ag_news",
        "split": split,
        "text": str(row["text"]).strip(),
        "label": label,
        "label_name": AG_NEWS_LABELS[label],
    }


def normalize_dbpedia_row(row: dict[str, Any], *, split: str, index: int) -> dict[str, Any]:
    label = int(row["label"])
    title = str(row["title"]).strip()
    content = str(row["content"]).strip()
    text = "\n\n".join(part for part in (title, content) if part)
    return {
        "id": f"dbpedia_14:{split}:{index}",
        "dataset": "dbpedia_14",
        "split": split,
        "title": title,
        "content": content,
        "text": text,
        "label": label,
        "label_name": DBPEDIA_14_LABELS[label],
    }


def normalize_trec_row(row: dict[str, Any], *, split: str, index: int) -> dict[str, Any]:
    raw_label = row.get("coarse_label", row.get("label"))
    if isinstance(raw_label, str):
        normalized_label = raw_label.strip().upper()
        if normalized_label in TREC_COARSE_LABEL_CODES:
            label = TREC_COARSE_LABEL_CODES.index(normalized_label)
        elif normalized_label in TREC_LABELS:
            label = TREC_LABELS.index(normalized_label)
        else:
            raise ValueError(f"unsupported TREC coarse label: {raw_label!r}")
    else:
        label = int(raw_label)
    return {
        "id": f"trec:{split}:{index}",
        "dataset": "trec",
        "split": split,
        "text": str(row["text"]).strip(),
        "label": label,
        "label_name": TREC_LABELS[label],
    }


def reservoir_sample_per_class(
    rows: Iterable[dict[str, Any]],
    *,
    per_class: int,
    seed: int,
    label_field: str = "label",
) -> list[dict[str, Any]]:
    if per_class <= 0:
        raise ValueError("per_class must be positive")
    reservoirs: dict[int, list[dict[str, Any]]] = {}
    seen: dict[int, int] = {}
    randomizers: dict[int, random.Random] = {}
    for row in rows:
        label = int(row[label_field])
        seen[label] = seen.get(label, 0) + 1
        reservoir = reservoirs.setdefault(label, [])
        randomizer = randomizers.setdefault(label, random.Random(f"{seed}:{label}"))
        copied = dict(row)
        if len(reservoir) < per_class:
            reservoir.append(copied)
            continue
        replacement_index = randomizer.randrange(seen[label])
        if replacement_index < per_class:
            reservoir[replacement_index] = copied
    sampled: list[dict[str, Any]] = []
    for label in sorted(reservoirs):
        sampled.extend(sorted(reservoirs[label], key=lambda row: str(row.get("id", ""))))
    return sampled


def build_helpsteer_attribute_index(
    rows: Iterable[dict[str, Any]],
) -> dict[tuple[str, str], dict[str, float]]:
    index: dict[tuple[str, str], dict[str, float]] = {}
    for row in rows:
        key = (str(row["prompt"]), str(row["response"]))
        index[key] = {attribute: float(row[attribute]) for attribute in HELPSTEER_ATTRIBUTES}
    return index


def normalize_helpsteer_preference_row(
    row: dict[str, Any],
    attribute_index: dict[tuple[str, str], dict[str, float]],
    *,
    index: int,
) -> dict[str, Any]:
    prompt = str(row["prompt"])
    response_1 = str(row["response_1"])
    response_2 = str(row["response_2"])
    strength = int(row.get("preference_strength", 0))
    preferred_response = 2 if strength > 0 else 1 if strength < 0 else 0
    split = str(row.get("split", "train"))
    return {
        "id": f"helpsteer2_preference:{split}:{index}",
        "dataset": "helpsteer2_preference",
        "split": split,
        "prompt": prompt,
        "response_1": response_1,
        "response_2": response_2,
        "preferred_response": preferred_response,
        "preference_strength": strength,
        "preference_magnitude": abs(strength),
        "preference_statement": str(row.get("preference_statement", "")),
        "preference_elaboration": str(row.get("preference_elaboration", "")),
        "response_1_word_count": _word_count(response_1),
        "response_2_word_count": _word_count(response_2),
        "response_1_char_count": len(response_1),
        "response_2_char_count": len(response_2),
        "response_1_attributes": attribute_index.get((prompt, response_1)),
        "response_2_attributes": attribute_index.get((prompt, response_2)),
        "source_a": row.get("source_a", row.get("response_1_source")),
        "source_b": row.get("source_b", row.get("response_2_source")),
    }


def _word_count(text: str) -> int:
    return len(text.split())
