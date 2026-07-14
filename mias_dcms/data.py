from __future__ import annotations
import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, TypeVar
from .binary_protocol import BINARY_SYSTEM_PROMPT, binary_user_prompt, canonical_binary_answer, normalize_binary_label
from .utils import _require_torch

try:
    from torch.utils.data import Dataset
except ModuleNotFoundError:  # pragma: no cover - exercised by lightweight runtime consumers
    class Dataset:  # type: ignore[no-redef]
        pass

T = TypeVar('T')
EMPTY_THINKING_BLOCK = '<think>\n\n</think>\n\n'

@dataclass(frozen=True)
class PairExample:
    sample_id: str
    query: str
    document: str
    label: int
    sample_weight: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict, compare=False)

def format_query_document(query: str, document: str) -> str:
    return f'{BINARY_SYSTEM_PROMPT}\n{binary_user_prompt(query, document)}'

def format_binary_chat_prompt(query: str, document: str) -> str:
    return f'<|im_start|>system\n{BINARY_SYSTEM_PROMPT}<|im_end|>\n<|im_start|>user\n{binary_user_prompt(query, document)}<|im_end|>\n<|im_start|>assistant\n{EMPTY_THINKING_BLOCK}'

def format_generation_answer(label: int) -> str:
    return canonical_binary_answer(label)

def format_binary_chat_answer(label: int) -> str:
    return canonical_binary_answer(label) + '<|im_end|>'

def discover_jsonl_files(path: str | Path) -> list[Path]:
    source = Path(path)
    if source.is_file():
        return [source]
    if not source.exists():
        raise FileNotFoundError(f'Data path does not exist: {source}')
    files = sorted(source.glob('*.jsonl'))
    if not files:
        raise FileNotFoundError(f'No .jsonl files found under: {source}')
    return files

def _parse_label(row: dict[str, Any], label_field: str) -> int:
    if label_field in row:
        value = row[label_field]
    elif 'label' in row:
        value = row['label']
    elif 'groundtruth' in row:
        value = row['groundtruth']
    elif 'parsed_answer' in row:
        return normalize_binary_label(row['parsed_answer'], field_name='parsed_answer')
    else:
        raise KeyError(f'Missing label field: {label_field}')
    return normalize_binary_label(value, field_name=label_field)

def load_examples(data_path: str | Path, query_field: str='query', document_field: str='document', label_field: str='groundtruth') -> list[PairExample]:
    examples: list[PairExample] = []
    seen_ids: dict[str, str] = {}
    for file_path in discover_jsonl_files(data_path):
        with file_path.open('r', encoding='utf-8') as handle:
            for line_number, line in enumerate(handle, start=1):
                stripped = line.strip()
                if not stripped:
                    continue
                row = json.loads(stripped)
                try:
                    query = str(row[query_field])
                    document = str(row[document_field])
                    label = _parse_label(row, label_field)
                except KeyError as exc:
                    raise KeyError(f'{file_path}:{line_number} missing field {exc}') from exc
                sample_id = str(row.get('id', row.get('sample_id', f'{file_path.stem}:{line_number}')))
                source_location = f'{file_path}:{line_number}'
                if sample_id in seen_ids:
                    raise ValueError(f'Duplicate sample id {sample_id!r} at {source_location}; first seen at {seen_ids[sample_id]}. Sample ids must be globally unique.')
                seen_ids[sample_id] = source_location
                sample_weight = float(row.get('sample_weight', row.get('weight', 1.0)) or 1.0)
                examples.append(PairExample(sample_id=sample_id, query=query, document=document, label=label, sample_weight=sample_weight, metadata=dict(row)))
    if not examples:
        raise ValueError(f'No examples loaded from: {data_path}')
    return examples

def split_examples(examples: list[PairExample], val_ratio: float=0.1, seed: int=42, stratified: bool=True) -> tuple[list[PairExample], list[PairExample]]:
    if val_ratio <= 0:
        return (list(examples), [])
    if val_ratio >= 1:
        raise ValueError('val_ratio must be smaller than 1.0')
    rng = random.Random(seed)
    if not stratified:
        shuffled = list(examples)
        rng.shuffle(shuffled)
        val_size = max(1, int(round(len(shuffled) * val_ratio)))
        return (shuffled[val_size:], shuffled[:val_size])
    train: list[PairExample] = []
    val: list[PairExample] = []
    by_label: dict[int, list[PairExample]] = {}
    for example in examples:
        by_label.setdefault(example.label, []).append(example)
    for label_examples in by_label.values():
        label_examples = list(label_examples)
        rng.shuffle(label_examples)
        val_size = max(1, int(round(len(label_examples) * val_ratio)))
        val.extend(label_examples[:val_size])
        train.extend(label_examples[val_size:])
    rng.shuffle(train)
    rng.shuffle(val)
    return (train, val)

def split_examples_three_way(examples: list[PairExample], train_ratio: float=0.1, val_ratio: float=0.1, seed: int=42, stratified: bool=True, group_duplicates: bool=True) -> tuple[list[PairExample], list[PairExample], list[PairExample]]:
    if train_ratio <= 0:
        raise ValueError('train_ratio must be greater than 0')
    if val_ratio < 0:
        raise ValueError('val_ratio must be non-negative')
    if train_ratio + val_ratio >= 1:
        raise ValueError('train_ratio + val_ratio must be smaller than 1.0')
    rng = random.Random(seed)

    def split_items_by_row_count(items: list[T], row_count: Callable[[T], int]) -> tuple[list[T], list[T], list[T]]:
        shuffled = list(items)
        rng.shuffle(shuffled)
        total_rows = sum((row_count(item) for item in shuffled))
        train_size = max(1, int(round(total_rows * train_ratio)))
        val_size = int(round(total_rows * val_ratio))
        if val_ratio > 0 and total_rows - train_size > 0:
            val_size = max(1, val_size)
        if train_size + val_size >= total_rows:
            val_size = max(0, total_rows - train_size - 1)
        train_part: list[T] = []
        val_part: list[T] = []
        test_part: list[T] = []
        train_count = 0
        val_count = 0
        for item in shuffled:
            size = row_count(item)
            if train_count < train_size:
                train_part.append(item)
                train_count += size
            elif val_count < val_size:
                val_part.append(item)
                val_count += size
            else:
                test_part.append(item)
        return (train_part, val_part, test_part)

    def split_group(group: list[PairExample]) -> tuple[list[PairExample], list[PairExample], list[PairExample]]:
        return split_items_by_row_count(group, lambda _: 1)

    def group_by_pair(rows: list[PairExample]) -> list[list[PairExample]]:
        groups: dict[tuple[str, str], list[PairExample]] = {}
        for example in rows:
            groups.setdefault((example.query, example.document), []).append(example)
        for group in groups.values():
            labels = {example.label for example in group}
            if len(labels) > 1:
                raise ValueError('Duplicate query-document pair has conflicting labels.')
        return list(groups.values())

    def flatten(grouped_rows: list[list[PairExample]]) -> list[PairExample]:
        return [example for group in grouped_rows for example in group]
    if not stratified:
        if group_duplicates:
            train_groups, val_groups, test_groups = split_items_by_row_count(group_by_pair(examples), len)
            return (flatten(train_groups), flatten(val_groups), flatten(test_groups))
        return split_group(examples)
    train: list[PairExample] = []
    val: list[PairExample] = []
    test: list[PairExample] = []
    by_label: dict[int, list[PairExample]] = {}
    if group_duplicates:
        for group in group_by_pair(examples):
            by_label.setdefault(group[0].label, []).extend(group)
    else:
        for example in examples:
            by_label.setdefault(example.label, []).append(example)
    for label_examples in by_label.values():
        if group_duplicates:
            train_groups, val_groups, test_groups = split_items_by_row_count(group_by_pair(label_examples), len)
            train_part, val_part, test_part = (flatten(train_groups), flatten(val_groups), flatten(test_groups))
        else:
            train_part, val_part, test_part = split_group(label_examples)
        train.extend(train_part)
        val.extend(val_part)
        test.extend(test_part)
    rng.shuffle(train)
    rng.shuffle(val)
    rng.shuffle(test)
    return (train, val, test)

def filter_examples_by_ids(examples: list[PairExample], sample_ids: set[str]) -> list[PairExample]:
    return [example for example in examples if example.sample_id in sample_ids]

def examples_to_rows(examples: list[PairExample]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for example in examples:
        row = dict(example.metadata)
        row.update({'id': example.sample_id, 'query': example.query, 'document': example.document, 'label': int(example.label), 'groundtruth': int(example.label), 'sample_weight': float(example.sample_weight)})
        rows.append(row)
    return rows

def examples_from_rows(rows: list[dict[str, Any]]) -> list[PairExample]:
    examples: list[PairExample] = []
    for row in rows:
        label = normalize_binary_label(row.get('label', row.get('groundtruth')), field_name='training row label')
        examples.append(PairExample(sample_id=str(row['id']), query=str(row.get('query', '')), document=str(row.get('document', '')), label=label, sample_weight=float(row.get('sample_weight', row.get('weight', 1.0)) or 1.0), metadata=dict(row)))
    return examples

class GenerationQueryDocumentDataset(Dataset):

    def __init__(self, examples: list[PairExample], tokenizer: Any, max_length: int=2048, cache_tokenization: bool=True, input_format: str='query_document_marked_v1') -> None:
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.cache_tokenization = cache_tokenization
        self.input_format = input_format
        self._encoded_cache: list[dict[str, Any]] | None = None
        if cache_tokenization:
            self._encoded_cache = [self._encode_example(example) for example in examples]

    def __len__(self) -> int:
        return len(self.examples)

    def sequence_lengths(self) -> list[int]:
        if self._encoded_cache is not None:
            return [len(item['input_ids']) for item in self._encoded_cache]
        return [len(self[index]['input_ids']) for index in range(len(self))]

    def _encode_prompt(self, example: PairExample) -> list[int]:
        if self.input_format == 'query_document_marked_v1':
            encoded = self.tokenizer(format_query_document(example.query, example.document), truncation=True, max_length=max(self.max_length - 1, 1), padding=False)
        elif self.input_format == 'chat_binary':
            encoded = self.tokenizer(format_binary_chat_prompt(example.query, example.document), truncation=True, max_length=max(self.max_length - 1, 1), padding=False, add_special_tokens=False)
        elif self.input_format == 'tokenizer_pair_v1':
            encoded = self.tokenizer(example.query, example.document, truncation=True, max_length=max(self.max_length - 1, 1), padding=False)
        else:
            raise ValueError(f'Unsupported input_format: {self.input_format}')
        return list(encoded['input_ids'])

    def _encode_example(self, example: PairExample) -> dict[str, Any]:
        prompt_ids = self._encode_prompt(example)
        answer_text = format_binary_chat_answer(example.label) if self.input_format == 'chat_binary' else format_generation_answer(example.label)
        answer_ids = self.tokenizer(answer_text, add_special_tokens=False, padding=False)['input_ids']
        if not answer_ids:
            raise ValueError('Generation target must encode to at least one token.')
        max_prompt_length = max(self.max_length - len(answer_ids), 1)
        if len(prompt_ids) > max_prompt_length:
            prompt_ids = prompt_ids[-max_prompt_length:]
        input_ids = prompt_ids + list(answer_ids)
        labels = [-100] * len(prompt_ids) + list(answer_ids)
        return {'input_ids': input_ids, 'attention_mask': [1] * len(input_ids), 'labels': labels, 'prompt_input_ids': prompt_ids, 'prompt_attention_mask': [1] * len(prompt_ids), 'target_label': int(example.label), 'sample_weight': float(example.sample_weight), 'sample_id': example.sample_id}

    def __getitem__(self, index: int) -> dict[str, Any]:
        if self._encoded_cache is not None:
            return dict(self._encoded_cache[index])
        example = self.examples[index]
        return self._encode_example(example)

class GenerationPairCollator:

    def __init__(self, tokenizer: Any, pad_to_multiple_of: int | None=8) -> None:
        self.tokenizer = tokenizer
        self.pad_to_multiple_of = pad_to_multiple_of

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        torch = _require_torch()
        sample_ids = [str(feature['sample_id']) for feature in features]
        target_labels = torch.tensor([feature['target_label'] for feature in features], dtype=torch.float)
        sample_weights = torch.tensor([float(feature.get('sample_weight', 1.0)) for feature in features], dtype=torch.float)
        prompt_features = [{'input_ids': feature['prompt_input_ids'], 'attention_mask': feature['prompt_attention_mask']} for feature in features]
        prompt_batch = self.tokenizer.pad(prompt_features, padding=True, pad_to_multiple_of=self.pad_to_multiple_of, return_tensors='pt')
        full_features = [{'input_ids': feature['input_ids'], 'attention_mask': feature['attention_mask']} for feature in features]
        batch = self.tokenizer.pad(full_features, padding=True, pad_to_multiple_of=self.pad_to_multiple_of, return_tensors='pt')
        labels = torch.full_like(batch['input_ids'], fill_value=-100)
        for row_index, feature in enumerate(features):
            label_values = torch.tensor(feature['labels'], dtype=torch.long)
            if getattr(self.tokenizer, 'padding_side', 'right') == 'left':
                start = labels.size(1) - label_values.numel()
                labels[row_index, start:] = label_values
            else:
                labels[row_index, :label_values.numel()] = label_values
        batch['labels'] = labels
        batch['prompt_input_ids'] = prompt_batch['input_ids']
        batch['prompt_attention_mask'] = prompt_batch['attention_mask']
        batch['target_labels'] = target_labels
        batch['sample_weights'] = sample_weights
        batch['sample_ids'] = sample_ids
        return batch

class TokenBudgetBatchSampler:

    def __init__(self, lengths: list[int], max_batch_size: int, max_tokens_per_batch: int, shuffle: bool=True, seed: int=42) -> None:
        if max_batch_size < 1:
            raise ValueError('max_batch_size must be at least 1')
        if max_tokens_per_batch < 1:
            raise ValueError('max_tokens_per_batch must be at least 1')
        self.lengths = [max(1, int(length)) for length in lengths]
        self.max_batch_size = max_batch_size
        self.max_tokens_per_batch = max_tokens_per_batch
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0

    def _build_batches(self) -> list[list[int]]:
        indices = sorted(range(len(self.lengths)), key=lambda index: self.lengths[index])
        batches: list[list[int]] = []
        batch: list[int] = []
        batch_max_length = 0
        for index in indices:
            length = self.lengths[index]
            next_max_length = max(batch_max_length, length)
            next_size = len(batch) + 1
            exceeds_size = next_size > self.max_batch_size
            exceeds_tokens = next_max_length * next_size > self.max_tokens_per_batch
            if batch and (exceeds_size or exceeds_tokens):
                batches.append(batch)
                batch = []
                batch_max_length = 0
            batch.append(index)
            batch_max_length = max(batch_max_length, length)
        if batch:
            batches.append(batch)
        return batches

    def __iter__(self):
        batches = self._build_batches()
        if self.shuffle:
            rng = random.Random(self.seed + self.epoch)
            rng.shuffle(batches)
            self.epoch += 1
        return iter(batches)

    def __len__(self) -> int:
        return len(self._build_batches())
