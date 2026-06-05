"""Query-document binary classification toolkit.

Responsibilities are split by module: `data` owns the JSONL dataloader and
prompt construction, `crc` owns CRC/defer/selection logic, `model` and
`trainer` own LoRA training and local scoring, and `embeddings` owns embedding
artifact loading and coverage checks.
"""
