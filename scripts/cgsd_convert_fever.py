#!/usr/bin/env python
"""把 FEVER documents.json 转成 CGSD JSONL。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.cgsd_cli_common import binary_to_int
from src.utils import resolve_input_path, resolve_output_path, write_jsonl


DEFAULT_FEVER_QUERY = "Does the evidence support the claim?"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--documents_path", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--fixed_query", default=DEFAULT_FEVER_QUERY)
    return parser.parse_args()


def split_claim_evidence(original_text: str) -> tuple[str, str]:
    text = str(original_text)
    claim_marker = "Claim:\n"
    evidence_marker = "\n\nEvidence:\n"
    if claim_marker not in text or evidence_marker not in text:
        raise ValueError("FEVER original_text must contain 'Claim:\\n' and '\\n\\nEvidence:\\n'")
    claim_part, evidence = text.split(evidence_marker, 1)
    claim = claim_part.removeprefix(claim_marker).strip()
    return claim, evidence.strip()


def convert_fever_documents(
    documents: list[dict[str, Any]],
    *,
    fixed_query: str = DEFAULT_FEVER_QUERY,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for document in documents:
        document_id = str(document.get("id", document.get("doc_id", "")))
        if not document_id:
            raise ValueError(f"FEVER document row is missing id/doc_id: {document!r}")
        query, evidence = split_claim_evidence(str(document.get("original_text", "")))
        label = binary_to_int(document.get("label"), field_name=f"FEVER label for {document_id}")
        row = {
            "id": document_id,
            "query": str(fixed_query),
            "document": f"Claim:\n{query}\n\nEvidence:\n{evidence}",
            "groundtruth": label,
            "document_id": document_id,
        }
        rows.append(row)
    return rows


def main() -> None:
    args = parse_args()
    documents_path = resolve_input_path(args.documents_path, PROJECT_ROOT)
    output_path = resolve_output_path(args.output_path, PROJECT_ROOT)
    documents = json.loads(documents_path.read_text(encoding="utf-8"))
    if not isinstance(documents, list):
        raise ValueError(f"{documents_path} must contain a JSON list")
    rows = convert_fever_documents(
        documents,
        fixed_query=str(args.fixed_query),
    )
    write_jsonl(rows, output_path)
    print(json.dumps({"converted_rows": len(rows), "output_path": str(output_path)}, sort_keys=True))


if __name__ == "__main__":
    main()
