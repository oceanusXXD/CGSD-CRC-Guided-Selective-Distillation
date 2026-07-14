from __future__ import annotations

import argparse
from collections import Counter
import gzip
import json
from pathlib import Path
import sys
from typing import Any
from urllib.request import urlopen


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mias_dcms.benchmark_data import (  # noqa: E402
    AG_NEWS_LABELS,
    DBPEDIA_14_LABELS,
    TREC_LABELS,
    build_helpsteer_attribute_index,
    normalize_ag_news_row,
    normalize_dbpedia_row,
    normalize_helpsteer_preference_row,
    normalize_trec_row,
    reservoir_sample_per_class,
)
from mias_dcms.algorithm_decision import recommend_algorithm_action  # noqa: E402
from mias_dcms.sampling_diagnostics import (  # noqa: E402
    aggregate_dual_order_probability,
    classification_random_baseline,
    classification_shift_report,
    preference_domain_effects,
    preference_random_baseline,
    preference_shift_report,
    select_classification_rows,
    select_rows,
    selector_safe_view,
    uncertainty_group_dependence_report,
)
from mias_dcms.evaluation_comparison import compare_scored_models  # noqa: E402
from mias_dcms.shift_gate import analyze_classification_shift, analyze_preference_shift  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Dataset preparation and sampling-shift diagnostics")
    subparsers = parser.add_subparsers(dest="command", required=True)

    classification_download = subparsers.add_parser("download-classification")
    classification_download.add_argument("--dataset", choices=("ag_news", "trec", "dbpedia_14"), required=True)
    classification_download.add_argument("--output-dir", type=Path, required=True)
    classification_download.add_argument("--per-class", type=int, default=5000)
    classification_download.add_argument("--seed", type=int, default=42)

    helpsteer_download = subparsers.add_parser("download-helpsteer2")
    helpsteer_download.add_argument("--output-dir", type=Path, required=True)

    classification_scoring = subparsers.add_parser("score-classification")
    classification_scoring.add_argument("--data-path", type=Path, required=True)
    classification_scoring.add_argument("--output-path", type=Path, required=True)
    classification_scoring.add_argument("--label-names")
    _add_scoring_arguments(classification_scoring)

    merge_classification = subparsers.add_parser("merge-classification-scores")
    merge_classification.add_argument("--source-path", type=Path, required=True)
    merge_classification.add_argument("--scored-paths", type=Path, nargs="+", required=True)
    merge_classification.add_argument("--output-path", type=Path, required=True)

    sanitize_classification = subparsers.add_parser("sanitize-classification-scores")
    sanitize_classification.add_argument("--input-path", type=Path, required=True)
    sanitize_classification.add_argument("--output-path", type=Path, required=True)

    preference_scoring = subparsers.add_parser("score-preference")
    preference_scoring.add_argument("--data-path", type=Path, required=True)
    preference_scoring.add_argument("--output-path", type=Path, required=True)
    _add_scoring_arguments(preference_scoring)

    classification_training = subparsers.add_parser("train-classification-lora")
    classification_training.add_argument("--train-path", type=Path, required=True)
    classification_training.add_argument("--label-names")
    _add_training_arguments(classification_training)

    dpo_training = subparsers.add_parser("train-dpo")
    dpo_training.add_argument("--train-path", type=Path, required=True)
    dpo_training.add_argument("--beta", type=float, default=0.1)
    _add_training_arguments(dpo_training)

    evaluation_comparison = subparsers.add_parser("compare-evaluations")
    evaluation_comparison.add_argument("--task", choices=("classification", "preference"), required=True)
    evaluation_comparison.add_argument("--base-path", type=Path, required=True)
    evaluation_comparison.add_argument("--random-path", type=Path, required=True)
    evaluation_comparison.add_argument("--uncertainty-path", type=Path, required=True)
    evaluation_comparison.add_argument("--output-path", type=Path, required=True)

    shift_analysis = subparsers.add_parser("analyze-shifts")
    shift_analysis.add_argument("--classification-diagnostics", type=Path)
    shift_analysis.add_argument("--preference-diagnostics", type=Path)
    shift_analysis.add_argument("--output-path", type=Path, required=True)
    shift_analysis.add_argument("--min-tv-delta", type=float, default=0.02)
    shift_analysis.add_argument("--min-enrichment-delta", type=float, default=0.10)
    shift_analysis.add_argument("--required-budgets", type=int, default=2)
    shift_analysis.add_argument("--min-relative-length-shift", type=float, default=0.10)
    shift_analysis.add_argument("--min-attribute-shift", type=float, default=0.25)
    shift_analysis.add_argument("--min-direction-tv-delta", type=float, default=0.05)
    shift_analysis.add_argument("--min-prompt-js-delta", type=float, default=0.01)
    shift_analysis.add_argument("--min-position-bias", type=float, default=0.20)
    shift_analysis.add_argument("--required-domains", type=int, default=2)

    algorithm_decision = subparsers.add_parser("decide-algorithm")
    algorithm_decision.add_argument("--shift-analysis", type=Path, required=True)
    algorithm_decision.add_argument("--classification-comparison", type=Path)
    algorithm_decision.add_argument("--preference-comparison", type=Path)
    algorithm_decision.add_argument("--output-path", type=Path, required=True)
    algorithm_decision.add_argument("--min-performance-drop", type=float, default=0.01)
    algorithm_decision.add_argument("--min-order-bias-improvement", type=float, default=0.05)

    classification_diagnostics = subparsers.add_parser("diagnose-classification")
    classification_diagnostics.add_argument("--scored-path", type=Path, required=True)
    classification_diagnostics.add_argument(
        "--oracle_store_path",
        type=Path,
        help="Optional JSON/JSONL labels keyed by id. Labels are joined only after selection for diagnostics.",
    )
    classification_diagnostics.add_argument("--output-dir", type=Path, required=True)
    classification_diagnostics.add_argument("--budgets", default="100,500,1000")
    classification_diagnostics.add_argument("--methods", default="random,entropy,margin")
    classification_diagnostics.add_argument("--seed", type=int, default=42)
    classification_diagnostics.add_argument("--random-repetitions", type=int, default=1000)
    classification_diagnostics.add_argument("--dependence-permutations", type=int, default=999)
    classification_diagnostics.add_argument("--quantile-bins", type=int, default=10)
    classification_diagnostics.add_argument("--dcms-target", choices=("uniform", "pool"), default="uniform")
    classification_diagnostics.add_argument(
        "--dcms-slack-grid",
        default="0,0.01,0.02,0.05,0.1,0.2,0.5",
    )
    classification_diagnostics.add_argument("--dcms-kappa", type=float, default=0.05)

    preference_diagnostics = subparsers.add_parser("diagnose-preference")
    preference_diagnostics.add_argument("--scored-path", type=Path, required=True)
    preference_diagnostics.add_argument("--output-dir", type=Path, required=True)
    preference_diagnostics.add_argument("--budget", type=int, required=True)
    preference_diagnostics.add_argument("--methods", default="random,entropy,margin")
    preference_diagnostics.add_argument("--seed", type=int, default=42)
    preference_diagnostics.add_argument("--random-repetitions", type=int, default=1000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "download-classification":
        download_classification_dataset(
            dataset_name=args.dataset,
            output_dir=args.output_dir,
            per_class=args.per_class,
            seed=args.seed,
        )
    elif args.command == "download-helpsteer2":
        download_helpsteer2(output_dir=args.output_dir)
    elif args.command == "score-classification":
        run_classification_scoring(args)
    elif args.command == "merge-classification-scores":
        merge_classification_scores(
            source_path=args.source_path,
            scored_paths=args.scored_paths,
            output_path=args.output_path,
        )
    elif args.command == "sanitize-classification-scores":
        sanitize_classification_scores(input_path=args.input_path, output_path=args.output_path)
    elif args.command == "score-preference":
        run_preference_scoring(args)
    elif args.command == "train-classification-lora":
        run_classification_training(args)
    elif args.command == "train-dpo":
        run_dpo_training(args)
    elif args.command == "compare-evaluations":
        run_evaluation_comparison(args)
    elif args.command == "analyze-shifts":
        run_shift_analysis(args)
    elif args.command == "decide-algorithm":
        run_algorithm_decision(args)
    elif args.command == "diagnose-classification":
        diagnose_classification(
            scored_path=args.scored_path,
            output_dir=args.output_dir,
            budgets=_parse_positive_ints(args.budgets),
            methods=_parse_methods(args.methods),
            seed=args.seed,
            oracle_store_path=args.oracle_store_path,
            random_repetitions=args.random_repetitions,
            dependence_permutations=args.dependence_permutations,
            quantile_bins=args.quantile_bins,
            dcms_target=args.dcms_target,
            dcms_slack_grid=_parse_nonnegative_floats(args.dcms_slack_grid),
            dcms_kappa=args.dcms_kappa,
        )
    elif args.command == "diagnose-preference":
        diagnose_preference(
            scored_path=args.scored_path,
            output_dir=args.output_dir,
            budget=args.budget,
            methods=_parse_methods(args.methods),
            seed=args.seed,
            random_repetitions=args.random_repetitions,
        )
    else:
        raise ValueError(f"unsupported command: {args.command}")


def _add_scoring_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--batch-size", type=int, default=16, help="Candidate sequences per model batch")
    parser.add_argument("--row-batch-size", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--torch-dtype", default="auto")
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--start-offset",
        type=int,
        default=0,
        help="Start at this zero-based row offset; use distinct outputs for parallel shards.",
    )
    parser.add_argument(
        "--sort-by-text-length",
        action="store_true",
        help="Score shorter classification prompts together; output ids can be restored with merge-classification-scores.",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument(
        "--save-representations",
        action="store_true",
        help="Save one frozen prompt hidden representation per classification row for BADGE/GALAXY.",
    )


def _add_training_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--target-modules", default="q_proj,k_proj,v_proj,o_proj")
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--mixed-precision", choices=("no", "fp16", "bf16"), default="no")
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--no-gradient-checkpointing", action="store_true")


def download_classification_dataset(
    *,
    dataset_name: str,
    output_dir: Path,
    per_class: int,
    seed: int,
) -> dict[str, Any]:
    from datasets import load_dataset

    output_dir.mkdir(parents=True, exist_ok=True)
    if dataset_name == "ag_news":
        dataset_id = "fancyzhx/ag_news"
        config_name = None
        normalizer = normalize_ag_news_row
    elif dataset_name == "trec":
        dataset_id = "CogComp/trec"
        config_name = None
        normalizer = normalize_trec_row
    elif dataset_name == "dbpedia_14":
        dataset_id = "fancyzhx/dbpedia_14"
        config_name = "dbpedia_14"
        normalizer = normalize_dbpedia_row
    else:
        raise ValueError(f"unsupported dataset: {dataset_name}")

    summary: dict[str, Any] = {
        "dataset": dataset_name,
        "source": dataset_id,
        "seed": seed,
        "per_class": per_class if dataset_name in {"dbpedia_14", "trec"} else None,
        "splits": {},
    }
    for split in ("train", "test"):
        if dataset_name == "trec":
            normalized_rows = (
                normalize_trec_row(row, split=split, index=index)
                for index, row in enumerate(_download_trec_raw_rows(split))
            )
            rows = reservoir_sample_per_class(
                normalized_rows,
                per_class=per_class,
                seed=seed,
            )
            output_path = output_dir / f"{split}.jsonl"
            _write_jsonl(output_path, rows)
            summary["splits"][split] = {
                "path": str(output_path),
                "size": len(rows),
                "class_counts": _class_counts(rows),
                "source_format": "official_trec_label_file_fallback",
            }
            continue
        dataset = load_dataset(dataset_id, config_name, split=split, streaming=True)
        normalized_rows = (
            normalizer(row, split=split, index=index) for index, row in enumerate(dataset)
        )
        if dataset_name == "dbpedia_14" and split == "train":
            rows = reservoir_sample_per_class(normalized_rows, per_class=per_class, seed=seed)
            filename = f"train_{per_class}_per_class.jsonl"
        else:
            rows = list(normalized_rows)
            filename = f"{split}.jsonl"
        output_path = output_dir / filename
        _write_jsonl(output_path, rows)
        summary["splits"][split] = {
            "path": str(output_path),
            "size": len(rows),
            "class_counts": _class_counts(rows),
        }
    _write_json(output_dir / "dataset_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


_TREC_RAW_URLS = {
    "train": "https://cogcomp.seas.upenn.edu/Data/QA/QC/train_5500.label",
    "test": "https://cogcomp.seas.upenn.edu/Data/QA/QC/TREC_10.label",
}


def _download_trec_raw_rows(split: str) -> list[dict[str, str]]:
    if split not in _TREC_RAW_URLS:
        raise ValueError(f"unsupported TREC split: {split!r}")
    rows: list[dict[str, str]] = []
    with urlopen(_TREC_RAW_URLS[split], timeout=60) as response:
        for raw_line in response:
            line = raw_line.replace(b"\xf0", b" ").strip().decode("utf-8")
            fine_label, separator, text = line.partition(" ")
            if not separator or ":" not in fine_label:
                raise ValueError(f"malformed TREC row in split {split!r}: {line[:120]!r}")
            rows.append(
                {
                    "text": text,
                    "coarse_label": fine_label.split(":", 1)[0],
                    "fine_label": fine_label,
                }
            )
    return rows


def download_helpsteer2(*, output_dir: Path) -> dict[str, Any]:
    from huggingface_hub import hf_hub_download

    output_dir.mkdir(parents=True, exist_ok=True)
    base_rows: list[dict[str, Any]] = []
    for filename in ("train.jsonl.gz", "validation.jsonl.gz"):
        path = hf_hub_download("nvidia/HelpSteer2", filename, repo_type="dataset")
        base_rows.extend(_read_gzip_jsonl(Path(path)))
    attribute_index = build_helpsteer_attribute_index(base_rows)
    preference_path = Path(
        hf_hub_download(
            "nvidia/HelpSteer2",
            "preference/preference.jsonl.gz",
            repo_type="dataset",
        )
    )
    by_split: dict[str, list[dict[str, Any]]] = {}
    matched_attributes = 0
    response_count = 0
    for index, row in enumerate(_read_gzip_jsonl(preference_path)):
        normalized = normalize_helpsteer_preference_row(row, attribute_index, index=index)
        split = str(normalized["split"])
        by_split.setdefault(split, []).append(normalized)
        for key in ("response_1_attributes", "response_2_attributes"):
            response_count += 1
            if normalized[key] is not None:
                matched_attributes += 1

    summary: dict[str, Any] = {
        "dataset": "helpsteer2_preference",
        "source": "nvidia/HelpSteer2",
        "attribute_rows": len(base_rows),
        "attribute_match_count": matched_attributes,
        "attribute_response_count": response_count,
        "attribute_match_rate": matched_attributes / response_count if response_count else 0.0,
        "splits": {},
    }
    for split, rows in sorted(by_split.items()):
        output_path = output_dir / f"{split}.jsonl"
        _write_jsonl(output_path, rows)
        summary["splits"][split] = {
            "path": str(output_path),
            "size": len(rows),
            "preference_direction_counts": dict(
                sorted(Counter(str(row["preferred_response"]) for row in rows).items())
            ),
        }
    _write_json(output_dir / "dataset_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def run_classification_scoring(args: argparse.Namespace) -> dict[str, Any]:
    source_rows = _read_jsonl(args.data_path)
    start_offset = int(args.start_offset)
    if start_offset < 0 or start_offset > len(source_rows):
        raise ValueError(f"start offset must be between 0 and {len(source_rows)}")
    rows = source_rows[start_offset:]
    if args.limit is not None:
        rows = rows[: args.limit]
    if bool(args.sort_by_text_length):
        rows.sort(key=lambda row: (len(str(row.get("text", ""))), str(row["id"])))
    label_names = _resolve_label_names(rows, args.label_names)
    resumed_size = _resume_size(rows, args.output_path, resume=args.resume)
    scorer = _load_scorer(args) if resumed_size < len(rows) else None
    scored = score_classification_to_path(
        rows,
        output_path=args.output_path,
        label_names=label_names,
        scorer=scorer,
        row_batch_size=args.row_batch_size,
        resume=args.resume,
        save_representations=bool(args.save_representations),
    )
    evaluated = [row for row in scored if row.get("prediction_correct") is not None]
    summary = {
        "task": "classification_zero_shot",
        "model": args.model,
        "data_path": str(args.data_path),
        "output_path": str(args.output_path),
        "start_offset": start_offset,
        "sort_by_text_length": bool(args.sort_by_text_length),
        "size": len(scored),
        "resumed_size": resumed_size,
        "newly_scored_size": len(scored) - resumed_size,
        "label_names": label_names,
        "accuracy": (
            sum(bool(row["prediction_correct"]) for row in evaluated) / len(evaluated)
            if evaluated
            else None
        ),
    }
    _write_json(_summary_path(args.output_path), summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def merge_classification_scores(
    *,
    source_path: Path,
    scored_paths: list[Path],
    output_path: Path,
) -> dict[str, Any]:
    """Combine independently scored shards after proving exact id coverage."""
    source_rows = _read_jsonl(source_path)
    source_ids = [str(row.get("id", "")) for row in source_rows]
    if not all(source_ids) or len(set(source_ids)) != len(source_ids):
        raise ValueError("source rows must contain unique non-empty ids")

    scored_by_id: dict[str, dict[str, Any]] = {}
    for scored_path in scored_paths:
        for row in _read_jsonl(scored_path):
            sample_id = str(row.get("id", ""))
            if not sample_id:
                raise ValueError(f"scored row in {scored_path} is missing id")
            if sample_id in scored_by_id:
                raise ValueError(f"duplicate scored id across shards: {sample_id!r}")
            scored_by_id[sample_id] = row

    source_id_set = set(source_ids)
    missing = [sample_id for sample_id in source_ids if sample_id not in scored_by_id]
    extra = sorted(set(scored_by_id) - source_id_set)
    if missing or extra:
        raise ValueError(
            "scored shard coverage does not match source rows: "
            f"missing={missing[:5]}, extra={extra[:5]}"
        )

    merged_rows = [scored_by_id[sample_id] for sample_id in source_ids]
    _write_jsonl(output_path, merged_rows)
    summary = {
        "task": "merge_classification_scores",
        "source_path": str(source_path),
        "scored_paths": [str(path) for path in scored_paths],
        "output_path": str(output_path),
        "size": len(merged_rows),
    }
    _write_json(_summary_path(output_path), summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def sanitize_classification_scores(*, input_path: Path, output_path: Path) -> dict[str, Any]:
    """Materialize a selector-safe scored classification pool without rescoring."""
    rows = selector_safe_view(_read_jsonl(input_path))
    _write_jsonl(output_path, rows)
    summary = {
        "task": "sanitize_classification_scores",
        "input_path": str(input_path),
        "output_path": str(output_path),
        "size": len(rows),
    }
    _write_json(_summary_path(output_path), summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def run_preference_scoring(args: argparse.Namespace) -> dict[str, Any]:
    rows = _read_jsonl(args.data_path)
    if args.limit is not None:
        rows = rows[: args.limit]
    resumed_size = _resume_size(rows, args.output_path, resume=args.resume)
    scorer = _load_scorer(args) if resumed_size < len(rows) else None
    scored = score_preference_to_path(
        rows,
        output_path=args.output_path,
        scorer=scorer,
        row_batch_size=args.row_batch_size,
        resume=args.resume,
    )
    evaluated = [row for row in scored if row.get("prediction_correct") is not None]
    summary = {
        "task": "pairwise_zero_shot_dual_order",
        "model": args.model,
        "data_path": str(args.data_path),
        "output_path": str(args.output_path),
        "size": len(scored),
        "resumed_size": resumed_size,
        "newly_scored_size": len(scored) - resumed_size,
        "accuracy_excluding_ties": (
            sum(bool(row["prediction_correct"]) for row in evaluated) / len(evaluated)
            if evaluated
            else None
        ),
        "mean_order_disagreement": (
            sum(float(row["order_disagreement"]) for row in scored) / len(scored) if scored else None
        ),
    }
    _write_json(_summary_path(args.output_path), summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def run_classification_training(args: argparse.Namespace) -> dict[str, Any]:
    from mias_dcms.benchmark_training import train_classification_lora

    rows = _read_jsonl(args.train_path)
    label_names = _resolve_label_names(rows, args.label_names)
    summary = train_classification_lora(
        rows,
        label_names=label_names,
        config=_training_config(args),
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def run_dpo_training(args: argparse.Namespace) -> dict[str, Any]:
    from mias_dcms.benchmark_training import train_preference_dpo

    rows = _read_jsonl(args.train_path)
    summary = train_preference_dpo(rows, config=_training_config(args, beta=args.beta))
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def run_evaluation_comparison(args: argparse.Namespace) -> dict[str, Any]:
    report = compare_scored_models(
        task=args.task,
        base_rows=_read_jsonl(args.base_path),
        random_rows=_read_jsonl(args.random_path),
        uncertainty_rows=_read_jsonl(args.uncertainty_path),
    )
    report["paths"] = {
        "base": str(args.base_path),
        "random": str(args.random_path),
        "uncertainty": str(args.uncertainty_path),
    }
    _write_json(args.output_path, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return report


def run_shift_analysis(args: argparse.Namespace) -> dict[str, Any]:
    if args.classification_diagnostics is None and args.preference_diagnostics is None:
        raise ValueError("provide at least one diagnostics path")
    analysis: dict[str, Any] = {}
    if args.classification_diagnostics is not None:
        classification_report = json.loads(
            args.classification_diagnostics.read_text(encoding="utf-8")
        )
        analysis["classification"] = analyze_classification_shift(
            classification_report,
            min_tv_delta=args.min_tv_delta,
            min_enrichment_delta=args.min_enrichment_delta,
            required_budgets=args.required_budgets,
        )
    if args.preference_diagnostics is not None:
        preference_report = json.loads(args.preference_diagnostics.read_text(encoding="utf-8"))
        analysis["preference"] = analyze_preference_shift(
            preference_report,
            min_relative_length_shift=args.min_relative_length_shift,
            min_attribute_shift=args.min_attribute_shift,
            min_direction_tv_delta=args.min_direction_tv_delta,
            min_prompt_js_delta=args.min_prompt_js_delta,
            min_position_bias=args.min_position_bias,
            required_domains=args.required_domains,
        )
    _write_json(args.output_path, analysis)
    print(json.dumps(analysis, ensure_ascii=False, indent=2))
    return analysis


def run_algorithm_decision(args: argparse.Namespace) -> dict[str, Any]:
    shift_analysis = json.loads(args.shift_analysis.read_text(encoding="utf-8"))
    classification_comparison = (
        json.loads(args.classification_comparison.read_text(encoding="utf-8"))
        if args.classification_comparison is not None
        else None
    )
    preference_comparison = (
        json.loads(args.preference_comparison.read_text(encoding="utf-8"))
        if args.preference_comparison is not None
        else None
    )
    decision = recommend_algorithm_action(
        shift_analysis=shift_analysis,
        classification_comparison=classification_comparison,
        preference_comparison=preference_comparison,
        min_performance_drop=args.min_performance_drop,
        min_order_bias_improvement=args.min_order_bias_improvement,
    )
    _write_json(args.output_path, decision)
    print(json.dumps(decision, ensure_ascii=False, indent=2))
    return decision


def score_classification_rows(
    rows: list[dict[str, Any]],
    *,
    label_names: list[str],
    scorer: Any,
    row_batch_size: int,
    save_representations: bool = False,
) -> list[dict[str, Any]]:
    from mias_dcms.zero_shot_scoring import build_classification_messages

    if row_batch_size <= 0:
        raise ValueError("row_batch_size must be positive")
    candidates = _label_codes(len(label_names))
    scored: list[dict[str, Any]] = []
    for start in range(0, len(rows), row_batch_size):
        batch = rows[start : start + row_batch_size]
        messages = [build_classification_messages(str(row["text"]), label_names) for row in batch]
        if save_representations and hasattr(scorer, "score_messages_with_representations"):
            probability_rows, representation_rows = scorer.score_messages_with_representations(
                messages,
                candidates,
            )
        else:
            probability_rows = scorer.score_messages(messages, candidates)
            representation_rows = [None for _ in batch]
        if len(probability_rows) != len(batch):
            raise ValueError("scorer returned an unexpected number of probability rows")
        for row, probabilities, representation in zip(
            batch,
            probability_rows,
            representation_rows,
            strict=True,
        ):
            predicted_label = max(range(len(probabilities)), key=probabilities.__getitem__)
            groundtruth = int(row["label"]) if row.get("label") is not None else None
            scored_row = {
                **row,
                "label_codes": candidates,
                "probabilities": [float(value) for value in probabilities],
                "predicted_label": predicted_label,
                "predicted_label_name": label_names[predicted_label],
                **_uncertainty_fields(probabilities),
            }
            if groundtruth is not None:
                scored_row["prediction_correct"] = predicted_label == groundtruth
            if representation is not None:
                scored_row["representation_embedding"] = [float(value) for value in representation]
            scored.append(scored_row)
    return scored


def score_classification_to_path(
    rows: list[dict[str, Any]],
    *,
    output_path: Path,
    label_names: list[str],
    scorer: Any | None,
    row_batch_size: int,
    resume: bool,
    save_representations: bool = False,
) -> list[dict[str, Any]]:
    existing, pending = _prepare_incremental_rows(rows, output_path, resume=resume)
    if pending and scorer is None:
        raise ValueError("a scorer is required when rows remain to be scored")
    new_rows: list[dict[str, Any]] = []
    if not resume or not output_path.exists():
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text("", encoding="utf-8")
    for start in range(0, len(pending), row_batch_size):
        batch = pending[start : start + row_batch_size]
        scored_batch = score_classification_rows(
            batch,
            label_names=label_names,
            scorer=scorer,
            row_batch_size=len(batch),
            save_representations=save_representations,
        )
        _append_jsonl(output_path, scored_batch)
        new_rows.extend(scored_batch)
    return _restore_input_order(rows, existing + new_rows)


def score_preference_rows(
    rows: list[dict[str, Any]],
    *,
    scorer: Any,
    row_batch_size: int,
) -> list[dict[str, Any]]:
    from mias_dcms.zero_shot_scoring import build_pairwise_messages

    if row_batch_size <= 0:
        raise ValueError("row_batch_size must be positive")
    scored: list[dict[str, Any]] = []
    for start in range(0, len(rows), row_batch_size):
        batch = rows[start : start + row_batch_size]
        messages: list[list[dict[str, str]]] = []
        for row in batch:
            prompt = str(row["prompt"])
            response_1 = str(row["response_1"])
            response_2 = str(row["response_2"])
            messages.append(build_pairwise_messages(prompt, response_1, response_2))
            messages.append(build_pairwise_messages(prompt, response_2, response_1))
        probability_rows = scorer.score_messages(messages, ["A", "B"])
        if len(probability_rows) != 2 * len(batch):
            raise ValueError("scorer returned an unexpected number of pairwise probability rows")
        for index, row in enumerate(batch):
            probability_first_order_12 = float(probability_rows[2 * index][0])
            probability_first_order_21 = float(probability_rows[2 * index + 1][0])
            probability_response_1 = aggregate_dual_order_probability(
                probability_first_order_12=probability_first_order_12,
                probability_first_order_21=probability_first_order_21,
            )
            probabilities = [probability_response_1, 1.0 - probability_response_1]
            predicted_response = 1 if probability_response_1 >= 0.5 else 2
            preferred_response = int(row.get("preferred_response", 0))
            scored.append(
                {
                    **row,
                    "probability_first_order_12": probability_first_order_12,
                    "probability_first_order_21": probability_first_order_21,
                    "probability_response_1": probability_response_1,
                    "probabilities": probabilities,
                    "predicted_response": predicted_response,
                    "prediction_correct": (
                        predicted_response == preferred_response if preferred_response in (1, 2) else None
                    ),
                    "order_disagreement": abs(
                        probability_first_order_12 - (1.0 - probability_first_order_21)
                    ),
                    **_uncertainty_fields(probabilities),
                }
            )
    return scored


def score_preference_to_path(
    rows: list[dict[str, Any]],
    *,
    output_path: Path,
    scorer: Any | None,
    row_batch_size: int,
    resume: bool,
) -> list[dict[str, Any]]:
    existing, pending = _prepare_incremental_rows(rows, output_path, resume=resume)
    if pending and scorer is None:
        raise ValueError("a scorer is required when rows remain to be scored")
    new_rows: list[dict[str, Any]] = []
    if not resume or not output_path.exists():
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text("", encoding="utf-8")
    for start in range(0, len(pending), row_batch_size):
        batch = pending[start : start + row_batch_size]
        scored_batch = score_preference_rows(
            batch,
            scorer=scorer,
            row_batch_size=len(batch),
        )
        _append_jsonl(output_path, scored_batch)
        new_rows.extend(scored_batch)
    return _restore_input_order(rows, existing + new_rows)


def diagnose_classification(
    *,
    scored_path: Path,
    output_dir: Path,
    budgets: list[int],
    methods: list[str],
    seed: int,
    oracle_store_path: Path | None = None,
    random_repetitions: int = 1000,
    dependence_permutations: int = 999,
    quantile_bins: int = 10,
    dcms_target: str = "uniform",
    dcms_slack_grid: list[float] | tuple[float, ...] = (0.0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5),
    dcms_kappa: float = 0.05,
) -> dict[str, Any]:
    rows = _read_jsonl(scored_path)
    audit_rows = (
        _attach_classification_oracle(rows, _read_classification_oracle_store(oracle_store_path))
        if oracle_store_path is not None
        else rows
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    if budgets and max(budgets) > len(rows):
        raise ValueError(f"largest budget {max(budgets)} exceeds row count {len(rows)}")
    random_baseline = classification_random_baseline(
        audit_rows,
        budgets=budgets,
        repetitions=random_repetitions,
        seed=seed,
    )
    summary: dict[str, Any] = {
        "scored_path": str(scored_path),
        "oracle_store_path": str(oracle_store_path) if oracle_store_path is not None else None,
        "pool_size": len(rows),
        "seed": seed,
        "random_baseline": random_baseline,
        "dependence": {},
        "methods": {},
        "method_summaries": {},
    }
    for method in methods:
        if method in {"entropy", "margin"}:
            summary["dependence"][method] = uncertainty_group_dependence_report(
                audit_rows,
                method=method,
                quantile_bins=quantile_bins,
                permutations=dependence_permutations,
                seed=seed,
            )
        elif method in {"badge", "galaxy"}:
            summary["dependence"][method] = {
                "method": method,
                "representation_based": True,
                "note": "selection uses frozen prompt representations; uncertainty dependence is not directly applicable",
            }
        method_reports: dict[str, Any] = {}
        safe_rows = selector_safe_view(rows)
        rows_by_id = {str(row["id"]): row for row in audit_rows}
        normalized_method = str(method).strip().lower().replace("+", "_").replace("-", "_")
        cached_selection: list[dict[str, Any]] | None = None
        if budgets and not normalized_method.endswith("_dcms"):
            cached_selection, _ = select_classification_rows(
                safe_rows,
                method=method,
                budget=max(budgets),
                seed=seed,
                dcms_target=dcms_target,
                dcms_slack_grid=dcms_slack_grid,
                dcms_kappa=dcms_kappa,
            )
        for budget in budgets:
            if cached_selection is None:
                selected_safe, selection_metadata = select_classification_rows(
                    safe_rows,
                    method=method,
                    budget=budget,
                    seed=seed,
                    dcms_target=dcms_target,
                    dcms_slack_grid=dcms_slack_grid,
                    dcms_kappa=dcms_kappa,
                )
            else:
                selected_safe = cached_selection[:budget]
                selection_metadata = None
            selected = [rows_by_id[str(row["id"])] for row in selected_safe]
            _write_jsonl(output_dir / f"{method}_budget_{budget}.jsonl", selected)
            selected_ids = {str(row["id"]) for row in selected}
            budget_report = classification_shift_report(audit_rows, selected_ids=selected_ids)
            _attach_random_calibration(
                budget_report,
                random_baseline["budgets"][str(budget)],
                global_max_z_q95=float(random_baseline["global_envelope"]["max_tv_z_q95"]),
            )
            if selection_metadata is not None:
                selection_path = output_dir / f"{method}_budget_{budget}_selection.json"
                _write_json(selection_path, selection_metadata)
                budget_report["dcms"] = {
                    "target": selection_metadata["target"],
                    "target_moments": selection_metadata["target_moments"],
                    "utility_retained": selection_metadata["utility_retained"],
                    "max_constraint_violation": selection_metadata["max_constraint_violation"],
                    "selected_slack": selection_metadata["selected_slack"],
                    "solver_status": selection_metadata["solver_status"],
                    "selection_artifact": str(selection_path),
                }
            method_reports[str(budget)] = budget_report
        summary["methods"][method] = method_reports
        if method != "random":
            summary["method_summaries"][method] = _classification_method_summary(method_reports)
    _write_json(output_dir / "classification_diagnostics.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def _attach_random_calibration(
    report: dict[str, Any],
    baseline: dict[str, Any],
    *,
    global_max_z_q95: float,
) -> None:
    tv = float(report["category_tv"])
    tv_mean = float(baseline["tv_mean"])
    tv_std = float(baseline["tv_std"])
    tv_q95 = float(baseline["tv_q95"])
    tv_zscore = (tv - tv_mean) / tv_std if tv_std > 0.0 else 0.0
    report.update(
        {
            "random_tv_mean": tv_mean,
            "random_tv_q95": tv_q95,
            "excess_tv_vs_random_mean": tv - tv_mean,
            "excess_tv_vs_random_q95": tv - tv_q95,
            "tv_zscore": tv_zscore,
            "global_envelope_exceeded": tv_zscore > global_max_z_q95,
        }
    )
    for label, values in report.get("per_class", {}).items():
        random_values = baseline.get("per_class", {}).get(label)
        if not random_values or values.get("enrichment") is None:
            continue
        enrichment = float(values["enrichment"])
        values.update(
            {
                "random_enrichment_mean": float(random_values["enrichment_mean"]),
                "random_enrichment_q05": float(random_values["enrichment_q05"]),
                "random_enrichment_q95": float(random_values["enrichment_q95"]),
                "enrichment_excess_vs_random_q95": enrichment
                - float(random_values["enrichment_q95"]),
            }
        )


def _classification_method_summary(method_reports: dict[str, Any]) -> dict[str, Any]:
    budgets = sorted((int(value) for value in method_reports), key=int)
    centered = [
        float(method_reports[str(budget)]["excess_tv_vs_random_mean"]) for budget in budgets
    ]
    positive = [max(0.0, value) for value in centered]
    return {
        "aas_centered": _normalized_trapezoid_area(budgets, centered),
        "aas_positive": _normalized_trapezoid_area(budgets, positive),
        "global_envelope_exceeded_budgets": [
            str(budget)
            for budget in budgets
            if method_reports[str(budget)]["global_envelope_exceeded"]
        ],
        "worst_group_coverage": min(
            float(method_reports[str(budget)]["worst_group_coverage"]) for budget in budgets
        ),
    }


def _normalized_trapezoid_area(xs: list[int], ys: list[float]) -> float:
    if len(xs) != len(ys) or not xs:
        raise ValueError("xs and ys must be non-empty and have the same length")
    if len(xs) == 1:
        return ys[0]
    width = xs[-1] - xs[0]
    if width <= 0:
        return sum(ys) / len(ys)
    area = sum(
        (xs[index] - xs[index - 1]) * (ys[index] + ys[index - 1]) / 2.0
        for index in range(1, len(xs))
    )
    return area / width


def diagnose_preference(
    *,
    scored_path: Path,
    output_dir: Path,
    budget: int,
    methods: list[str],
    seed: int,
    random_repetitions: int = 1000,
) -> dict[str, Any]:
    rows = _read_jsonl(scored_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    if budget > len(rows):
        raise ValueError(f"budget {budget} exceeds row count {len(rows)}")
    trainable_rows = [row for row in rows if int(row.get("preferred_response", 0)) in (1, 2)]
    if budget > len(trainable_rows):
        raise ValueError(
            f"budget {budget} exceeds non-tie row count {len(trainable_rows)} required for DPO"
        )
    random_baseline = preference_random_baseline(
        rows,
        budget=budget,
        repetitions=random_repetitions,
        seed=seed,
    )
    dpo_random_baseline = preference_random_baseline(
        trainable_rows,
        budget=budget,
        repetitions=random_repetitions,
        seed=seed,
    )
    summary: dict[str, Any] = {
        "scored_path": str(scored_path),
        "pool_size": len(rows),
        "trainable_pool_size": len(trainable_rows),
        "tie_pool_size": len(rows) - len(trainable_rows),
        "budget": budget,
        "seed": seed,
        "random_baseline": random_baseline,
        "dpo_random_baseline": dpo_random_baseline,
        "methods": {},
        "dpo_methods": {},
    }
    for method in methods:
        safe_rows = selector_safe_view(rows)
        rows_by_id = {str(row["id"]): row for row in rows}
        selected_safe = select_rows(safe_rows, method=method, budget=budget, seed=seed)
        selected = [rows_by_id[str(row["id"])] for row in selected_safe]
        _write_jsonl(output_dir / f"{method}_budget_{budget}.jsonl", selected)
        selected_ids = {str(row["id"]) for row in selected}
        all_pair_report = preference_shift_report(rows, selected_ids=selected_ids)
        _attach_preference_calibration(all_pair_report, random_baseline)
        summary["methods"][method] = all_pair_report
        trainable_safe = selector_safe_view(trainable_rows)
        trainable_by_id = {str(row["id"]): row for row in trainable_rows}
        dpo_selected_safe = select_rows(trainable_safe, method=method, budget=budget, seed=seed)
        dpo_selected = [trainable_by_id[str(row["id"])] for row in dpo_selected_safe]
        _write_jsonl(output_dir / f"{method}_dpo_budget_{budget}.jsonl", dpo_selected)
        dpo_selected_ids = {str(row["id"]) for row in dpo_selected}
        dpo_report = preference_shift_report(
            trainable_rows, selected_ids=dpo_selected_ids
        )
        _attach_preference_calibration(dpo_report, dpo_random_baseline)
        summary["dpo_methods"][method] = dpo_report
    reference_method = next(iter(summary["methods"].values()), {})
    order_pool_mean = (
        reference_method.get("scoring", {})
        .get("order_disagreement", {})
        .get("pool_mean")
    )
    summary["scorer_reliability"] = {
        "dual_order_aggregation": "symmetric_response_probability",
        "mean_order_disagreement": order_pool_mean,
        "requires_repair_before_dpo": (
            order_pool_mean is not None and float(order_pool_mean) >= 0.20
        ),
    }
    _write_json(output_dir / "preference_diagnostics.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def _attach_preference_calibration(
    report: dict[str, Any],
    baseline: dict[str, Any],
) -> None:
    effects = preference_domain_effects(report)
    random_q95 = {
        domain: float(values["q95"]) for domain, values in baseline["domains"].items()
    }
    excess = {
        domain: effects[domain] - random_q95[domain]
        for domain in effects
    }
    report.update(
        {
            "domain_effects": effects,
            "random_domain_q95": random_q95,
            "domain_excess_vs_random_q95": excess,
            "domains_exceeding_random_q95": sorted(
                domain for domain, value in excess.items() if value > 0.0
            ),
        }
    )


def _parse_positive_ints(value: str) -> list[int]:
    parsed = sorted({int(item.strip()) for item in value.split(",") if item.strip()})
    if not parsed or any(item <= 0 for item in parsed):
        raise ValueError("expected a comma-separated list of positive integers")
    return parsed


def _parse_nonnegative_floats(value: str) -> list[float]:
    parsed = sorted({float(item.strip()) for item in value.split(",") if item.strip()})
    if not parsed or any(item < 0.0 for item in parsed):
        raise ValueError("expected a comma-separated list of non-negative numbers")
    return parsed


def _parse_methods(value: str) -> list[str]:
    aliases = {
        "entropy+dcms": "entropy_dcms",
        "badge+dcms": "badge_dcms",
    }
    methods = [aliases.get(item.strip().lower(), item.strip().lower()) for item in value.split(",") if item.strip()]
    unsupported = sorted(
        set(methods) - {"random", "entropy", "margin", "badge", "galaxy", "entropy_dcms", "badge_dcms"}
    )
    if not methods or unsupported:
        raise ValueError(f"unsupported methods: {unsupported}")
    return methods


def _load_scorer(args: argparse.Namespace) -> Any:
    from mias_dcms.zero_shot_scoring import CausalCandidateScorer

    device_map = None if str(args.device_map).lower() in {"none", "cpu"} else args.device_map
    return CausalCandidateScorer.from_pretrained(
        args.model,
        batch_size=args.batch_size,
        max_length=args.max_length,
        device_map=device_map,
        torch_dtype=args.torch_dtype,
        trust_remote_code=args.trust_remote_code,
    )


def _training_config(args: argparse.Namespace, *, beta: float = 0.1) -> Any:
    from mias_dcms.benchmark_training import LoraTrainingConfig

    target_modules = tuple(item.strip() for item in args.target_modules.split(",") if item.strip())
    if not target_modules:
        raise ValueError("target_modules cannot be empty")
    return LoraTrainingConfig(
        model_name_or_path=args.model,
        output_dir=args.output_dir,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_length=args.max_length,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=target_modules,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        max_grad_norm=args.max_grad_norm,
        mixed_precision=args.mixed_precision,
        dtype=args.dtype,
        gradient_checkpointing=not args.no_gradient_checkpointing,
        seed=args.seed,
        num_workers=args.num_workers,
        beta=beta,
    )


def _resolve_label_names(rows: list[dict[str, Any]], explicit: str | None) -> list[str]:
    if explicit:
        label_names = [item.strip() for item in explicit.split(",") if item.strip()]
        _label_codes(len(label_names))
        return label_names
    dataset_name = str(rows[0].get("dataset", "")) if rows else ""
    if dataset_name == "ag_news":
        return list(AG_NEWS_LABELS)
    if dataset_name == "dbpedia_14":
        return list(DBPEDIA_14_LABELS)
    if dataset_name == "trec":
        return list(TREC_LABELS)
    names_by_label = {
        int(row["label"]): str(row["label_name"])
        for row in rows
        if row.get("label") is not None and row.get("label_name") is not None
    }
    if names_by_label and sorted(names_by_label) == list(range(len(names_by_label))):
        return [names_by_label[index] for index in range(len(names_by_label))]
    raise ValueError("could not infer label names; pass --label-names")


def _label_codes(count: int) -> list[str]:
    if not 2 <= count <= 26:
        raise ValueError("label count must be between 2 and 26")
    return [chr(ord("A") + index) for index in range(count)]


def _uncertainty_fields(probabilities: Any) -> dict[str, float]:
    import math

    values = [float(value) for value in probabilities]
    ordered = sorted(values, reverse=True)
    return {
        "entropy": -sum(value * math.log(value) for value in values if value > 0.0),
        "margin": ordered[0] - ordered[1],
    }


def _class_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    return dict(sorted(Counter(str(int(row["label"])) for row in rows).items(), key=lambda item: int(item[0])))


def _resume_size(rows: list[dict[str, Any]], output_path: Path, *, resume: bool) -> int:
    existing, _ = _prepare_incremental_rows(rows, output_path, resume=resume)
    return len(existing)


def _prepare_incremental_rows(
    rows: list[dict[str, Any]], output_path: Path, *, resume: bool
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    input_ids = [str(row["id"]) for row in rows]
    if len(input_ids) != len(set(input_ids)):
        raise ValueError("input rows contain duplicate ids")
    existing = _read_jsonl(output_path) if resume and output_path.exists() else []
    existing_ids = [str(row["id"]) for row in existing]
    if len(existing_ids) != len(set(existing_ids)):
        raise ValueError("existing scored output contains duplicate ids")
    unexpected = sorted(set(existing_ids) - set(input_ids))
    if unexpected:
        raise ValueError(f"existing scored output contains ids outside the input: {unexpected[:5]}")
    existing_id_set = set(existing_ids)
    pending = [row for row in rows if str(row["id"]) not in existing_id_set]
    return existing, pending


def _restore_input_order(
    input_rows: list[dict[str, Any]], scored_rows: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    by_id = {str(row["id"]): row for row in scored_rows}
    missing = [str(row["id"]) for row in input_rows if str(row["id"]) not in by_id]
    if missing:
        raise ValueError(f"scored output is missing ids: {missing[:5]}")
    return [by_id[str(row["id"])] for row in input_rows]


def _read_gzip_jsonl(path: Path) -> list[dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _read_classification_oracle_store(path: Path) -> dict[str, dict[str, Any]]:
    if path.suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict) or not all(isinstance(value, dict) for value in payload.values()):
            raise ValueError("classification oracle JSON must be an object keyed by id")
        return {str(sample_id): dict(row) for sample_id, row in payload.items()}
    rows = _read_jsonl(path)
    oracle: dict[str, dict[str, Any]] = {}
    for row in rows:
        sample_id = row.get("id")
        if sample_id is None:
            raise ValueError("classification oracle JSONL rows must contain id")
        key = str(sample_id)
        if key in oracle:
            raise ValueError(f"classification oracle contains duplicate id: {key!r}")
        oracle[key] = dict(row)
    return oracle


def _attach_classification_oracle(
    scored_rows: list[dict[str, Any]],
    oracle_store: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    audit_rows: list[dict[str, Any]] = []
    for row in scored_rows:
        sample_id = str(row.get("id", ""))
        if not sample_id:
            raise ValueError("scored classification rows must contain id")
        oracle = oracle_store.get(sample_id)
        if oracle is None:
            raise ValueError(f"classification oracle is missing scored id: {sample_id!r}")
        if "label" not in oracle:
            raise ValueError(f"classification oracle row {sample_id!r} is missing label")
        existing_label = row.get("label")
        if existing_label is not None and str(existing_label) != str(oracle["label"]):
            raise ValueError(f"classification label mismatch for id: {sample_id!r}")
        audit_rows.append({**row, "label": oracle["label"]})
    extra_ids = sorted(set(oracle_store) - {str(row["id"]) for row in scored_rows})
    if extra_ids:
        raise ValueError(f"classification oracle contains ids outside scored rows: {extra_ids[:5]}")
    return audit_rows


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _append_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _summary_path(output_path: Path) -> Path:
    return output_path.with_name(f"{output_path.stem}.summary.json")


if __name__ == "__main__":
    main()
