#!/usr/bin/env python

from __future__ import annotations

import argparse
import json
import math
import os
import signal
import subprocess
import sys
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.cli_common import (
    add_stage_cache_args,
    estimate_query_document_prompt_tokens,
    input_artifact_path,
    load_selected_train_rows,
    load_split_ids,
    load_stage_examples,
    load_teacher_labels,
    output_artifact_path,
    output_dir_from_arg,
    print_existing_stage_result,
    read_jsonl,
    selected_train_rows_path,
    split_examples,
    split_ids_path,
    stage_cache_decision,
    summarize_teacher_label_usage,
    train_label_snapshot,
    write_stage_usage,
    StageCacheDecision,
)
from src.binary_protocol import BINARY_SCORE_SOURCE, normalize_binary_token
from src.data import PairExample, format_binary_chat_prompt
from src.utils import read_json, write_json, write_jsonl


SCORE_SOURCE = f"{BINARY_SCORE_SOURCE}_vllm_openai"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--round_index", type=int, required=True)
    parser.add_argument("--model_path", default="model/qwen3-0.6b")
    parser.add_argument("--data_path", default="datasets")
    parser.add_argument("--query_field", default="query")
    parser.add_argument("--document_field", default="document")
    parser.add_argument("--label_field", default="groundtruth")
    parser.add_argument("--split_ids_path", default=None)
    parser.add_argument("--selected_train_rows_path", default=None)
    parser.add_argument("--checkpoint_dir", default=None)
    parser.add_argument("--teacher_labels_path", default=None)
    parser.add_argument("--teacher_temperature", type=float, default=1.0)
    parser.add_argument("--all_predictions_path", default=None)
    parser.add_argument("--guide_predictions_path", default=None)
    parser.add_argument("--final_predictions_path", default=None)
    parser.add_argument("--pool_predictions_path", default=None)
    parser.add_argument("--train_label_snapshot_path", default=None)
    parser.add_argument("--usage_path", default=None)

    parser.add_argument("--base_url", default="http://localhost:8000/v1")
    parser.add_argument("--api_key", default="EMPTY")
    parser.add_argument("--served_model_name", default=None)
    parser.add_argument("--lora_model_name", default=None)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--parallel_requests", type=int, default=1024)
    parser.add_argument("--request_retries", type=int, default=3)
    parser.add_argument("--progress_timeout", type=int, default=None)
    parser.add_argument("--top_logprobs", type=int, default=20)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max_tokens", type=int, default=1)
    parser.add_argument("--start_server", action="store_true", default=False)
    parser.add_argument("--python_executable", default=None)
    parser.add_argument("--server_log_path", default=None)
    parser.add_argument("--partial_predictions_path", default=None)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.98)
    parser.add_argument("--max_model_len", type=int, default=40960)
    parser.add_argument("--max_num_seqs", type=int, default=4096)
    parser.add_argument("--max_num_batched_tokens", type=int, default=524288)
    parser.add_argument("--max_lora_rank", type=int, default=None)
    parser.add_argument("--enforce_eager", action="store_true", default=True)
    parser.add_argument("--no_enforce_eager", dest="enforce_eager", action="store_false")
    add_stage_cache_args(parser)
    return parser.parse_args()


def _checkpoint_for_round(output_dir: Path, round_index: int, explicit: str | None) -> Path | None:
    if explicit:
        return input_artifact_path(explicit, output_dir / f"round_{round_index}" / "model")
    if round_index <= 0:
        return None
    return output_dir / f"round_{round_index}" / "model"


def _resolve_host_port(base_url: str) -> tuple[str, int]:
    raw = str(base_url or "").strip()
    if raw and "://" not in raw:
        raw = f"http://{raw}"
    parsed = urlparse(raw)
    return str(parsed.hostname or "localhost"), int(parsed.port or 8000)


def _server_model_names(args: argparse.Namespace) -> tuple[str, str]:
    base_name = str(args.served_model_name or Path(str(args.model_path)).name or "qwen3-0.6b")
    lora_name = str(args.lora_model_name or f"round_{int(args.round_index)}")
    return base_name, lora_name


def _adapter_weight_path(adapter_dir: Path) -> Path | None:
    for name in ("adapter_model.safetensors", "adapter_model.bin"):
        candidate = adapter_dir / name
        if candidate.exists():
            return candidate
    return None


def _same_existing_path(left: str | Path, right: str | Path) -> bool:
    left_path = Path(left)
    right_path = Path(right)
    if not left_path.is_absolute():
        left_path = input_artifact_path(left_path, PROJECT_ROOT / str(left_path))
    if not right_path.is_absolute():
        right_path = input_artifact_path(right_path, PROJECT_ROOT / str(right_path))
    if not left_path.exists() or not right_path.exists():
        return str(left).rstrip("/") == str(right).rstrip("/")
    return left_path.resolve() == right_path.resolve()


def _validate_lora_checkpoint(*, checkpoint_dir: Path, model_path: Path) -> tuple[Path, int]:
    """Validate PEFT LoRA files before starting vLLM.

    Without this preflight, a missing or mismatched adapter can make the driver
    exit during server startup; the child vLLM log then only shows a shutdown,
    hiding the actual checkpoint problem in driver stderr.
    """
    model_config_path = checkpoint_dir / "model_config.json"
    adapter_dir = checkpoint_dir / "adapter"
    adapter_config_path = adapter_dir / "adapter_config.json"
    missing = [
        str(path)
        for path in (model_config_path, adapter_dir, adapter_config_path)
        if not path.exists()
    ]
    adapter_weight_path = _adapter_weight_path(adapter_dir)
    if adapter_weight_path is None:
        missing.append(str(adapter_dir / "adapter_model.safetensors|adapter_model.bin"))
    if missing:
        raise FileNotFoundError(
            "LoRA checkpoint is incomplete; vLLM would fail to load it. "
            f"Missing: {missing}. Expected a trained round checkpoint at {checkpoint_dir}."
        )

    model_config = read_json(model_config_path)
    mode = str(model_config.get("mode", ""))
    if not mode.startswith("lora"):
        raise ValueError(f"Checkpoint {checkpoint_dir} is not a LoRA checkpoint: mode={mode!r}")

    recorded_model_path = model_config.get("model_path")
    if recorded_model_path and not _same_existing_path(str(recorded_model_path), model_path):
        raise ValueError(
            "LoRA checkpoint base model does not match the vLLM base model. "
            f"checkpoint model_path={recorded_model_path!r}, requested model_path={str(model_path)!r}."
        )

    adapter_config = read_json(adapter_config_path)
    if str(adapter_config.get("peft_type", "")).upper() != "LORA":
        raise ValueError(
            f"Adapter {adapter_config_path} is not a PEFT LoRA adapter: "
            f"peft_type={adapter_config.get('peft_type')!r}"
        )

    rank = int(adapter_config.get("r", model_config.get("lora_r", 16)) or 16)
    return adapter_dir, rank


def _start_vllm_server(args: argparse.Namespace, *, checkpoint_dir: Path | None) -> subprocess.Popen[str]:
    host, port = _resolve_host_port(args.base_url)
    model_path = input_artifact_path(args.model_path, PROJECT_ROOT / "model" / "qwen3-0.6b")
    served_model_name, lora_model_name = _server_model_names(args)
    cmd = [
        str(args.python_executable or sys.executable),
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        str(model_path),
        "--host",
        host,
        "--port",
        str(port),
        "--served-model-name",
        served_model_name,
        "--disable-uvicorn-access-log",
        "--uvicorn-log-level",
        "warning",
        "--gpu-memory-utilization",
        f"{max(0.05, min(float(args.gpu_memory_utilization), 0.98)):.3f}",
        "--max-num-seqs",
        str(max(1, int(args.max_num_seqs))),
    ]
    if int(args.max_model_len) > 0:
        cmd.extend(["--max-model-len", str(int(args.max_model_len))])
    if int(args.max_num_batched_tokens) > 0:
        cmd.extend(["--max-num-batched-tokens", str(int(args.max_num_batched_tokens))])
    if bool(args.enforce_eager):
        cmd.append("--enforce-eager")
    if checkpoint_dir is not None:
        adapter_dir, adapter_rank = _validate_lora_checkpoint(
            checkpoint_dir=checkpoint_dir,
            model_path=model_path,
        )
        max_lora_rank = max(int(args.max_lora_rank or 0), adapter_rank, 16)
        # vLLM serves the base model under `served_model_name` and each LoRA
        # adapter under the left-hand name in `--lora-modules name=path`.
        # Prediction requests must therefore use `lora_model_name`; otherwise
        # they silently hit the base model instead of the trained round adapter.
        cmd.extend(
            [
                "--enable-lora",
                "--max-lora-rank",
                str(max_lora_rank),
                "--lora-modules",
                f"{lora_model_name}={adapter_dir}",
            ]
        )

    log_path = (
        output_artifact_path(args.server_log_path, PROJECT_ROOT / "experiments" / "runs" / "vllm_openai.log")
        if args.server_log_path
        else PROJECT_ROOT / "experiments" / "runs" / "vllm_openai.log"
    )
    log_path.parent.mkdir(parents=True, exist_ok=True)
    handle = log_path.open("a", encoding="utf-8")
    env = os.environ.copy()
    env.setdefault("PYTHONNOUSERSITE", "1")
    env.setdefault("PYTHONUNBUFFERED", "1")
    env.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    try:
        proc = subprocess.Popen(
            cmd,
            cwd=str(PROJECT_ROOT),
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=handle,
            stderr=subprocess.STDOUT,
            close_fds=True,
            text=True,
            start_new_session=True,
        )
    except BaseException:
        handle.close()
        raise
    # Keep the log handle alive for the full server process lifetime.
    setattr(proc, "_server_log_handle", handle)
    return proc


def _close_server_log_handle(proc: subprocess.Popen[str]) -> None:
    handle = getattr(proc, "_server_log_handle", None)
    if handle is not None and not handle.closed:
        handle.close()


def _terminate_vllm_server(proc: subprocess.Popen[str]) -> None:
    if proc.poll() is not None:
        _close_server_log_handle(proc)
        return
    # vLLM may spawn engine/core worker children. Because the server is started
    # in a new session, signal its process group so abnormal driver exits do not
    # leave orphaned GPU worker processes behind.
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        proc.wait(timeout=30)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        proc.wait(timeout=30)
    finally:
        _close_server_log_handle(proc)


def _client(args: argparse.Namespace) -> Any:
    from openai import OpenAI
    import httpx

    max_connections = max(1, int(args.parallel_requests))

    return OpenAI(
        api_key=str(args.api_key or "EMPTY"),
        base_url=str(args.base_url),
        timeout=float(args.timeout),
        max_retries=0,
        http_client=httpx.Client(
            timeout=httpx.Timeout(float(args.timeout)),
            limits=httpx.Limits(
                max_connections=max_connections,
                max_keepalive_connections=max_connections,
            ),
        ),
    )


def _wait_for_model(client: Any, *, model_name: str, timeout: int, proc: subprocess.Popen[str] | None) -> None:
    deadline = time.time() + max(30, int(timeout))
    last_exc: Exception | None = None
    while time.time() < deadline:
        if proc is not None and proc.poll() is not None:
            raise RuntimeError(f"vLLM server exited early with code {proc.returncode}") from last_exc
        try:
            models = client.models.list()
            served = {str(item.id) for item in getattr(models, "data", []) if getattr(item, "id", None)}
            if model_name in served:
                return
        except Exception as exc:
            last_exc = exc
        time.sleep(1.0)
    raise RuntimeError(f"vLLM server did not expose model {model_name!r} before timeout") from last_exc


def _completion_prompt(example: PairExample) -> str:
    return format_binary_chat_prompt(example.query, example.document)


def _normalize_token(token: Any) -> str:
    return normalize_binary_token(token)


def _collect_binary_logprobs(choice: Any) -> tuple[float | None, float | None]:
    completion_logprobs = getattr(choice, "logprobs", None)
    completion_top_logprobs = list(getattr(completion_logprobs, "top_logprobs", []) or [])
    if completion_top_logprobs and isinstance(completion_top_logprobs[0], dict):
        one_lp: float | None = None
        zero_lp: float | None = None
        for raw_token, raw_value in completion_top_logprobs[0].items():
            token_norm = _normalize_token(raw_token)
            if raw_value is None:
                continue
            if token_norm == "one":
                one_lp = float(raw_value) if one_lp is None else max(one_lp, float(raw_value))
            elif token_norm == "zero":
                zero_lp = float(raw_value) if zero_lp is None else max(zero_lp, float(raw_value))

        generated_tokens = list(getattr(completion_logprobs, "tokens", []) or [])
        generated_lps = list(getattr(completion_logprobs, "token_logprobs", []) or [])
        if generated_tokens and generated_lps:
            generated_norm = _normalize_token(generated_tokens[0])
            generated_lp = generated_lps[0]
            if generated_lp is not None:
                if generated_norm == "one" and one_lp is None:
                    one_lp = float(generated_lp)
                if generated_norm == "zero" and zero_lp is None:
                    zero_lp = float(generated_lp)
        return one_lp, zero_lp

    content_items = list(getattr(getattr(choice, "logprobs", None), "content", []) or [])
    if not content_items:
        return None, None
    item = content_items[0]
    one_lp: float | None = None
    zero_lp: float | None = None
    for candidate in list(getattr(item, "top_logprobs", []) or []):
        token_norm = _normalize_token(getattr(candidate, "token", ""))
        value = getattr(candidate, "logprob", None)
        if value is None:
            continue
        if token_norm == "one":
            one_lp = float(value) if one_lp is None else max(one_lp, float(value))
        elif token_norm == "zero":
            zero_lp = float(value) if zero_lp is None else max(zero_lp, float(value))
    generated_norm = _normalize_token(getattr(item, "token", ""))
    generated_lp = getattr(item, "logprob", None)
    if generated_lp is not None:
        if generated_norm == "one" and one_lp is None:
            one_lp = float(generated_lp)
        if generated_norm == "zero" and zero_lp is None:
            zero_lp = float(generated_lp)
    return one_lp, zero_lp


def _predict_one(
    *,
    client: Any,
    model_name: str,
    example: PairExample,
    args: argparse.Namespace,
) -> dict[str, Any]:
    last_exc: Exception | None = None
    attempts = max(1, int(args.request_retries))
    for attempt in range(attempts):
        try:
            prompt = _completion_prompt(example)
            # Use raw completions, not chat completions. The prompt already
            # contains the exact chat markers and no-thinking block
            # used by training; sending it through chat completions would ask
            # vLLM to apply another chat template and can shift the scored
            # first classification token.
            response = client.completions.create(
                model=model_name,
                prompt=prompt,
                temperature=float(args.temperature),
                max_tokens=int(args.max_tokens),
                logprobs=int(args.top_logprobs),
            )
            choice = response.choices[0]
            content = str(getattr(choice, "text", "") or "")
            one_lp, zero_lp = _collect_binary_logprobs(choice)
            if one_lp is None and zero_lp is None:
                raise RuntimeError(
                    f"missing 1/0 logprobs for output={content!r}; "
                    "for this model family this usually means the raw prompt did not end "
                    "after the empty no-thinking block before scoring."
                )
            if one_lp is None:
                one_lp = -100.0
            if zero_lp is None:
                zero_lp = -100.0
            # `score` is the signed 1-vs-0 margin. CRC converts it to an
            # unsigned confidence during calibration.
            score = float(one_lp) - float(zero_lp)
            probability = float(1.0 / (1.0 + math.exp(-max(-60.0, min(60.0, score)))))
            prediction = 1 if score > 0.0 else 0
            row = dict(example.metadata)
            row.update(
                {
                    "id": str(example.sample_id),
                    "query": example.query,
                    "document": example.document,
                    "label": int(example.label),
                    "groundtruth": int(example.label),
                    "score": score,
                    "zero_logit": float(zero_lp),
                    "one_logit": float(one_lp),
                    "probability": probability,
                    "prediction": int(prediction),
                    "generated_text": str(int(prediction)),
                    "score_source": SCORE_SOURCE,
                    "round_index": int(args.round_index),
                    "vllm_raw_text": content,
                }
            )
            return row
        except Exception as exc:
            last_exc = exc
            if attempt + 1 >= attempts:
                print(
                    json.dumps(
                        {
                            "event": "prediction_request_failed",
                            "sample_id": str(example.sample_id),
                            "attempts": attempts,
                            "error": repr(exc),
                        },
                        ensure_ascii=False,
                        sort_keys=True,
                    ),
                    file=sys.stderr,
                    flush=True,
                )
                break
            time.sleep(min(8.0, 0.5 * (2**attempt)))
    raise RuntimeError(f"vLLM prediction failed for sample_id={example.sample_id}") from last_exc


def _progress_timeout_seconds(args: argparse.Namespace) -> float:
    if args.progress_timeout is not None:
        return max(0.0, float(args.progress_timeout))
    attempts = max(1, int(args.request_retries))
    retry_sleep = sum(min(8.0, 0.5 * (2**attempt)) for attempt in range(max(0, attempts - 1)))
    return max(300.0, float(args.timeout) * attempts + retry_sleep + 60.0)


def _predict_many(
    *,
    client: Any,
    model_name: str,
    examples: list[PairExample],
    args: argparse.Namespace,
    output_path: Path,
    partial_path: Path,
) -> list[dict[str, Any]]:
    # The partial file supports long inference resumes and is written only by
    # the main thread to keep JSONL rows from interleaving.
    existing_by_id: dict[str, dict[str, Any]] = {}
    if partial_path.exists():
        for row in read_jsonl(partial_path):
            sample_id = str(row.get("id", ""))
            if sample_id:
                existing_by_id[sample_id] = row
    pending = [example for example in examples if str(example.sample_id) not in existing_by_id]
    workers = max(1, min(int(args.parallel_requests), len(pending) if pending else 1))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    partial_path.parent.mkdir(parents=True, exist_ok=True)
    completed = len(existing_by_id)
    print(
        json.dumps(
            {
                "prediction_cache_rows": len(existing_by_id),
                "pending_rows": len(pending),
                "total": len(examples),
                "parallel_requests": workers,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    if pending:
        pending_iter = iter(pending)
        in_flight: dict[Any, tuple[str, float]] = {}
        failures: list[tuple[str, str]] = []
        progress_timeout = _progress_timeout_seconds(args)
        poll_interval = min(10.0, max(1.0, progress_timeout / 20.0)) if progress_timeout > 0 else 10.0
        last_progress_at = time.monotonic()
        pool = ThreadPoolExecutor(max_workers=workers)

        def submit_next() -> bool:
            try:
                example = next(pending_iter)
            except StopIteration:
                return False
            future = pool.submit(_predict_one, client=client, model_name=model_name, example=example, args=args)
            in_flight[future] = (str(example.sample_id), time.monotonic())
            return True

        try:
            with partial_path.open("a", encoding="utf-8") as writer:
                for _ in range(workers):
                    if not submit_next():
                        break

                while in_flight:
                    done, _ = wait(in_flight, timeout=poll_interval, return_when=FIRST_COMPLETED)
                    now = time.monotonic()
                    if not done:
                        if progress_timeout > 0 and now - last_progress_at >= progress_timeout:
                            stale = sorted(
                                (
                                    {
                                        "sample_id": sample_id,
                                        "in_flight_seconds": round(now - started_at, 1),
                                    }
                                    for sample_id, started_at in in_flight.values()
                                ),
                                key=lambda item: item["in_flight_seconds"],
                                reverse=True,
                            )[:20]
                            message = (
                                "no vLLM predictions completed before progress timeout; "
                                f"completed={completed}, total={len(examples)}, "
                                f"in_flight={len(in_flight)}, progress_timeout={progress_timeout:.1f}s"
                            )
                            print(
                                json.dumps(
                                    {
                                        "event": "prediction_progress_timeout",
                                        "completed": completed,
                                        "total": len(examples),
                                        "in_flight": len(in_flight),
                                        "progress_timeout_seconds": progress_timeout,
                                        "stale_samples": stale,
                                    },
                                    ensure_ascii=False,
                                    sort_keys=True,
                                ),
                                file=sys.stderr,
                                flush=True,
                            )
                            raise TimeoutError(message)
                        continue

                    last_progress_at = now
                    for future in done:
                        sample_id, _started_at = in_flight.pop(future)
                        try:
                            row = future.result()
                        except Exception as exc:
                            error = repr(exc)
                            failures.append((sample_id, error))
                            print(
                                json.dumps(
                                    {
                                        "event": "prediction_future_failed",
                                        "sample_id": sample_id,
                                        "error": error,
                                    },
                                    ensure_ascii=False,
                                    sort_keys=True,
                                ),
                                file=sys.stderr,
                                flush=True,
                            )
                        else:
                            sample_id = str(row["id"])
                            existing_by_id[sample_id] = row
                            writer.write(json.dumps(row, ensure_ascii=False))
                            writer.write("\n")
                            writer.flush()
                            completed += 1
                            if completed % 1000 == 0 or completed == len(examples):
                                print(
                                    json.dumps({"predicted": completed, "total": len(examples)}, sort_keys=True),
                                    flush=True,
                                )
                        submit_next()
        except BaseException:
            close = getattr(client, "close", None)
            if callable(close):
                close()
            pool.shutdown(wait=False, cancel_futures=True)
            raise
        else:
            pool.shutdown(wait=True)
        if failures:
            preview = [{"sample_id": sample_id, "error": error} for sample_id, error in failures[:20]]
            raise RuntimeError(
                "vLLM prediction failed after retries; "
                f"failures={len(failures)}, completed={completed}, total={len(examples)}, preview={preview}"
            )
        rows = [existing_by_id[str(example.sample_id)] for example in examples]
    else:
        rows = [existing_by_id[str(example.sample_id)] for example in examples]
    write_jsonl(rows, output_path)
    return rows


def _apply_teacher_label(row: dict[str, Any], teacher_labels_by_id: dict[str, dict[str, Any]]) -> None:
    sample_id = str(row["id"])
    teacher = teacher_labels_by_id.get(sample_id)
    if teacher:
        label = teacher.get("teacher_label", teacher.get("label", teacher.get("groundtruth")))
        row["teacher_label"] = label
        row["label"] = label
        row["groundtruth"] = label
        row["teacher_confidence"] = float(teacher.get("teacher_confidence", teacher.get("confidence", 1.0)) or 1.0)
        row["teacher_source"] = "teacher_api_file"
        row["teacher_label_source"] = str(teacher.get("teacher_label_source", "teacher_api_file"))
        row["teacher_confidence_source"] = str(teacher.get("teacher_confidence_source", "teacher_api_file"))
        return
    row["teacher_label"] = int(row["groundtruth"])
    row["teacher_confidence"] = 1.0
    row["teacher_source"] = "groundtruth_substitute_for_real_teacher_api"
    row["teacher_label_source"] = "groundtruth"
    row["teacher_confidence_source"] = "fixed_1.0_groundtruth_substitute"


def main() -> None:
    args = parse_args()
    output_dir = output_dir_from_arg(args.output_dir)
    round_dir = output_dir / f"round_{args.round_index}"
    all_predictions_path = output_artifact_path(args.all_predictions_path, round_dir / "all_student_predictions.jsonl")
    guide_predictions_path = output_artifact_path(
        args.guide_predictions_path,
        round_dir / "guide_student_predictions.jsonl",
    )
    final_predictions_path = output_artifact_path(
        args.final_predictions_path,
        round_dir / "final_student_predictions.jsonl",
    )
    pool_predictions_path = output_artifact_path(args.pool_predictions_path, round_dir / "pool_student_predictions.jsonl")
    partial_predictions_path = output_artifact_path(
        args.partial_predictions_path,
        round_dir / "all_student_predictions.partial.jsonl",
    )
    train_label_snapshot_path = output_artifact_path(
        args.train_label_snapshot_path,
        round_dir / "predict_train_label_snapshot.json",
    )
    usage_path = output_artifact_path(args.usage_path, round_dir / "predict_usage.json")
    if args.show_result:
        print_existing_stage_result(stage_name="predict_vllm_openai", summary_path=usage_path)
        return
    required_outputs = [
        all_predictions_path,
        guide_predictions_path,
        final_predictions_path,
        pool_predictions_path,
        train_label_snapshot_path,
        usage_path,
    ]
    try:
        cache_decision = stage_cache_decision(
            stage_name="predict_vllm_openai",
            required_outputs=required_outputs,
            cache_policy=args.cache_policy,
        )
    except RuntimeError as exc:
        if args.cache_policy == "reuse" and partial_predictions_path.exists():
            existing = [str(path) for path in required_outputs if path.exists()]
            missing = [str(path) for path in required_outputs if not path.exists()]
            cache_decision = StageCacheDecision(
                stage_name="predict_vllm_openai",
                cache_policy="reuse",
                cache_hit=False,
                action="run",
                existing_outputs=existing,
                missing_outputs=missing,
            )
            print(
                json.dumps(
                    {
                        "stage_name": "predict_vllm_openai",
                        "cache_policy": "reuse",
                        "partial_cache_resume": True,
                        "existing_outputs": existing,
                        "missing_outputs": missing,
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                ),
                flush=True,
            )
        else:
            raise
    if cache_decision.cache_hit:
        print_existing_stage_result(stage_name="predict_vllm_openai", summary_path=usage_path)
        return
    if args.cache_policy == "overwrite" and partial_predictions_path.exists():
        partial_predictions_path.unlink()

    split_payload = read_json(input_artifact_path(args.split_ids_path, split_ids_path(output_dir))) if args.split_ids_path else load_split_ids(output_dir)
    checkpoint_dir = _checkpoint_for_round(output_dir, int(args.round_index), args.checkpoint_dir)
    served_model_name, lora_model_name = _server_model_names(args)
    request_model_name = served_model_name if checkpoint_dir is None else lora_model_name
    client = _client(args)
    proc = _start_vllm_server(args, checkpoint_dir=checkpoint_dir) if bool(args.start_server) else None
    try:
        _wait_for_model(client, model_name=request_model_name, timeout=int(args.timeout), proc=proc)

        examples = load_stage_examples(
            data_path=args.data_path,
            query_field=args.query_field,
            document_field=args.document_field,
            label_field=args.label_field,
        )
        guide_examples, pool_examples = split_examples(examples, split_payload)
        examples_by_id = {str(example.sample_id): example for example in examples}
        final_ids = [str(sample_id) for sample_id in split_payload["final_ids"]]
        final_examples = [examples_by_id[sample_id] for sample_id in final_ids]
        prediction_examples = guide_examples + final_examples + pool_examples

        teacher_labels_by_id = (
            load_teacher_labels(args.teacher_labels_path, teacher_temperature=float(args.teacher_temperature))
            if args.teacher_labels_path
            else {}
        )
        selected_train_rows = (
            read_jsonl(input_artifact_path(args.selected_train_rows_path, selected_train_rows_path(output_dir)))
            if args.selected_train_rows_path
            else load_selected_train_rows(output_dir)
        )
        write_json(train_label_snapshot(selected_train_rows), train_label_snapshot_path)

        predictions = _predict_many(
            client=client,
            model_name=request_model_name,
            examples=prediction_examples,
            args=args,
            output_path=all_predictions_path,
            partial_path=partial_predictions_path,
        )
        for row in predictions:
            _apply_teacher_label(row, teacher_labels_by_id)
        write_jsonl(predictions, all_predictions_path)
        by_id = {str(row["id"]): row for row in predictions}
        guide_ids = split_payload["guide_ids"]
        guide_predictions = [by_id[str(sample_id)] for sample_id in guide_ids]
        final_predictions = [by_id[str(sample_id)] for sample_id in final_ids]
        pool_predictions = [by_id[str(sample_id)] for sample_id in split_payload["pool_ids"]]
        write_jsonl(guide_predictions, guide_predictions_path)
        write_jsonl(final_predictions, final_predictions_path)
        write_jsonl(pool_predictions, pool_predictions_path)

        teacher_usage = summarize_teacher_label_usage(predictions, purpose="predict_teacher_label_attachment")
        write_stage_usage(
            usage_path,
            {
                "stage_name": "predict_vllm_openai",
                "round_index": int(args.round_index),
                "cache": cache_decision.to_dict(),
                "student_model_calls": len(predictions),
                "student_model_role": "base_model" if checkpoint_dir is None else "round_lora_adapter",
                "vllm_base_url": str(args.base_url),
                "vllm_model": request_model_name,
                "parallel_requests": int(args.parallel_requests),
                "partial_predictions_path": str(partial_predictions_path),
                "estimated_student_prompt_tokens": estimate_query_document_prompt_tokens(predictions),
                "estimated_student_completion_tokens": len(predictions),
                "teacher_label_usage": teacher_usage,
                "groundtruth_substitute_calls": teacher_usage["groundtruth_substitute_calls"],
                "teacher_api_file_calls": teacher_usage["teacher_api_file_calls"],
                "all_predictions_path": str(all_predictions_path),
                "guide_predictions_path": str(guide_predictions_path),
                "final_predictions_path": str(final_predictions_path),
                "pool_predictions_path": str(pool_predictions_path),
            },
        )
        print(json.dumps({"round_index": args.round_index, "predicted": len(predictions)}, ensure_ascii=False, sort_keys=True))
    finally:
        close = getattr(client, "close", None)
        if callable(close):
            close()
        if proc is not None:
            # If the driver raises from any worker future, this finally block
            # intentionally stops the child vLLM server. In that case the
            # server log will show "Shutdown initiated"; the driver stdout/stderr
            # log is the place to inspect for the real Python exception.
            _terminate_vllm_server(proc)


if __name__ == "__main__":
    main()
