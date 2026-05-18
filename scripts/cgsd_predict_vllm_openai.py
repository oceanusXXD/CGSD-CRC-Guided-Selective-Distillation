#!/usr/bin/env python
"""通过 vLLM OpenAI-compatible server 执行 CGSD student 预测。

该脚本负责高吞吐全量推理：对 D_guide、D_cert 和 pool 中的每个样本
构造 query-document prompt，只生成 1 个 token，并读取首 token 位置
`1` 与 `0` 的 logprob。保存的 `score = one_lp - zero_lp` 是后续
CRC 路由分数 `sigmoid(abs(score)/T)` 的原始 margin。

vLLM 被放在独立 server 进程中，避免主预测进程直接 import vLLM CUDA
扩展；`--start_server` 启动的 server 会在脚本结束时清理。
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
from threading import Lock
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.cgsd_cli_common import (
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
    runtime_args_from_cli,
    split_examples,
    stage_cache_decision,
    summarize_teacher_label_usage,
    train_label_snapshot,
    write_stage_usage,
)
from src.binary_protocol import BINARY_SCORE_SOURCE, BINARY_SYSTEM_PROMPT, binary_user_prompt, normalize_binary_token
from src.data import PairExample
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
    parser.add_argument("--calibration_predictions_path", default=None)
    parser.add_argument("--final_calibration_predictions_path", default=None)
    parser.add_argument("--pool_predictions_path", default=None)
    parser.add_argument("--train_label_snapshot_path", default=None)
    parser.add_argument("--usage_path", default=None)

    parser.add_argument("--base_url", default="http://127.0.0.1:18021/v1")
    parser.add_argument("--api_key", default="EMPTY")
    parser.add_argument("--served_model_name", default=None)
    parser.add_argument("--lora_model_name", default=None)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--parallel_requests", type=int, default=1024)
    parser.add_argument("--request_retries", type=int, default=3)
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
    return str(parsed.hostname or "127.0.0.1"), int(parsed.port or 8000)


def _server_model_names(args: argparse.Namespace) -> tuple[str, str]:
    base_name = str(args.served_model_name or Path(str(args.model_path)).name or "qwen3-0.6b")
    lora_name = str(args.lora_model_name or f"cgsd_round_{int(args.round_index)}")
    return base_name, lora_name


def _start_vllm_server(args: argparse.Namespace, *, checkpoint_dir: Path | None) -> subprocess.Popen[str]:
    host, port = _resolve_host_port(args.base_url)
    model_path = str(input_artifact_path(args.model_path, PROJECT_ROOT / "model" / "qwen3-0.6b"))
    served_model_name, lora_model_name = _server_model_names(args)
    cmd = [
        str(args.python_executable or sys.executable),
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        model_path,
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
        adapter_dir = checkpoint_dir / "adapter"
        cmd.extend(["--enable-lora", "--lora-modules", f"{lora_model_name}={adapter_dir}"])

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
    # 把日志句柄挂在 Popen 上，保证 server 生命周期内不会被提前关闭。
    setattr(proc, "_cgsd_log_handle", handle)
    return proc


def _client(args: argparse.Namespace) -> Any:
    from openai import OpenAI

    return OpenAI(
        api_key=str(args.api_key or "EMPTY"),
        base_url=str(args.base_url),
        timeout=int(args.timeout),
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


def _messages(example: PairExample) -> list[dict[str, str]]:
    """构造 Qwen chat prompt。"""
    return [
        {
            "role": "system",
            "content": BINARY_SYSTEM_PROMPT,
        },
        {
            "role": "user",
            "content": binary_user_prompt(example.query, example.document),
        },
    ]


def _normalize_token(token: Any) -> str:
    return normalize_binary_token(token)


def _collect_binary_logprobs(choice: Any) -> tuple[float | None, float | None]:
    """从首个生成 token 的 top_logprobs 中提取 1/0 logprob。"""
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
    for attempt in range(max(1, int(args.request_retries))):
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=_messages(example),
                temperature=float(args.temperature),
                max_tokens=int(args.max_tokens),
                logprobs=True,
                top_logprobs=int(args.top_logprobs),
                extra_body={
                    "enable_thinking": False,
                    "chat_template_kwargs": {"enable_thinking": False},
                },
            )
            choice = response.choices[0]
            content = str(choice.message.content or "")
            one_lp, zero_lp = _collect_binary_logprobs(choice)
            if one_lp is None and zero_lp is None:
                raise RuntimeError(f"missing 1/0 logprobs for output={content!r}")
            if one_lp is None:
                one_lp = -100.0
            if zero_lp is None:
                zero_lp = -100.0
            # score 是有方向的 1-vs-0 margin；CRC 的无方向确信度
            # 在 calibrate 阶段再统一转成 sigmoid(abs(score)/T)。
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
            time.sleep(min(8.0, 0.5 * (2**attempt)))
    raise RuntimeError(f"vLLM prediction failed for sample_id={example.sample_id}") from last_exc


def _predict_many(
    *,
    client: Any,
    model_name: str,
    examples: list[PairExample],
    args: argparse.Namespace,
    output_path: Path,
    partial_path: Path,
) -> list[dict[str, Any]]:
    # partial 文件用于长时间 vLLM 全量推理的断点续跑。多线程写入时
    # 必须用锁，避免高并发下 JSONL 行交错导致文件损坏。
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
        write_lock = Lock()
        with partial_path.open("a", encoding="utf-8") as writer, ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(_predict_one, client=client, model_name=model_name, example=example, args=args): str(example.sample_id)
                for example in pending
            }
            for future in as_completed(futures):
                row = future.result()
                sample_id = str(row["id"])
                existing_by_id[sample_id] = row
                with write_lock:
                    writer.write(json.dumps(row, ensure_ascii=False))
                    writer.write("\n")
                    writer.flush()
                    completed += 1
                    if completed % 1000 == 0:
                        print(json.dumps({"predicted": completed, "total": len(examples)}, sort_keys=True), flush=True)
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
    calibration_predictions_path = output_artifact_path(
        args.calibration_predictions_path,
        round_dir / "calibration_student_predictions.jsonl",
    )
    final_calibration_predictions_path = output_artifact_path(
        args.final_calibration_predictions_path,
        round_dir / "final_calibration_student_predictions.jsonl",
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
        print_existing_stage_result(stage_name="cgsd_predict_vllm_openai", summary_path=usage_path)
        return
    cache_decision = stage_cache_decision(
        stage_name="cgsd_predict_vllm_openai",
        required_outputs=[
            all_predictions_path,
            calibration_predictions_path,
            final_calibration_predictions_path,
            pool_predictions_path,
            train_label_snapshot_path,
            usage_path,
        ],
        cache_policy=args.cache_policy,
    )
    if cache_decision.cache_hit:
        print_existing_stage_result(stage_name="cgsd_predict_vllm_openai", summary_path=usage_path)
        return
    if args.cache_policy == "overwrite" and partial_predictions_path.exists():
        partial_predictions_path.unlink()

    split_payload = read_json(input_artifact_path(args.split_ids_path, output_dir / "cgsd_split_ids.json")) if args.split_ids_path else load_split_ids(output_dir)
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
        calibration_examples, pool_examples = split_examples(examples, split_payload)
        examples_by_id = {str(example.sample_id): example for example in examples}
        final_calibration_ids = [str(sample_id) for sample_id in split_payload.get("final_calibration_ids", [])]
        final_calibration_examples = [examples_by_id[sample_id] for sample_id in final_calibration_ids]
        prediction_examples = calibration_examples + final_calibration_examples + pool_examples

        teacher_labels_by_id = (
            load_teacher_labels(args.teacher_labels_path, teacher_temperature=float(args.teacher_temperature))
            if args.teacher_labels_path
            else {}
        )
        selected_train_rows = (
            read_jsonl(input_artifact_path(args.selected_train_rows_path, output_dir / "cgsd_train_rows.jsonl"))
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
        calibration_predictions = [by_id[str(sample_id)] for sample_id in split_payload["calibration_ids"]]
        final_calibration_predictions = [by_id[str(sample_id)] for sample_id in final_calibration_ids]
        pool_predictions = [by_id[str(sample_id)] for sample_id in split_payload["pool_ids"]]
        write_jsonl(calibration_predictions, calibration_predictions_path)
        write_jsonl(final_calibration_predictions, final_calibration_predictions_path)
        write_jsonl(pool_predictions, pool_predictions_path)

        teacher_usage = summarize_teacher_label_usage(predictions, purpose="predict_teacher_label_attachment")
        write_stage_usage(
            usage_path,
            {
                "stage_name": "cgsd_predict_vllm_openai",
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
                "calibration_predictions_path": str(calibration_predictions_path),
                "final_calibration_predictions_path": str(final_calibration_predictions_path),
                "pool_predictions_path": str(pool_predictions_path),
            },
        )
        print(json.dumps({"round_index": args.round_index, "predicted": len(predictions)}, ensure_ascii=False, sort_keys=True))
    finally:
        if proc is not None and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=30)


if __name__ == "__main__":
    main()
