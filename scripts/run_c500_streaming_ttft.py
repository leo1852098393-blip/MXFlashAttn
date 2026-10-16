"""Measure streaming TTFT/P95 and decoder dispatch on the pinned C500 vLLM."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import statistics
import subprocess
import time
from pathlib import Path
from typing import Any


PROMPTS = [
    "Explain paged KV cache in one sentence.",
    "List three practical benefits of grouped-query attention.",
    "Give a concise checklist for validating an attention kernel.",
    "Describe how causal masking changes autoregressive decoding.",
]


def _mx_smi_used_bytes() -> int | None:
    try:
        output = subprocess.run(
            ["mx-smi", "--show-memory"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        ).stdout
    except (OSError, subprocess.TimeoutExpired):
        return None
    values = []
    for line in output.splitlines():
        if "vram used" in line.lower():
            parts = line.split(":")
            if len(parts) == 2:
                digits = "".join(ch for ch in parts[1] if ch.isdigit())
                if digits:
                    values.append(int(digits) * 1024)
    return values[0] if values else None


def _accelerator():
    import torch

    for name in ("maca", "cuda", "musa"):
        module = getattr(torch, name, None)
        if module is not None and (
            not hasattr(module, "is_available") or module.is_available()
        ):
            return module
    return None


async def _run_request(engine, prompt: str, params, request_id: str) -> dict[str, Any]:
    from vllm.sampling_params import RequestOutputKind

    arrival = time.perf_counter()
    first_token = None
    finish = None
    token_count = 0
    text = ""
    last_output = None
    async for output in engine.generate(prompt, params, request_id):
        now = time.perf_counter()
        last_output = output
        completion = output.outputs[0]
        delta_tokens = len(completion.token_ids)
        if delta_tokens and first_token is None:
            first_token = now
        token_count += delta_tokens
        text += completion.text
        if output.finished:
            finish = now
    if finish is None:
        finish = time.perf_counter()
    return {
        "prompt": prompt,
        "generated_tokens": token_count,
        "generated_text": text,
        "first_token_latency_ms": (
            (first_token - arrival) * 1000 if first_token is not None else None
        ),
        "request_finish_latency_ms": (finish - arrival) * 1000,
        "wall_latency_ms": (finish - arrival) * 1000,
        "tokens_per_second": token_count / (finish - arrival),
        "finished": bool(last_output and last_output.finished),
    }


async def _run(args: argparse.Namespace) -> dict[str, Any]:
    dispatch_module = None
    registration = None
    if args.candidate:
        from integrations.vllm import backend as dispatch_module

        registration = dispatch_module.register_vllm_v1_backend()
        if registration.get("registered") != "true":
            raise RuntimeError(f"MXFlashAttn registration failed: {registration}")
        os.environ["MXFLASHATTN_AUTOREGISTER"] = "1"
        os.environ["MXFA_NATIVE_DECODE"] = "1"
        os.environ["MXFLASHATTN_GRAPH_OFFICIAL_SPLIT"] = "1"
        os.environ.pop("MXFLASHATTN_DISABLE_DISPATCH_LOG", None)
        os.environ["MXFLASHATTN_DISPATCH_SAMPLE"] = str(args.dispatch_sample)
        os.environ["MXFLASHATTN_DISPATCH_LOG"] = str(
            args.output.with_suffix(".dispatch.jsonl").resolve()
        )
        os.environ["MXFLASHATTN_PLUGIN_IMPORT_LOG"] = str(
            args.output.with_suffix(".plugin.log").resolve()
        )
        # Keep the V1 engine core in-process. The C500 adapter registration and
        # the M1 environment switches must be visible to the attention worker.
        os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"

    import torch
    from vllm import AsyncLLMEngine
    from vllm.engine.arg_utils import AsyncEngineArgs
    from vllm.sampling_params import RequestOutputKind, SamplingParams

    engine_args = AsyncEngineArgs(
        model=args.model,
        trust_remote_code=True,
        max_model_len=args.max_model_len,
        max_num_seqs=args.batch_size,
        enforce_eager=True,
        dtype=args.dtype,
    )
    engine = AsyncLLMEngine.from_engine_args(engine_args)
    params = SamplingParams(
        temperature=0.0,
        max_tokens=args.max_tokens,
        output_kind=RequestOutputKind.DELTA,
    )
    prompts = [
        f"{PROMPTS[i % len(PROMPTS)]} Case {i:02d}."
        for i in range(args.request_count)
    ]
    records = []
    try:
        if dispatch_module is not None:
            dispatch_module.clear_dispatch_events(truncate_log=True)
        accelerator = _accelerator()
        if accelerator is not None and hasattr(accelerator, "reset_peak_memory_stats"):
            accelerator.reset_peak_memory_stats()
        # Submit all requests together so vLLM actually forms a continuous
        # batch. Sequential requests would exercise the batch=1 guard even
        # when max_num_seqs is configured to a larger value.
        records = list(
            await asyncio.gather(
                *(
                    _run_request(engine, prompt, params, f"stream-{index:03d}")
                    for index, prompt in enumerate(prompts)
                )
            )
        )
        if accelerator is not None and hasattr(accelerator, "max_memory_reserved"):
            peak_reserved = accelerator.max_memory_reserved()
        else:
            peak_reserved = None
        for record in records:
            record["peak_memory_reserved_bytes"] = peak_reserved
            record["mx_smi_vram_used_bytes"] = _mx_smi_used_bytes()
    finally:
        engine.shutdown()

    dispatch_events = []
    dispatch_log = args.output.with_suffix(".dispatch.jsonl")
    if dispatch_log.exists():
        dispatch_events = [
            json.loads(line)
            for line in dispatch_log.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    ttft = [r["first_token_latency_ms"] for r in records if r["first_token_latency_ms"] is not None]
    finish = [r["request_finish_latency_ms"] for r in records]
    throughput = [r["tokens_per_second"] for r in records]

    def percentile(values: list[float], p: float) -> float | None:
        if not values:
            return None
        ordered = sorted(values)
        rank = (len(ordered) - 1) * p
        lower = int(rank)
        upper = min(lower + 1, len(ordered) - 1)
        fraction = rank - lower
        return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction

    result = {
        "schema_version": 1,
        "kind": "c500_streaming_ttft",
        "mode": "candidate" if args.candidate else "baseline",
        "model": args.model,
        "vllm": __import__("importlib.metadata").metadata.version("vllm"),
        "torch": torch.__version__,
        "device": (
            _accelerator().get_device_name()
            if _accelerator() is not None and hasattr(_accelerator(), "get_device_name")
            else "unknown"
        ),
        "batch_size": args.batch_size,
        "max_model_len": args.max_model_len,
        "max_tokens": args.max_tokens,
        "request_count": args.request_count,
        "registration": registration,
        "dispatch_event_count": len(dispatch_events),
        "dispatch_paths": sorted({event.get("path") for event in dispatch_events}),
        "records": records,
        "summary": {
            "ttft_median_ms": statistics.median(ttft) if ttft else None,
            "ttft_p95_ms": percentile(ttft, 0.95),
            "finish_latency_median_ms": statistics.median(finish) if finish else None,
            "finish_latency_p95_ms": percentile(finish, 0.95),
            "tokens_per_second_median": statistics.median(throughput) if throughput else None,
            "all_requests_finished": all(r["finished"] for r in records),
            "all_requests_have_first_token": len(ttft) == len(records),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(result["summary"], ensure_ascii=False))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--candidate", action="store_true")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-model-len", type=int, default=2048)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--request-count", type=int, default=4)
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--dispatch-sample", type=int, default=2000)
    args = parser.parse_args()
    # vLLM reads this before importing its V1 engine modules. Keeping the
    # engine core in-process is required for the local C500 adapter registration
    # and matches the project's existing E2E harness.
    os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    asyncio.run(_run(args))


if __name__ == "__main__":
    main()
