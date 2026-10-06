"""Preflight the pinned vLLM environment before adding the C500 platform adapter."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import re
import subprocess
import time
from pathlib import Path

import torch

from integrations.vllm.config import QUICK_MODEL, validate_vllm_version


def _accelerator_module():
    for name in ("maca", "cuda", "musa"):
        module = getattr(torch, name, None)
        if module is None:
            continue
        available = getattr(module, "is_available", None)
        if available is None or available():
            return module
    return None


def _mx_smi_used_bytes() -> int | None:
    try:
        output = subprocess.run(
            ["mx-smi", "--show-memory"], capture_output=True, text=True, timeout=5, check=False
        ).stdout
    except (OSError, subprocess.TimeoutExpired):
        return None
    values = re.findall(r"vram used\s*:\s*(\d+)\s*KB", output, flags=re.IGNORECASE)
    return int(values[0]) * 1024 if values else None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=QUICK_MODEL)
    parser.add_argument("--prompt", default="Explain paged KV cache in one sentence.")
    parser.add_argument(
        "--suite",
        action="store_true",
        help="Run a fixed set of prompts in one loaded model instance.",
    )
    parser.add_argument("--prompt-count", type=int, default=4, help="Number of deterministic suite prompts.")
    parser.add_argument("--output", type=Path, default=Path("artifacts/vllm-baseline.json"))
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--baseline-only", action="store_true")
    parser.add_argument("--candidate", action="store_true", help="Register and exercise MXFlashAttn decoder dispatch.")
    args = parser.parse_args()
    if args.baseline_only and args.candidate:
        raise SystemExit("choose only one of --baseline-only or --candidate")
    if not args.baseline_only and not args.candidate:
        raise SystemExit(
            "choose --baseline-only for vendor baseline or --candidate for MXFlashAttn dispatch"
        )
    try:
        version = importlib.metadata.version("vllm")
    except importlib.metadata.PackageNotFoundError as error:
        raise SystemExit("vLLM is not installed in this environment") from error
    validate_vllm_version(version)
    dispatch_module = None
    registration = None
    if args.candidate:
        from integrations.vllm import backend as dispatch_module

        registration = dispatch_module.register_vllm_v1_backend()
        if registration.get("registered") != "true":
            raise SystemExit(f"MXFlashAttn backend registration failed: {registration}")
        dispatch_log = args.output.with_suffix(".dispatch.jsonl").resolve()
        # Event recording costs measurable time per attention call, and the
        # vendor baseline pays nothing equivalent. MXFLASHATTN_DISABLE_DISPATCH_LOG
        # turns the file sink off (in-memory events still drive the verdict) so
        # throughput can be compared without instrumentation overhead.
        os.environ["MXFLASHATTN_DISPATCH_LOG"] = str(dispatch_log)
        if os.getenv("MXFLASHATTN_DISABLE_DISPATCH_LOG", "0") == "1":
            # Keep the log path so the verdict still has evidence, but write
            # only a sample: instrumentation cost must not distort throughput.
            os.environ["MXFLASHATTN_DISPATCH_SAMPLE"] = os.getenv(
                "MXFLASHATTN_DISPATCH_SAMPLE", "2000"
            )
        os.environ["MXFLASHATTN_AUTOREGISTER"] = "1"
        os.environ["MXFLASHATTN_REGISTRATION_ERROR_LOG"] = str(args.output.with_suffix(".registration-error.log").resolve())
        os.environ["MXFLASHATTN_PLUGIN_IMPORT_LOG"] = str(args.output.with_suffix(".plugin.log").resolve())
    else:
        dispatch_log = None
    from vllm import LLM, SamplingParams

    _eager = os.getenv("MXFLASHATTN_EAGER", "1") != "0"
    engine = LLM(model=args.model, enforce_eager=_eager, max_model_len=2048)
    prompt_templates = [
        "Explain paged KV cache in one sentence.",
        "List three practical benefits of grouped-query attention.",
        "Give a concise checklist for validating an attention kernel.",
        "Describe how causal masking changes autoregressive decoding.",
        "Explain why variable-length attention needs cumulative offsets.",
        "Compare MQA and GQA for inference memory usage.",
        "Name two checks for a causal attention implementation.",
        "Explain the role of a block table in paged attention.",
        "Give one reason BF16 may differ slightly from FP16.",
        "Describe a safe fallback policy for an accelerator kernel.",
        "Explain why a vendor baseline must be measured separately.",
        "List the metadata needed by a decoder attention adapter.",
    ]
    prompts = [f"{prompt_templates[i % len(prompt_templates)]} Case {i:02d}." for i in range(args.prompt_count)] if args.suite else [args.prompt]
    records = []
    dispatch_offset = 0
    for index, prompt in enumerate(prompts):
        if dispatch_module is not None:
            # Only the first prompt truncates the evidence file; later prompts
            # append so the log covers the entire suite instead of its last item.
            dispatch_module.clear_dispatch_events(truncate_log=(index == 0))
        accelerator = _accelerator_module()
        if accelerator is not None:
            reset = getattr(accelerator, "reset_peak_memory_stats", None)
            if reset is not None:
                reset()
        start = time.perf_counter()
        outputs = engine.generate(prompt, SamplingParams(temperature=0.0, max_tokens=args.max_tokens))
        elapsed = time.perf_counter() - start
        request = outputs[0]
        metrics = getattr(request, "metrics", None)
        arrival = getattr(metrics, "arrival_time", None)
        first_token = getattr(metrics, "first_token_time", None)
        finish = getattr(metrics, "finished_time", None)
        generated_tokens = len(request.outputs[0].token_ids)
        prompt_tokens = getattr(request, "prompt_token_ids", None)
        worker_events = []
        if dispatch_log is not None and dispatch_log.exists():
            # The log accumulates across the suite, so take only the lines this
            # prompt appended instead of re-attributing earlier prompts' events.
            lines = [line for line in dispatch_log.read_text(encoding="utf-8").splitlines() if line]
            worker_events = [json.loads(line) for line in lines[dispatch_offset:]]
            dispatch_offset = len(lines)
        record = {
            "index": index,
            "prompt": prompt,
            "prompt_tokens": len(prompt_tokens) if prompt_tokens is not None else None,
            "generated_tokens": generated_tokens,
            "generated_text": request.outputs[0].text,
            "wall_latency_ms": elapsed * 1000,
            "first_token_latency_ms": (first_token - arrival) * 1000 if first_token is not None and arrival is not None else None,
            "request_finish_latency_ms": (finish - arrival) * 1000 if finish is not None and arrival is not None else None,
            "tokens_per_second": generated_tokens / elapsed if elapsed > 0 else None,
            "peak_memory_allocated_bytes": accelerator.max_memory_allocated() if accelerator is not None and hasattr(accelerator, "max_memory_allocated") else None,
            "peak_memory_reserved_bytes": accelerator.max_memory_reserved() if accelerator is not None and hasattr(accelerator, "max_memory_reserved") else None,
            "mx_smi_vram_used_bytes": _mx_smi_used_bytes(),
            "dispatch_events": worker_events if dispatch_module is not None else [],
        }
        records.append(record)
    accelerator = _accelerator_module()
    device = accelerator.get_device_name() if accelerator is not None and hasattr(accelerator, "get_device_name") else "default vLLM platform"
    # v1.1 no longer emits the v0.8 label "mxflashattn_decoder". It records the
    # backend that actually served each decode call: "vendor_direct" for the fast
    # path and "mxmac_aten_correctness" for the validated fallback. Both come
    # only from the MXFlashAttn decode branch, so either proves the adapter ran;
    # a non-fallback vendor_direct additionally proves it used the vendor kernel.
    candidate_dispatch_verified = any(
        event.get("path") in {"vendor_direct", "mxmac_aten_correctness"}
        for record in records
        for event in record.get("dispatch_events", [])
    )
    result = {
        "schema_version": 2,
        "kind": "vllm_mxflashattn_candidate" if args.candidate else "vllm_baseline_only",
        "model": args.model,
        "vllm": version,
        "torch": torch.__version__,
        "device": device,
        "max_tokens": args.max_tokens,
        "suite": args.suite,
        "registration": registration,
        "candidate_requires_mxflashattn_decode": args.candidate,
        "candidate_dispatch_verified": candidate_dispatch_verified,
        "status": ("candidate_dispatch_verified" if candidate_dispatch_verified else "candidate_not_verified") if args.candidate else "baseline_only",
        "runs": records,
        "warning": "Prefill may delegate to vendor; candidate records decoder dispatch events." if args.candidate else "This baseline-only run does not exercise MXFlashAttn.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Wrote vLLM baseline record to {args.output}")


if __name__ == "__main__":
    main()
