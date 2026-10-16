#!/usr/bin/env python3
"""Measure the C500 prefill prototype against the MetaX vendor path.

This script is intentionally an experiment tool.  It does not change the
runtime dispatcher and does not claim that the Python prototype is a kernel.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import platform
import statistics
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import torch

from mxflashattn.prefill import prefill_chunked_reference


def _version(module_name: str) -> str | None:
    try:
        module = __import__(module_name)
        return str(getattr(module, "__version__", "unknown"))
    except Exception:
        return None


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _timed(fn, *, warmup: int, iters: int) -> list[float]:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    values: list[float] = []
    for _ in range(iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        end.synchronize()
        values.append(float(start.elapsed_time(end)))
    return values


def _stats(values: list[float]) -> dict[str, float]:
    return {
        "median_ms": statistics.median(values),
        "p95_ms": sorted(values)[max(0, math.ceil(len(values) * 0.95) - 1)],
        "min_ms": min(values),
        "max_ms": max(values),
    }


def _vendor_prefill(q, k, v, *, causal: bool):
    from vllm_metax.v1.attention.backends.fa_utils import flash_attn_func

    return flash_attn_func(q=q, k=k, v=v, causal=causal)


def _case(batch: int, seq: int, heads: int, kv_heads: int, dim: int, dtype):
    torch.manual_seed(20261017 + batch + seq)
    q = torch.randn(batch, seq, heads, dim, device="cuda", dtype=dtype)
    k = torch.randn(batch, seq, kv_heads, dim, device="cuda", dtype=dtype)
    v = torch.randn_like(k)
    return q, k, v


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--chunk-size", type=int, default=256)
    parser.add_argument(
        "--cases",
        default="b1-s256,b1-s512,b1-s1024,b4-s512,b16-s1024",
        help="comma-separated cases",
    )
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA/MACA device is required")

    cases = {
        "b1-s256": (1, 256),
        "b1-s512": (1, 512),
        "b1-s1024": (1, 1024),
        "b4-s512": (4, 512),
        "b16-s1024": (16, 1024),
    }
    dtype = torch.float16
    rows = []
    for case_id in args.cases.split(","):
        if case_id not in cases:
            raise SystemExit(f"unknown case: {case_id}")
        batch, seq = cases[case_id]
        q, k, v = _case(batch, seq, 16, 4, 128, dtype)
        causal = True
        vendor = lambda: _vendor_prefill(q, k, v, causal=causal)
        prototype = lambda: prefill_chunked_reference(
            q, k, v, chunk_size=args.chunk_size, causal=causal
        )

        torch.cuda.reset_peak_memory_stats()
        reference_out = vendor()
        torch.cuda.synchronize()
        prototype_out = prototype()
        torch.cuda.synchronize()
        diff = (prototype_out.float() - reference_out.float()).abs()
        vendor_times = _timed(vendor, warmup=args.warmup, iters=args.iters)
        prototype_times = _timed(prototype, warmup=args.warmup, iters=args.iters)
        rows.append(
            {
                "case_id": case_id,
                "batch_size": batch,
                "sequence_length": seq,
                "dtype": str(dtype),
                "causal": causal,
                "chunk_size": args.chunk_size,
                "vendor": _stats(vendor_times),
                "prototype": _stats(prototype_times),
                "prototype_over_vendor_median": (
                    statistics.median(prototype_times) / statistics.median(vendor_times)
                ),
                "max_abs_error": float(diff.max().item()),
                "p99_abs_error": float(torch.quantile(diff.flatten(), 0.99).item()),
                "peak_memory_gib": torch.cuda.max_memory_allocated() / (1024**3),
            }
        )
        del q, k, v, reference_out, prototype_out
        torch.cuda.empty_cache()

    result = {
        "schema_version": 1,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "command": " ".join(os.sys.argv),
        "seed": 20261017,
        "warmup_count": args.warmup,
        "measurement_count": args.iters,
        "host": platform.node(),
        "device": torch.cuda.get_device_name(0),
        "device_memory_gib": torch.cuda.get_device_properties(0).total_memory / (1024**3),
        "torch_version": torch.__version__,
        "vllm_version": _version("vllm"),
        "rows": rows,
        "notes": [
            "prototype is correctness-first PyTorch code, not a production C500 kernel",
            "results compare vendor prefill with the prototype only",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
