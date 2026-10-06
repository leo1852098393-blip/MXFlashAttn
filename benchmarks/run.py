from __future__ import annotations

import argparse
import importlib.metadata
import itertools
import json
import os
import re
import statistics
import subprocess
import time
from pathlib import Path

import torch
import yaml

from mxflashattn import flash_attn_func, flash_attn_varlen_func, flash_attn_with_kvcache, get_last_dispatch_info
from mxflashattn import reference


def expand_cases(config: dict) -> list[dict]:
    matrix = config["matrix"]
    keys = list(matrix)
    return [
        dict(zip(keys, values))
        for values in itertools.product(*(matrix[key] for key in keys))
    ]


def _case_id(case: dict) -> str:
    return "_".join(f"{key}-{value}" for key, value in case.items())


def _sync(device: torch.device) -> None:
    module_names = {"maca": "maca", "musa": "musa", "cuda": "cuda"}
    module = getattr(torch, module_names.get(device.type, ""), None)
    synchronize = getattr(module, "synchronize", None)
    if synchronize is not None:
        synchronize(device)


def _peak_memory(device: torch.device) -> int | None:
    module = getattr(torch, {"maca": "maca", "musa": "musa", "cuda": "cuda"}.get(device.type, ""), None)
    getter = getattr(module, "max_memory_allocated", None)
    if getter is None:
        return None
    try:
        return int(getter(device))
    except (RuntimeError, TypeError):
        return None


def _reset_peak_memory(device: torch.device) -> None:
    module = getattr(torch, {"maca": "maca", "musa": "musa", "cuda": "cuda"}.get(device.type, ""), None)
    reset = getattr(module, "reset_peak_memory_stats", None)
    if reset is not None:
        try:
            reset(device)
        except (RuntimeError, TypeError):
            pass


def _make_inputs(case: dict, device: torch.device):
    dtype = getattr(torch, case["dtype"])
    batch = int(case["batch_size"])
    query_len = int(case["q_len"])
    kv_len = int(case["kv_len"])
    head_dim = int(case["head_dim"])
    kv_heads = 4
    query_heads = kv_heads * int(case["gqa_ratio"])
    q = torch.randn(batch, query_len, query_heads, head_dim, device=device, dtype=dtype)
    k = torch.randn(batch, kv_len, kv_heads, head_dim, device=device, dtype=dtype)
    v = torch.randn_like(k)
    return q, k, v


def _prepare_candidate(case: dict, q, k, v):
    layout = case["layout"]
    causal = bool(case.get("causal", True))
    if layout == "dense":
        return lambda: flash_attn_func(q, k, v, causal=causal)
    if layout == "varlen":
        batch, q_len = q.shape[:2]
        kv_len = k.shape[1]
        cu_q = torch.arange(0, (batch + 1) * q_len, q_len, dtype=torch.int32, device=q.device)
        cu_k = torch.arange(0, (batch + 1) * kv_len, kv_len, dtype=torch.int32, device=q.device)
        flat_q, flat_k, flat_v = (
            q.reshape(-1, *q.shape[2:]),
            k.reshape(-1, *k.shape[2:]),
            v.reshape(-1, *v.shape[2:]),
        )
        return lambda: flash_attn_varlen_func(
            flat_q,
            flat_k,
            flat_v,
            cu_q,
            cu_k,
            max_seqlen_q=q_len,
            max_seqlen_k=kv_len,
            causal=causal,
        )
    if layout == "paged":
        batch, kv_len = k.shape[:2]
        page_size = int(case.get("page_size", 16))
        pages_per_batch = (kv_len + page_size - 1) // page_size
        cache_k = torch.zeros(batch * pages_per_batch, page_size, *k.shape[2:], dtype=k.dtype, device=k.device)
        cache_v = torch.zeros_like(cache_k)
        table = torch.arange(batch * pages_per_batch, dtype=torch.int32, device=k.device).reshape(batch, pages_per_batch)
        for batch_index in range(batch):
            start = batch_index * pages_per_batch
            flat_k = k[batch_index]
            flat_v = v[batch_index]
            for page_index in range(pages_per_batch):
                source_start = page_index * page_size
                count = min(page_size, kv_len - source_start)
                cache_k[start + page_index, :count] = flat_k[source_start : source_start + count]
                cache_v[start + page_index, :count] = flat_v[source_start : source_start + count]
        lengths = torch.full((batch,), kv_len, dtype=torch.int32, device=k.device)
        return lambda: flash_attn_with_kvcache(
            q,
            cache_k,
            cache_v,
            cache_seqlens=lengths,
            block_table=table,
            causal=causal,
        )
    raise ValueError(f"unknown layout: {layout}")


def _reference(case: dict, q, k, v):
    causal = bool(case.get("causal", True))
    if case["layout"] in ("dense", "paged"):
        return reference.attention_batched(q, k, v, causal=causal)
    batch, q_len = q.shape[:2]
    kv_len = k.shape[1]
    cu_q = [index * q_len for index in range(batch + 1)]
    cu_k = [index * kv_len for index in range(batch + 1)]
    return reference.attention_varlen(q.reshape(-1, *q.shape[2:]), k.reshape(-1, *k.shape[2:]), v.reshape(-1, *v.shape[2:]), cu_q, cu_k, causal=causal)


def _measure(fn, warmup: int, repeats: int, device: torch.device) -> tuple[float, object]:
    result = None
    for _ in range(warmup):
        result = fn()
    _sync(device)
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        result = fn()
        _sync(device)
        samples.append((time.perf_counter() - start) * 1000)
    return statistics.median(samples), result


def run_benchmarks(
    config: dict,
    *,
    device: torch.device,
    warmup: int,
    repeats: int,
    limit: int | None = None,
    seed: int | None = None,
) -> dict:
    seed = int(config.get("seed", 0) if seed is None else seed)
    if seed < 0:
        raise ValueError("seed must be non-negative")
    torch.manual_seed(seed)
    cases = expand_cases(config)
    if limit is not None:
        cases = cases[:limit]
    output_cases = []
    for case in cases:
        q, k, v = _make_inputs(case, device)
        _reset_peak_memory(device)
        baseline_latency, baseline_output = _measure(lambda: _reference(case, q, k, v), warmup, repeats, device)
        baseline_peak_memory = _peak_memory(device)
        baseline_output = baseline_output.detach().cpu()
        candidate = _prepare_candidate(case, q, k, v)
        _reset_peak_memory(device)
        candidate_latency, candidate_output = _measure(candidate, warmup, repeats, device)
        dispatch_info = get_last_dispatch_info()
        candidate_output = candidate_output.detach().cpu()
        difference = (candidate_output.float() - baseline_output.float()).abs()
        max_abs_error = float(difference.max().item()) if difference.numel() else 0.0
        denom = baseline_output.float().abs().clamp_min(1e-6)
        max_relative_error = float((difference / denom).max().item()) if difference.numel() else 0.0
        query_tokens = int(q.shape[0] * q.shape[1])
        output_cases.append(
            {
                **case,
                "case_id": _case_id(case),
                "backend": dispatch_info.backend if dispatch_info else "unknown",
                "fallback_reason": dispatch_info.fallback_reason if dispatch_info else None,
                "max_abs_error": max_abs_error,
                "max_relative_error": max_relative_error,
                "baseline_latency_ms": baseline_latency,
                "latency_ms": candidate_latency,
                "attention_tokens_per_second": query_tokens / (candidate_latency / 1000),
                "decode_latency_ms": candidate_latency if q.shape[1] == 1 else None,
                "first_token_latency_ms": None,
                "baseline_peak_memory_bytes": baseline_peak_memory,
                "peak_memory_bytes": _peak_memory(device),
            }
        )
    return {
        "schema_version": 1,
        "metadata": {
            "device": str(device),
            "device_name": _device_name(device),
            "device_total_memory_bytes": _device_total_memory(device),
            "torch": torch.__version__,
            "runtime": _runtime_version(),
            "driver": _driver_version(),
            "flash_attn": _optional_distribution_version("flash-attn"),
            "vllm": _optional_version("vllm"),
            "warmup": warmup,
            "repeats": repeats,
            "seed": seed,
            "fallback_enabled": os.getenv("MXFLASHATTN_ALLOW_FALLBACK") == "1",
        },
        "cases": output_cases,
    }


def _device_name(device: torch.device) -> str:
    if device.type == "cpu":
        return "CPU reference"
    for module_name in (device.type, "cuda"):
        module = getattr(torch, module_name, None)
        getter = getattr(module, "get_device_name", None)
        if getter is not None:
            try:
                return str(getter(device))
            except (RuntimeError, TypeError):
                pass
    return str(device)


def _device_total_memory(device: torch.device) -> int | None:
    if device.type == "cpu":
        return None
    for module_name in (device.type, "cuda"):
        module = getattr(torch, module_name, None)
        getter = getattr(module, "get_device_properties", None)
        if getter is not None:
            try:
                return int(getter(device).total_memory)
            except (RuntimeError, TypeError, AttributeError, AssertionError):
                pass
    return None


def _runtime_version() -> str | None:
    for name in ("MXMACA_VERSION", "MACA_VERSION", "MUSA_VERSION"):
        if os.getenv(name):
            return os.environ[name]
    return _mx_smi_version(r"MACA Version:\s*([^\s]+)")


def _driver_version() -> str | None:
    for name in ("MXMACA_DRIVER_VERSION", "MACA_DRIVER_VERSION", "MUSA_DRIVER_VERSION"):
        if os.getenv(name):
            return os.environ[name]
    return _mx_smi_version(r"Kernel Mode Driver Version:\s*([^\s]+)")


def _mx_smi_version(pattern: str) -> str | None:
    try:
        result = subprocess.run(
            ["mx-smi"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    match = re.search(pattern, result.stdout)
    return match.group(1) if match else None


def _optional_version(module_name: str) -> str | None:
    try:
        return importlib.metadata.version(module_name.replace("_", "-"))
    except importlib.metadata.PackageNotFoundError:
        try:
            module = importlib.import_module(module_name)
        except (ImportError, OSError):
            return None
    return str(getattr(module, "__version__", "unknown"))


def _optional_distribution_version(distribution: str) -> str | None:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Run MXFlashAttn correctness and latency cases")
    parser.add_argument("--config", type=Path, default=Path("benchmarks/configs/c500.yaml"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/benchmark.json"))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--limit", type=int, default=36)
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()
    if args.warmup < 0 or args.repeats < 1 or args.limit < 1:
        parser.error("warmup must be non-negative; repeats and limit must be positive")
    with args.config.open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    result = run_benchmarks(
        config,
        device=torch.device(args.device),
        warmup=args.warmup,
        repeats=args.repeats,
        limit=args.limit,
        seed=args.seed,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Wrote {len(result['cases'])} benchmark cases to {args.output}")


if __name__ == "__main__":
    main()
