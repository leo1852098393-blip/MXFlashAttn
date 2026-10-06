"""Run a same-input six-column MXFlashAttn benchmark matrix.

The operator columns are measured in one process on one device.  The vLLM
columns are imported from separately captured model runs and are never used to
inflate the operator speedup numbers.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
import os
import statistics
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from mxflashattn import flash_attn_func, flash_attn_varlen_func, flash_attn_with_kvcache
from mxflashattn import reference


def _sync(device: torch.device) -> None:
    module = getattr(torch, device.type, None)
    sync = getattr(module, "synchronize", None)
    if sync:
        sync(device)


def _measure(fn, device: torch.device, repeats: int) -> tuple[float, torch.Tensor]:
    result = fn()
    _sync(device)
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        result = fn()
        _sync(device)
        samples.append((time.perf_counter() - start) * 1000)
    return statistics.median(samples), result


def _make_cache(k: torch.Tensor, v: torch.Tensor, page_size: int = 16):
    batch, kv_len, kv_heads, head_dim = k.shape
    pages = (kv_len + page_size - 1) // page_size
    cache_k = torch.zeros(batch * pages, page_size, kv_heads, head_dim, device=k.device, dtype=k.dtype)
    cache_v = torch.zeros_like(cache_k)
    table = torch.arange(batch * pages, device=k.device, dtype=torch.int32).reshape(batch, pages)
    for b in range(batch):
        for p in range(pages):
            start = p * page_size
            count = min(page_size, kv_len - start)
            cache_k[int(table[b, p]), :count] = k[b, start : start + count]
            cache_v[int(table[b, p]), :count] = v[b, start : start + count]
    lengths = torch.full((batch,), kv_len, device=k.device, dtype=torch.int32)
    return cache_k, cache_v, table, lengths


def _dense_sdpa(q, k, v, causal):
    ratio = q.shape[2] // k.shape[2]
    kh = k.transpose(1, 2).repeat_interleave(ratio, dim=1)
    vh = v.transpose(1, 2).repeat_interleave(ratio, dim=1)
    return F.scaled_dot_product_attention(
        q.transpose(1, 2), kh, vh, is_causal=causal, scale=1.0 / math.sqrt(q.shape[-1])
    ).transpose(1, 2)


def _case(index: int) -> dict:
    # 36 deterministic combinations cover decode/prefill, both layouts,
    # FP16/BF16, GQA/MQA, two head sizes, and causal/non-causal attention.
    return {
        "case_id": f"case-{index:02d}",
        "batch_size": 1 + (index % 2),
        "q_len": (1, 8)[(index // 2) % 2],
        "kv_len": (64, 128)[(index // 4) % 2],
        "head_dim": (64, 128)[(index // 8) % 2],
        "gqa_ratio": (1, 4)[(index // 16) % 2],
        "dtype": ("float16", "bfloat16")[(index // 32) % 2],
        "causal": bool(index % 3),
        "layout": ("dense", "varlen", "paged")[index % 3],
    }


def _run_case(case: dict, device: torch.device, repeats: int, vendor_dense, vendor_varlen, vendor_paged):
    dtype = getattr(torch, case["dtype"])
    b, q_len, kv_len, dim = case["batch_size"], case["q_len"], case["kv_len"], case["head_dim"]
    kv_heads = 2
    q_heads = kv_heads * case["gqa_ratio"]
    generator = torch.Generator(device="cpu").manual_seed(20261005 + int(case["case_id"][-2:]))
    q = torch.randn(b, q_len, q_heads, dim, dtype=dtype, generator=generator).to(device)
    k = torch.randn(b, kv_len, kv_heads, dim, dtype=dtype, generator=generator).to(device)
    v = torch.randn_like(k)
    causal = case["causal"]
    cache_k, cache_v, table, lengths = _make_cache(k, v)

    def reference_run():
        return reference.attention_batched(q, k, v, causal=causal)

    def sdpa_run():
        return _dense_sdpa(q, k, v, causal)

    def mx_run():
        if case["layout"] == "dense":
            return flash_attn_func(q, k, v, causal=causal)
        if case["layout"] == "varlen":
            cu_q = torch.arange(0, (b + 1) * q_len, q_len, dtype=torch.int32, device=device)
            cu_k = torch.arange(0, (b + 1) * kv_len, kv_len, dtype=torch.int32, device=device)
            return flash_attn_varlen_func(q.reshape(-1, q_heads, dim), k.reshape(-1, kv_heads, dim), v.reshape(-1, kv_heads, dim), cu_q, cu_k, max_seqlen_q=q_len, max_seqlen_k=kv_len, causal=causal).reshape_as(q)
        return flash_attn_with_kvcache(q, cache_k, cache_v, cache_seqlens=lengths, block_table=table, causal=causal)

    def vendor_run():
        if case["layout"] == "dense":
            return vendor_dense(q, k, v, causal=causal)
        if case["layout"] == "varlen":
            cu_q = torch.arange(0, (b + 1) * q_len, q_len, dtype=torch.int32, device=device)
            cu_k = torch.arange(0, (b + 1) * kv_len, kv_len, dtype=torch.int32, device=device)
            return vendor_varlen(q.reshape(-1, q_heads, dim), k.reshape(-1, kv_heads, dim), v.reshape(-1, kv_heads, dim), cu_q, cu_k, max_seqlen_q=q_len, max_seqlen_k=kv_len, causal=causal).reshape_as(q)
        return vendor_paged(q, cache_k, cache_v, cache_seqlens=lengths, block_table=table, causal=causal)

    funcs = {"reference": reference_run, "torch_sdpa": sdpa_run, "mxflashattn_api": mx_run, "metax_vendor": vendor_run}
    outputs, timings = {}, {}
    for name, fn in funcs.items():
        try:
            elapsed, output = _measure(fn, device, repeats)
            timings[name] = elapsed
            outputs[name] = output.detach().float().cpu()
        except Exception as exc:
            timings[name] = None
            outputs[name] = None
            case.setdefault("errors", {})[name] = f"{type(exc).__name__}: {exc}"
    base = outputs["reference"]
    errors = {}
    if base is not None:
        for name, output in outputs.items():
            if output is not None:
                delta = (output - base).abs()
                errors[name] = {"max_abs": float(delta.max()), "max_rel": float((delta / base.abs().clamp_min(1e-6)).max())}
    case["timings_ms"] = timings
    case["errors_vs_reference"] = errors
    case["tokens"] = b * q_len
    return case


def run_matrix(device: torch.device, repeats: int, count: int) -> dict:
    try:
        vendor = __import__("flash_attn", fromlist=["flash_attn_func", "flash_attn_varlen_func", "flash_attn_with_kvcache"])
        vendor_dense = vendor.flash_attn_func
        vendor_varlen = vendor.flash_attn_varlen_func
        vendor_paged = vendor.flash_attn_with_kvcache
    except (ImportError, AttributeError) as exc:
        vendor_dense = vendor_varlen = vendor_paged = None
        vendor_error = f"{type(exc).__name__}: {exc}"
    else:
        vendor_error = None
    cases = []
    for index in range(count):
        case = _case(index)
        if vendor_dense is None:
            case["errors"] = {"metax_vendor": vendor_error}
            case["timings_ms"] = {name: None for name in ("reference", "torch_sdpa", "mxflashattn_api", "metax_vendor")}
            cases.append(case)
        else:
            cases.append(_run_case(case, device, repeats, vendor_dense, vendor_varlen, vendor_paged))
    metadata = {"device": str(device), "torch": torch.__version__, "vllm": _version("vllm"), "flash_attn": _version("flash-attn"), "vllm_metax": _version("vllm_metax"), "mxmaca": os.getenv("MXMACA_VERSION", "unknown"), "driver": os.getenv("MXMACA_DRIVER_VERSION", "unknown"), "seed": 20261005, "repeats": repeats, "case_count": len(cases), "vendor_error": vendor_error}
    return {"schema_version": 1, "metadata": metadata, "cases": cases}


def _version(name: str):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--count", type=int, default=36)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--output", type=Path, default=Path("artifacts/fair-matrix.json"))
    args = parser.parse_args()
    result = run_matrix(torch.device(args.device), args.repeats, args.count)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Wrote {len(result['cases'])} fair benchmark cases to {args.output}")


if __name__ == "__main__":
    main()
