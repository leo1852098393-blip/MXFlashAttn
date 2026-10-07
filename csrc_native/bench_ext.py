"""Gate 3 微基准：自研 kernel（快路径）vs vendor，解码形状（无 vLLM 包装开销）。"""
import os, sys, time, importlib.util
import torch
from vllm_metax.v1.attention.backends.fa_utils import flash_attn_with_kvcache

EXT_DIR = "/root/MXFlashAttn_v1.3.0-kernel/csrc_native"
spec = importlib.util.spec_from_file_location("mxfadecode", os.path.join(EXT_DIR, "mxfadecode.so"))
ext = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ext)
print("[bench] extension loaded", flush=True)

dev = "cuda"

def bench(fn, iters=200, warmup=20):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1000.0  # ms

def run(tag, B, HKV, HQ, D, PS, PPT, sl_val, dt):
    torch.manual_seed(20261007)
    q = torch.randn(B, 1, HQ, D, device=dev, dtype=dt) * 0.5
    PAGES = B * PPT
    k_cache = torch.randn(PAGES, PS, HKV, D, device=dev, dtype=dt) * 0.5
    v_cache = torch.randn(PAGES, PS, HKV, D, device=dev, dtype=dt)
    perm = torch.randperm(PAGES, device=dev)
    bt = perm[: B * PPT].reshape(B, PPT).to(torch.int32)
    sl = torch.full((B,), sl_val, dtype=torch.int32, device=dev)
    scale = D ** -0.5
    stream = torch.cuda.current_stream().cuda_stream

    def vendor():
        return flash_attn_with_kvcache(q=q, k_cache=k_cache, v_cache=v_cache,
                                       block_table=bt, cache_seqlens=sl,
                                       softmax_scale=scale, causal=True)

    def ours():
        return ext.paged_decode(q, k_cache, v_cache, bt, sl, scale, PPT * PS, stream)

    tv = bench(vendor)
    to = bench(ours)
    # numerics spot check
    with torch.no_grad():
        vout = vendor()
        if isinstance(vout, tuple):
            vout = vout[0]
        oout = ours()
    diff = (oout.float() - vout.float()).abs().max().item()
    print("[%s] B=%d sl=%d D=%d %s | vendor %.4f ms | ours %.4f ms | ratio %.3fx | maxdiff %.2e"
          % (tag, B, sl_val, D, dt, tv, to, tv / to, diff), flush=True)

# Qwen3-0.6B 解码典型形状: 28 层, HQ=16, HKV=8, D=128
run("small", 8, 8, 16, 128, 16, 16, 128, torch.float16)
run("mid", 32, 8, 16, 128, 16, 32, 512, torch.float16)
run("big", 96, 8, 16, 128, 16, 64, 1024, torch.float16)
run("mid-bf16", 32, 8, 16, 128, 16, 32, 512, torch.bfloat16)
print("BENCH-DONE", flush=True)
