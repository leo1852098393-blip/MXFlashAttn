"""Gate 4 bench v3：分离 GPU 时间（CUDA event）与 CPU 派发时间。"""
import os, sys, time, importlib.util
import torch
from vllm_metax.v1.attention.backends.fa_utils import flash_attn_with_kvcache

EXT_DIR = "/root/MXFlashAttn_v1.3.0-kernel/csrc_native"
spec = importlib.util.spec_from_file_location("mxfadecode", os.path.join(EXT_DIR, "mxfadecode.so"))
ext = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ext)
print("[bench3] extension loaded", flush=True)

dev = "cuda"

def measure(fn, iters=200, warmup=30):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    # CPU enqueue rate（不同步，看派发多快）
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    t_enq = (time.perf_counter() - t0) / iters * 1000.0
    torch.cuda.synchronize()
    # GPU 时间：event 包住同一批
    ev0 = torch.cuda.Event(enable_timing=True)
    ev1 = torch.cuda.Event(enable_timing=True)
    ev0.record()
    for _ in range(iters):
        fn()
    ev1.record()
    torch.cuda.synchronize()
    t_gpu = ev0.elapsed_time(ev1) / iters
    return t_enq, t_gpu

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

    ve, vg = measure(vendor)
    oe, og = measure(ours)
    path = "fast" if D in (64, 128, 256) else "naive"
    print("[%s] %s B=%d sl=%d D=%d | vendor: enq %.4f gpu %.4f ms | ours: enq %.4f gpu %.4f ms | gpu ratio %.2fx"
          % (tag, path, B, sl_val, D, ve, vg, oe, og, vg / og), flush=True)

run("s16", 96, 8, 16, 128, 16, 8, 16, torch.float16)
run("s32", 96, 8, 16, 128, 16, 8, 32, torch.float16)
run("s128", 96, 8, 16, 128, 16, 16, 128, torch.float16)
run("s512", 32, 8, 16, 128, 16, 32, 512, torch.float16)
run("s1024", 96, 8, 16, 128, 16, 64, 1024, torch.float16)
print("BENCH3-DONE", flush=True)
