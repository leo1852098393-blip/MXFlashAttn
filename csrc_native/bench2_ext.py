"""Gate 3 微基准 v2：短序列真实工况 + 新旧 kernel 对比（D=96 强制走 naive 兜底路径）。"""
import os, sys, time, importlib.util
import torch
from vllm_metax.v1.attention.backends.fa_utils import flash_attn_with_kvcache

EXT_DIR = "/root/MXFlashAttn_v1.3.0-kernel/csrc_native"
spec = importlib.util.spec_from_file_location("mxfadecode", os.path.join(EXT_DIR, "mxfadecode.so"))
ext = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ext)
print("[bench2] extension loaded", flush=True)

dev = "cuda"

def bench(fn, iters=300, warmup=30):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1000.0

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
    with torch.no_grad():
        vout = vendor()
        if isinstance(vout, tuple):
            vout = vout[0]
        oout = ours()
    diff = (oout.float() - vout.float()).abs().max().item()
    path = "fast" if D in (64, 128, 256) else "naive"
    print("[%s] %s B=%d sl=%d D=%d %s | vendor %.4f ms | ours %.4f ms | ratio %.3fx | maxdiff %.2e"
          % (tag, path, B, sl_val, D, dt, tv, to, tv / to, diff), flush=True)

# 真实 smoke 工况：解码长度 8~48
run("s16", 96, 8, 16, 128, 16, 8, 16, torch.float16)
run("s32", 96, 8, 16, 128, 16, 8, 32, torch.float16)
run("s64", 96, 8, 16, 128, 16, 16, 64, torch.float16)
run("s128", 96, 8, 16, 128, 16, 16, 128, torch.float16)
# 新旧 kernel 对比：D=96 落 naive 兜底路径（vendor 同形状，公平）
run("naive-s32", 96, 8, 16, 96, 16, 8, 32, torch.float16)
run("naive-s128", 96, 8, 16, 96, 16, 16, 128, torch.float16)
run("naive-s512", 32, 8, 16, 96, 16, 32, 512, torch.float16)
print("BENCH2-DONE", flush=True)
