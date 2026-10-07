"""流式带宽探针：同形 paged 遍历，无注意力数学。对照三个加载通路。"""
import os, importlib.util
import torch

SRC = "/root/MXFlashAttn_v1.3.0-kernel/csrc_native"

def load_ext(name, so):
    spec = importlib.util.spec_from_file_location(name, os.path.join(SRC, so))
    ext = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ext)
    return ext

m1 = load_ext("mxmmain", "mxmmain.so")
print("[stream] loaded", flush=True)

dev = "cuda"

def measure_gpu(fn, iters=100, warmup=20):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    ev0 = torch.cuda.Event(enable_timing=True)
    ev1 = torch.cuda.Event(enable_timing=True)
    ev0.record()
    for _ in range(iters):
        fn()
    ev1.record()
    torch.cuda.synchronize()
    return ev0.elapsed_time(ev1) / iters

def run(tag, B, HKV, HD, PS, PPT, sl_val):
    torch.manual_seed(20261007)
    PAGES = B * PPT
    k_cache = torch.randn(PAGES, PS, HKV, HD, device=dev, dtype=torch.float16) * 0.5
    v_cache = torch.randn(PAGES, PS, HKV, HD, device=dev, dtype=torch.float16) * 0.5
    perm = torch.randperm(PAGES, device=dev)
    bt = perm[: B * PPT].reshape(B, PPT).to(torch.int32)
    sl = torch.full((B,), sl_val, dtype=torch.int32, device=dev)
    stream = torch.cuda.current_stream().cuda_stream
    data = B * HKV * sl_val * HD * 2 * 2  # K+V bytes

    res = []
    for var, name in [(0, "cp+计数"), (1, "cp+waitall"), (2, "纯全局")]:
        t = measure_gpu(lambda: m1.stream_probe(k_cache, v_cache, bt, sl, var, stream))
        res.append("%s %.4f ms = %.0f GB/s" % (name, t, data / (t / 1e3) / 1e9))
    print("[stream %s] B=%d sl=%d HD=%d | %s" % (tag, B, sl_val, HD, " | ".join(res)), flush=True)

run("s1024", 96, 8, 128, 16, 64, 1024)
run("s512",  32, 8, 128, 16, 32, 512)
run("s2048", 96, 8, 128, 16, 128, 2048)

# ---- 指令开销微探针：纯寄存器循环测 shuffle / exp2f / FMA 单条成本 ----
N = 96 * 8 * 1024  # 786K 线程
x = torch.ones(N, device=dev, dtype=torch.float32)

def op_probe(fn, iters=50, warmup=10, ops_per_thread=1000):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    ev0 = torch.cuda.Event(enable_timing=True)
    ev1 = torch.cuda.Event(enable_timing=True)
    ev0.record()
    for _ in range(iters):
        fn()
    ev1.record()
    torch.cuda.synchronize()
    t_ms = ev0.elapsed_time(ev1) / iters
    total_ops = N * ops_per_thread
    return t_ms * 1e6 / total_ops * 1e3 / 1e3, t_ms  # ns per op-lane

# 用 torch 原生算子近似不了 shuffle —— 直接跳过，改为测 exp2f 和 mul 的吞吐对照
import math
def bench_torch_op(fn, name, flops_per_elem):
    t = measure_gpu(fn, iters=50, warmup=10)
    print("[op] %-12s %.4f ms -> %.1f T elem-ops/s" % (name, t, N * flops_per_elem / (t / 1e3) / 1e12), flush=True)

bench_torch_op(lambda: torch.exp2(x), "exp2f", 1)
bench_torch_op(lambda: x * 2.0, "mul", 1)
print("STREAM-DONE", flush=True)
