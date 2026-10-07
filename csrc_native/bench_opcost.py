"""指令成本微探针：单 block 逐变体计时。"""
import os, importlib.util
import torch

SRC = "/root/MXFlashAttn_v1.3.0-kernel/csrc_native"

def load_ext(name, so):
    spec = importlib.util.spec_from_file_location(name, os.path.join(SRC, so))
    ext = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ext)
    return ext

m1 = load_ext("mxmmain", "mxmmain.so")

def measure_gpu(fn, iters=20, warmup=5):
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

x = torch.randn(256, device="cuda", dtype=torch.float32)
out = torch.zeros(5 * 256, device="cuda", dtype=torch.float32)
stream = torch.cuda.current_stream().cuda_stream

NAMES = ["empty     ", "16FMA     ", "8shuffle  ", "8exp2     ", "8shfl+8exp2"]
LOOPS = 20000
base = None
for v in range(5):
    t = measure_gpu(lambda: m1.opcost_probe(x, LOOPS, v, stream))
    per = t * 1e6 / LOOPS
    if v == 0:
        base = per
    print("[opcost] %s: %.4f ms 总, 循环体每迭代 ~%.1f ns (扣空载 %.1f ns)"
          % (NAMES[v], t, per - base + (base * 0 if v else 0), per - base if v else 0), flush=True)

print("OPCOST-DONE", flush=True)
