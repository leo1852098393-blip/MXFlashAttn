"""带宽天花板基准：torch clone/sum 对照 + M1 各路径换算带宽。"""
import torch, os, importlib.util

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

# 与 s1024 配置同规模的数据量：402MB
N = 96 * 8 * 1024 * 128 * 2  # K+V 元素数
x = torch.randn(N, device=dev, dtype=torch.float16)
print("[bw] data = %.1f MB" % (x.numel() * 2 / 1e6), flush=True)

t_clone = measure_gpu(lambda: x.clone())
bw_clone = 2 * x.numel() * 2 / (t_clone / 1e3) / 1e9
print("[bw] clone      %.4f ms -> %.0f GB/s (r+w)" % (t_clone, bw_clone), flush=True)

t_sum = measure_gpu(lambda: x.sum())
bw_sum = x.numel() * 2 / (t_sum / 1e3) / 1e9
print("[bw] sum        %.4f ms -> %.0f GB/s (read)" % (t_sum, bw_sum), flush=True)

t_copy = measure_gpu(lambda: x.fill_(1.0))
bw_fill = x.numel() * 2 / (t_copy / 1e3) / 1e9
print("[bw] fill       %.4f ms -> %.0f GB/s (write)" % (t_copy, bw_fill), flush=True)

y = torch.empty_like(x)
t_c2 = measure_gpu(lambda: y.copy_(x))
bw_c2 = 2 * x.numel() * 2 / (t_c2 / 1e3) / 1e9
print("[bw] copy_      %.4f ms -> %.0f GB/s (r+w)" % (t_c2, bw_c2), flush=True)

# 小 block 数下的 clone（对照低并发）
for nb in [16, 64, 256]:
    pass  # torch 无法直接控 block 数，跳过

# 换算：M1 s1024 实测 1.33ms 搬 402MB = 302GB/s；vendor 0.283ms = 1420GB/s
print("[bw] M1 s1024 = 302 GB/s | vendor = 1420 GB/s", flush=True)
print("BW-DONE", flush=True)
