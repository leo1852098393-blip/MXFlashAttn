"""平台体检：带宽 / 时钟 / expf 吞吐。"""
import os, time, importlib.util
import torch

SRC = "/root/MXFlashAttn_v1.3.0-kernel/csrc_native"
spec = importlib.util.spec_from_file_location("mxprobe", os.path.join(SRC, "mxprobe.so"))
ext = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ext)
print("[probe] loaded", flush=True)

dev = "cuda"
stream = torch.cuda.current_stream().cuda_stream
sink = torch.zeros(1, device=dev, dtype=torch.float32)

def gpu_time(fn, iters=20, warmup=5):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    ev0 = torch.cuda.Event(enable_timing=True); ev1 = torch.cuda.Event(enable_timing=True)
    ev0.record()
    for _ in range(iters):
        fn()
    ev1.record(); torch.cuda.synchronize()
    return ev0.elapsed_time(ev1) / iters

# 1. 流式读带宽: 512MB buffer
N4 = 128 * 1024 * 1024   # float4 数 = 512MB/16B... 128M*16=2GB 太大, 用 32M*16=512MB
N4 = 32 * 1024 * 1024
src = torch.randn(N4 * 4, device=dev, dtype=torch.float32)
grid, block = 132 * 8, 256
ms = gpu_time(lambda: ext.probe_stream(src.data_ptr(), sink.data_ptr(), N4, grid, block, stream))
bw = (N4 * 16) / (ms / 1000) / 1e9
print("stream_read: %.3f ms -> %.0f GB/s" % (ms, bw), flush=True)
del src

# 2. FMA 链: 104 SM × 64 warp × 32 lane 并行, 每线程 100000 依赖 FMA
iters = 100000
grid, block = 104 * 64, 32
ms = gpu_time(lambda: ext.probe_fma_chain(sink.data_ptr(), iters, grid, block, stream), iters=5, warmup=2)
total_fma = 104 * 64 * 32 * iters
print("fma_chain: %.3f ms -> %.1f GFMA/s" % (ms, total_fma / (ms / 1000) / 1e9), flush=True)

# 3. 并行 FMA (ILP 上限)
ms = gpu_time(lambda: ext.probe_fma_par(sink.data_ptr(), iters, grid, block, stream), iters=5, warmup=2)
print("fma_par:   %.3f ms -> %.1f GFMA/s" % (ms, total_fma / (ms / 1000) / 1e9), flush=True)

# 4. expf 吞吐
ms = gpu_time(lambda: ext.probe_expf(sink.data_ptr(), iters, grid, block, stream), iters=5, warmup=2)
print("expf_loop: %.3f ms -> %.1f Gexpf/s" % (ms, total_fma / (ms / 1000) / 1e9), flush=True)

print("PROBE-DONE", flush=True)
