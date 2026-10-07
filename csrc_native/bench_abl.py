"""Gate 4 消融实验：MODE 0=全量 1=无V 2=无expf 3=无K，定位 kernel 慢的元凶。"""
import os, importlib.util
import torch
from vllm_metax.v1.attention.backends.fa_utils import flash_attn_with_kvcache

EXT_DIR = "/root/MXFlashAttn_v1.3.0-kernel/csrc_native"
spec = importlib.util.spec_from_file_location("mxfadecode", os.path.join(EXT_DIR, "mxfadecode.so"))
ext = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ext)
print("[abl] loaded", flush=True)

dev = "cuda"
stream = torch.cuda.current_stream().cuda_stream

def gpu_time(fn, iters=100, warmup=20):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    ev0 = torch.cuda.Event(enable_timing=True); ev1 = torch.cuda.Event(enable_timing=True)
    ev0.record()
    for _ in range(iters):
        fn()
    ev1.record(); torch.cuda.synchronize()
    return ev0.elapsed_time(ev1) / iters

def run(B, HKV, HQ, D, PS, PPT, sl_val):
    torch.manual_seed(20261007)
    q = torch.randn(B, 1, HQ, D, device=dev, dtype=torch.float16) * 0.5
    PAGES = B * PPT
    k_cache = torch.randn(PAGES, PS, HKV, D, device=dev, dtype=torch.float16) * 0.5
    v_cache = torch.randn(PAGES, PS, HKV, D, device=dev, dtype=torch.float16)
    perm = torch.randperm(PAGES, device=dev)
    bt = perm[: B * PPT].reshape(B, PPT).to(torch.int32)
    sl = torch.full((B,), sl_val, dtype=torch.int32, device=dev)
    scale = D ** -0.5

    def vendor():
        return flash_attn_with_kvcache(q=q, k_cache=k_cache, v_cache=v_cache,
                                       block_table=bt, cache_seqlens=sl,
                                       softmax_scale=scale, causal=True)
    tv = gpu_time(vendor)
    row = "B=%d sl=%d | vendor %.4f" % (B, sl_val, tv)
    for mode, name in [(0, "full"), (1, "noV"), (2, "noexpf"), (3, "noK")]:
        t = gpu_time(lambda: ext.paged_decode_mode(q, k_cache, v_cache, bt, sl, scale, PPT * PS, stream, mode))
        row += " | %s %.4f" % (name, t)
    print(row, flush=True)

run(96, 8, 16, 128, 16, 8, 16)
run(96, 8, 16, 128, 16, 8, 32)
run(96, 8, 16, 128, 16, 16, 128)
run(32, 8, 16, 128, 16, 32, 512)
run(96, 8, 16, 128, 16, 64, 1024)
print("ABL-DONE", flush=True)
