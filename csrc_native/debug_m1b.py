"""mode 0 小样本逐元素 dump：B=1 GQ=2 D=128 sl=1/3，定位 mode 0 到底错在哪。"""
import os, importlib.util
import numpy as np
import torch
from vllm_metax.v1.attention.backends.fa_utils import flash_attn_with_kvcache

SRC = "/root/MXFlashAttn_v1.3.0-kernel/csrc_native"

def load_ext(name, so):
    spec = importlib.util.spec_from_file_location(name, os.path.join(SRC, so))
    ext = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ext)
    return ext

m1 = load_ext("mxmmain", "mxmmain.so")
dev = "cuda"
np.set_printoptions(precision=4, suppress=True, linewidth=200)

torch.manual_seed(20261007)
B, HKV, HQ, D, PS, PPT = 1, 1, 2, 128, 16, 1
GQ = HQ // HKV

q = torch.randn(B, 1, HQ, D, device=dev, dtype=torch.float16) * 0.5
k_cache = torch.randn(PPT, PS, HKV, D, device=dev, dtype=torch.float16) * 0.5
v_cache = torch.randn(PPT, PS, HKV, D, device=dev, dtype=torch.float16)
bt = torch.arange(PPT, dtype=torch.int32, device=dev).reshape(B, PPT)

for sl_val in (1, 2, 3):
    sl = torch.full((B,), sl_val, dtype=torch.int32, device=dev)
    scale = D ** -0.5
    stream = torch.cuda.current_stream().cuda_stream

    ref = flash_attn_with_kvcache(q=q, k_cache=k_cache, v_cache=v_cache,
                                  block_table=bt, cache_seqlens=sl,
                                  softmax_scale=scale, causal=True).float()
    o0 = m1.paged_decode(q, k_cache, v_cache, bt, sl, scale, PPT * PS, stream, 0).float()
    o1 = m1.paged_decode(q, k_cache, v_cache, bt, sl, scale, PPT * PS, stream, 1).float()

    print(f"===== sl={sl_val} =====")
    for h in range(HQ):
        r = ref[0, 0, h].cpu().numpy()
        a = o0[0, 0, h].cpu().numpy()
        b = o1[0, 0, h].cpu().numpy()
        print(f"head{h} ref    :", r[:6])
        print(f"head{h} mode0  :", a[:6], " maxdiff=%.4g" % np.abs(a - r).max())
        print(f"head{h} mode1  :", b[:6], " maxdiff=%.4g" % np.abs(b - r).max())
    # 参照行：pos0 的 V 行（sl=1 时两 head 输出都应等于它）
    print("V[0,0,0,:6] =", v_cache[0, 0, 0, :6].cpu().numpy())
    if sl_val >= 2:
        print("V[0,1,0,:6] =", v_cache[0, 1, 0, :6].cpu().numpy())
    # K 行也打出来，便于比对打分
    print("K[0,0,0,:6] =", k_cache[0, 0, 0, :6].cpu().numpy())
print("DBG-DONE", flush=True)
