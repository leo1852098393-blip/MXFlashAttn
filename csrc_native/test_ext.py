"""Gate 2 step 1: 自研 kernel 的 torch extension vs vendor 数值对齐（fp16 + bf16）"""
import os, sys
import numpy as np
import torch
from vllm_metax.v1.attention.backends.fa_utils import flash_attn_with_kvcache

EXT_DIR = "/root/MXFlashAttn_v1.3.0-kernel/csrc_native"
sys.path.insert(0, EXT_DIR)
import importlib.util
spec = importlib.util.spec_from_file_location("mxfadecode", os.path.join(EXT_DIR, "mxfadecode.so"))
ext = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ext)
print("[step] extension loaded", flush=True)

B, HKV, GQ, HQ, D, PS, PPT = 2, 2, 2, 4, 128, 16, 16
PAGES = B * PPT
sl = [200, 256]
dev = "cuda"
scale = D ** -0.5

def run_case(dt):
    tag = str(dt)
    torch.manual_seed(20261007)
    q = (torch.randn(B, 1, HQ, D, device=dev, dtype=dt) * 0.5)
    k_cache = torch.randn(PAGES, PS, HKV, D, device=dev, dtype=dt) * 0.5
    v_cache = torch.randn(PAGES, PS, HKV, D, device=dev, dtype=dt)
    perm = torch.randperm(PAGES, device=dev)
    bt = perm[: B * PPT].reshape(B, PPT).to(torch.int32)
    cache_seqlens = torch.tensor(sl, dtype=torch.int32, device=dev)

    vout = flash_attn_with_kvcache(q=q, k_cache=k_cache, v_cache=v_cache,
                                   block_table=bt, cache_seqlens=cache_seqlens,
                                   softmax_scale=scale, causal=True)
    if isinstance(vout, tuple):
        vout = vout[0]
    stream = torch.cuda.current_stream().cuda_stream
    oout = ext.paged_decode(q, k_cache, v_cache, bt, cache_seqlens, scale, max(sl), stream)
    torch.cuda.synchronize()

    diff = (oout.float() - vout.float()).abs().max().item()
    # double reference
    qc = q.float().cpu().numpy(); kc = k_cache.float().cpu().numpy(); vc = v_cache.float().cpu().numpy()
    btc = bt.cpu().numpy()
    ref = np.zeros((B, 1, HQ, D), dtype=np.float64)
    scl = 1.0 / np.sqrt(D)
    for b in range(B):
        for hq in range(HQ):
            kvh = hq // GQ
            scores = np.zeros(max(sl), dtype=np.float64)
            for s in range(sl[b]):
                phys = btc[b, s // PS]
                scores[s] = np.dot(qc[b, 0, hq].astype(np.float64), kc[phys, s % PS, kvh].astype(np.float64)) * scl
            m = scores[: sl[b]].max()
            p = np.exp(scores[: sl[b]] - m); p = p / p.sum()
            for d in range(D):
                acc = 0.0
                for s in range(sl[b]):
                    acc += p[s] * float(vc[btc[b, s // PS], s % PS, kvh, d])
                ref[b, 0, hq, d] = acc
    dv = (oout.float().cpu().numpy() - ref).max()
    dvv = (vout.float().cpu().numpy() - ref).max()
    print("[%s] ours vs vendor : %.3e | ours vs ref : %.3e | vendor vs ref : %.3e" % (tag, diff, dv, dvv), flush=True)
    return diff

d16 = run_case(torch.float16)
dbf = run_case(torch.bfloat16)
ok = d16 < 1e-3 and dbf < 1e-2  # bf16 output quantization is coarser
print("GATE2-EXT %s" % ("PASSED" if ok else "FAILED"), flush=True)
