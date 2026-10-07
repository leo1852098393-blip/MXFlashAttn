"""M1 验证：cp_async 流水内核 vs vendor vs v7。正确性 + GPU 时间。
用法：python test_m1.py [tag ...]  # 传 tag 只跑指定配置（子进程隔离坏配置）"""
import os, sys, importlib.util
import numpy as np
import torch
from vllm_metax.v1.attention.backends.fa_utils import flash_attn_with_kvcache

SRC = "/root/MXFlashAttn_v1.3.0-kernel/csrc_native"
_ONLY = sys.argv[1:]  # 传 tag 只跑指定配置

def load_ext(name, so):
    spec = importlib.util.spec_from_file_location(name, os.path.join(SRC, so))
    ext = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ext)
    return ext

v7 = load_ext("mxfadecode", "mxfadecode.so")
m1 = load_ext("mxmmain", "mxmmain.so")
print("[m1] extensions loaded", flush=True)

dev = "cuda"

def measure_gpu(fn, iters=200, warmup=30):
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

def run(tag, B, HKV, HQ, D, PS, PPT, sl_val, dt, qmul=1.0):
    if _ONLY and tag not in _ONLY:
        return
    torch.manual_seed(20261007)
    q = torch.randn(B, 1, HQ, D, device=dev, dtype=dt) * 0.5 * qmul
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

    out_v = vendor().float()
    out_m1 = m1.paged_decode(q, k_cache, v_cache, bt, sl, scale, PPT * PS, stream, 0).float()
    diff = (out_m1 - out_v).abs().max().item()
    # bsm 实验路径的正确性 + 耗时
    out_b = m1.paged_decode(q, k_cache, v_cache, bt, sl, scale, PPT * PS, stream, 1).float()
    diff_b = (out_b - out_v).abs().max().item()
    # dir 直连全局路径（定位基准）
    out_d = m1.paged_decode(q, k_cache, v_cache, bt, sl, scale, PPT * PS, stream, 2).float()
    diff_d = (out_d - out_v).abs().max().item()
    # v3：vendor 结构复刻（mode3 = NS1, mode4 = NS2, mode7 = NS1 数学剥离诊断）
    out_v3 = m1.paged_decode(q, k_cache, v_cache, bt, sl, scale, PPT * PS, stream, 3).float()
    diff_3 = (out_v3 - out_v).abs().max().item()
    out_v4 = m1.paged_decode(q, k_cache, v_cache, bt, sl, scale, PPT * PS, stream, 4).float()
    diff_4 = (out_v4 - out_v).abs().max().item()
    # mode 10 = 统一 max（FlashDecoding++ 式，循环内零重缩放）
    out_um = m1.paged_decode(q, k_cache, v_cache, bt, sl, scale, PPT * PS, stream, 10).float()
    diff_um = (out_um - out_v).abs().max().item()
    # mode 11 = 分桶惰性缩放（bucketed UM，无溢出，缩放代数与 online softmax 等价）
    out_um2 = m1.paged_decode(q, k_cache, v_cache, bt, sl, scale, PPT * PS, stream, 11).float()
    diff_um2 = (out_um2 - out_v).abs().max().item()

    t_vendor = measure_gpu(vendor)
    t_m1 = measure_gpu(lambda: m1.paged_decode(q, k_cache, v_cache, bt, sl, scale, PPT * PS, stream, 0))
    t_bsm = measure_gpu(lambda: m1.paged_decode(q, k_cache, v_cache, bt, sl, scale, PPT * PS, stream, 1))
    t_v3 = measure_gpu(lambda: m1.paged_decode(q, k_cache, v_cache, bt, sl, scale, PPT * PS, stream, 3))
    t_v4 = measure_gpu(lambda: m1.paged_decode(q, k_cache, v_cache, bt, sl, scale, PPT * PS, stream, 4))
    t_mf = measure_gpu(lambda: m1.paged_decode(q, k_cache, v_cache, bt, sl, scale, PPT * PS, stream, 7))
    t_mf2 = measure_gpu(lambda: m1.paged_decode(q, k_cache, v_cache, bt, sl, scale, PPT * PS, stream, 8))
    t_mf3 = measure_gpu(lambda: m1.paged_decode(q, k_cache, v_cache, bt, sl, scale, PPT * PS, stream, 9))
    t_um = measure_gpu(lambda: m1.paged_decode(q, k_cache, v_cache, bt, sl, scale, PPT * PS, stream, 10))
    t_um2 = measure_gpu(lambda: m1.paged_decode(q, k_cache, v_cache, bt, sl, scale, PPT * PS, stream, 11))

    # split-KV：NP 扫描（找并发甜点；低并发场景扫到 64）
    # ROWS=BDZ*GQ，NTH=256(GQ8=512)：GQ2/GQ4→ROWS16，GQ8→ROWS32；c_len 须按实际 ROWS 对齐
    ROWS_BLK = 32 if (HQ // HKV) == 8 else 16
    if sl_val >= 4096:
        np_list = [2, 4, 8, 16, 32, 64]
    elif sl_val >= 512:
        np_list = [1, 2, 4, 8, 16]
    else:
        np_list = [4]
    split_parts = []
    for np_ in np_list:
        c_len = ((sl_val + np_ - 1) // np_ + ROWS_BLK - 1) // ROWS_BLK * ROWS_BLK
        out_s = m1.paged_decode_split(q, k_cache, v_cache, bt, sl, scale, stream, np_, c_len).float()
        diff_s = (out_s - out_v).abs().max().item()
        t_s = measure_gpu(lambda: m1.paged_decode_split(q, k_cache, v_cache, bt, sl, scale, stream, np_, c_len))
        split_parts.append("NP%d %.4f (%.2fx) %s" % (np_, t_s, t_s / t_vendor,
                                                     "P" if diff_s < 2e-2 else "F(%0.1e)" % diff_s))

    # split × 分桶惰性缩放（MXFA_M1_UM=2，binding 每次调用读 env，可热切换）
    os.environ["MXFA_M1_UM"] = "2"
    split_parts2 = []
    for np_ in np_list:
        c_len = ((sl_val + np_ - 1) // np_ + ROWS_BLK - 1) // ROWS_BLK * ROWS_BLK
        out_s = m1.paged_decode_split(q, k_cache, v_cache, bt, sl, scale, stream, np_, c_len).float()
        diff_s = (out_s - out_v).abs().max().item()
        t_s = measure_gpu(lambda: m1.paged_decode_split(q, k_cache, v_cache, bt, sl, scale, stream, np_, c_len))
        split_parts2.append("NP%d %.4f (%.2fx) %s" % (np_, t_s, t_s / t_vendor,
                                                      "P" if diff_s < 2e-2 else "F(%0.1e)" % diff_s))
    os.environ.pop("MXFA_M1_UM", None)  # 恢复 binding 默认（现为 2）

    if D != 128:
        v7s = "skip(v7 D=64 越界)"
    elif PPT * PS > 4096:
        v7s = "skip(v7 上限 4096 ctx)"  # 实测 PPT=512 时 v7 启动失败 invalid value
    else:
        t_v7 = measure_gpu(lambda: v7.paged_decode(q, k_cache, v_cache, bt, sl, scale, PPT * PS, stream))
        v7s = "%.4f" % t_v7

    print("[%s] B=%d sl=%d D=%d %s | vendor %.4f ms | v7 %s | "
          "v3-NS1 %.4f (%.2fx) %s | UM %.4f (%.2fx) %s | UM2 %.4f (%.2fx) %s | MF无数学 %.4f (%.2fx) | MF无shuffle %.4f (%.2fx) | MF无exp %.4f (%.2fx) | split %s | split2 %s"
          % (tag, B, sl_val, D, str(dt), t_vendor, v7s,
             t_v3, t_v3 / t_vendor, "P" if diff_3 < 2e-2 else "F(%0.1e)" % diff_3,
             t_um, t_um / t_vendor, "P" if diff_um < 2e-2 else "F(%0.1e)" % diff_um,
             t_um2, t_um2 / t_vendor, "P" if diff_um2 < 2e-2 else "F(%0.1e)" % diff_um2,
             t_mf, t_mf / t_vendor, t_mf2, t_mf2 / t_vendor,
             t_mf3, t_mf3 / t_vendor, " | ".join(split_parts), " ".join(split_parts2)), flush=True)

run("s1",     96, 8, 16, 128, 16, 1,  1, torch.float16)
run("s15",    96, 8, 16, 128, 16, 1,  15, torch.float16)
run("s16",   96, 8, 16, 128, 16, 8,  16, torch.float16)
run("s17",   96, 8, 16, 128, 16, 8,  17, torch.float16)
run("s32",   96, 8, 16, 128, 16, 8,  32, torch.float16)
run("s128",  96, 8, 16, 128, 16, 16, 128, torch.float16)
run("s512",  32, 8, 16, 128, 16, 32, 512, torch.float16)
run("s1024", 96, 8, 16, 128, 16, 64, 1024, torch.float16)
run("s2048", 96, 8, 16, 128, 16, 128, 2048, torch.float16)
run("lowB-s4096", 8, 2, 4, 128, 16, 256, 4096, torch.float16)
run("gq4",   96, 8, 32, 128, 16, 32, 512, torch.float16)
run("d64",   96, 8, 16, 64,  16, 32, 512, torch.float16)
# 溢出边界：q×400 → s_log2 峰值 ~200+，UM1(φ=0) 预期 F(inf)，UM2 分桶必须 P
run("ovf",    8, 2, 4, 128, 16, 32, 512, torch.float16, qmul=400.0)
# ---- 稳健性补测（收官前）----
run("s8192", 16, 2, 4, 128, 16, 512, 8192, torch.float16)   # 极长上下文，NP 扫到 64
run("gq3",   96, 8, 24, 128, 16, 32, 512, torch.float16)    # 非 2 幂 GQA：预期 F（NTH 整除性破坏）
run("gq6",   48, 8, 48, 128, 16, 32, 512, torch.float16)    # 偶数非 2 幂：预期 F（佐证 GQ∈{2,4,8} 守卫）
run("gq5",   48, 8, 40, 128, 16, 32, 512, torch.float16)
run("gq7",   32, 8, 56, 128, 16, 32, 512, torch.float16)
run("bf16",  96, 8, 16, 128, 16, 32, 512, torch.bfloat16)   # bf16 精度路径
print("M1-TEST-DONE", flush=True)
print("M1-TEST-DONE", flush=True)
