import os, time, importlib.util
import torch

spec = importlib.util.spec_from_file_location(
    "wl", "/root/MXFlashAttn_v1.3.0-kernel/csrc_native/wl.so")
wl = importlib.util.module_from_spec(spec)
spec.loader.exec_module(wl)

dev = "cuda"
stream = torch.cuda.current_stream().cuda_stream

in_t = torch.arange(64, device=dev, dtype=torch.float32)
oB, oU1, oU4, oU8, oScan = wl.probe(in_t, stream)

print("lane: ", [int(i) for i in in_t[:20].tolist()])
print("B(3): ", [int(i) for i in oB[:20].tolist()])
print("U1:  ", [int(i) for i in oU1[:20].tolist()])
print("U4:  ", [int(i) for i in oU4[:20].tolist()])
print("U8:  ", [int(i) for i in oU8[:20].tolist()])

# scan 归约：每组 16 个 (base..base+15) 求和 = 16*base + 120
ok = True
for g in range(4):
    exp = 16 * g * 16 + 120
    got = oScan[g * 16:(g + 1) * 16]
    if not torch.allclose(got, torch.full((16,), float(exp), device=dev)):
        ok = False
        print("SCAN-FAIL group", g, "exp", exp, "got", got.tolist())
print("SCAN", "PASS" if ok else "FAIL")

# 成本：单 wave，64 线程
def bench(variant, iters):
    fn = lambda: wl.cost(in_t, iters, variant, stream)
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(10):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / 10 / iters * 1e9  # ns/iter

t0, t1, t2 = bench(0, 200000), bench(1, 200000), bench(2, 200000)
print("COST bsm-xor(4 shfl): %.1f ns/iter" % t0)
print("COST scan+bcast(5 mov_shfl): %.1f ns/iter" % t1)
print("COST noop: %.1f ns/iter" % t2)
print("WL-DONE", flush=True)
