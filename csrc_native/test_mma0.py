"""M0 布局探针：64 变体 vs 4 参考自动匹配。"""
import os, importlib.util
import numpy as np
import torch

SRC = "/root/MXFlashAttn_v1.3.0-kernel/csrc_native"
spec = importlib.util.spec_from_file_location("mxmma0", os.path.join(SRC, "mxmma0.so"))
ext = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ext)
print("[m0-probe] loaded", flush=True)

torch.manual_seed(7)
A = (torch.randn(16, 16, device="cuda", dtype=torch.float16) * 0.5)
B = (torch.randn(16, 16, device="cuda", dtype=torch.float16) * 0.5)
D = torch.zeros(32 * 512, device="cuda", dtype=torch.float32)
WS = torch.zeros(1, dtype=torch.int32, device="cuda")

ext.probe(A.data_ptr(), B.data_ptr(), D.data_ptr(), WS.data_ptr(),
          torch.cuda.current_stream().cuda_stream)
torch.cuda.synchronize()

print("warpSize =", WS.cpu().item(), flush=True)

An = A.cpu().numpy().astype(np.float64)
Bn = B.cpu().numpy().astype(np.float64)
refs = {"A@B": An @ Bn, "A@Bt": An @ Bn.T, "At@B": An.T @ Bn, "At@Bt": An.T @ Bn.T}

Dn = D.cpu().numpy().astype(np.float64).reshape(32, 512)
best = []
for ai in range(2):
    for bi in range(2):
        for pa in range(2):
            for pb in range(2):
                for sw in range(2):
                    idx = ((((ai * 2 + bi) * 2 + pa) * 2 + pb) * 2 + sw)
                    for oi, tag in ((0, "D[4q+j][c]"), (1, "D[c][4q+j]")):
                        out = Dn[idx, oi * 256:(oi + 1) * 256].reshape(16, 16)
                        for rn, ref in refs.items():
                            diff = np.abs(out - ref).max()
                            best.append((diff, ai, bi, pa, pb, sw, oi, rn))

best.sort(key=lambda x: x[0])
print("--- top 8 ---", flush=True)
for diff, ai, bi, pa, pb, sw, oi, rn in best[:8]:
    print("diff=%.3e  aMode=%d bMode=%d pkA=%d pkB=%d argSwap=%d out=%s ref=%s"
          % (diff, ai, bi, pa, pb, sw, oi, rn), flush=True)

diff, ai, bi, pa, pb, sw, oi, rn = best[0]
ok = diff < 2e-2
print("M0-PROBE %s (best=%.3e, aMode=%d bMode=%d pkA=%d pkB=%d argSwap=%d out=%d ref=%s)"
      % ("PASSED" if ok else "FAILED", diff, ai, bi, pa, pb, sw, oi, rn), flush=True)
