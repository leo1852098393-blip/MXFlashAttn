"""M1 debug: 小样本逐元素对照。"""
import os, importlib.util
import numpy as np
import torch

SRC = "/root/MXFlashAttn_v1.3.0-kernel/csrc_native"
spec = importlib.util.spec_from_file_location("mxmmain", os.path.join(SRC, "mxmmain.so"))
m1 = importlib.util.module_from_spec(spec); spec.loader.exec_module(m1)

dev = "cuda"
torch.manual_seed(0)
B, HQ, HKV, D, PS = 1, 2, 1, 128, 16
q = torch.randn(B, 1, HQ, D, device=dev, dtype=torch.float16) * 0.5
k_cache = torch.randn(2, PS, HKV, D, device=dev, dtype=torch.float16) * 0.5
v_cache = torch.randn(2, PS, HKV, D, device=dev, dtype=torch.float16) * 0.5
bt = torch.tensor([[0, 1]], dtype=torch.int32, device=dev)
sl = torch.tensor([1], dtype=torch.int32, device=dev)
scale = D ** -0.5
stream = torch.cuda.current_stream().cuda_stream

out = m1.paged_decode(q, k_cache, v_cache, bt, sl, scale, 16, stream, 0).float()
torch.cuda.synchronize()

vrow = v_cache[0, 0, 0, :].float()   # position 0, kvh 0
print("sl=1, head0 (qh=0):", flush=True)
print("out[:6]  =", out[0, 0, 0, :6].tolist(), flush=True)
print("Vrow[:6] =", vrow[:6].tolist(), flush=True)

# 我们的输出最像哪个 V 行？
for h, kvh in ((0, 0), (1, 0)):
    o = out[0, 0, h, :]
    best = []
    for pg in range(2):
        for pos in range(PS):
            for kh in range(HKV):
                vv = v_cache[pg, pos, kh, :].float()
                best.append(((o - vv).abs().max().item(), pg, pos, kh))
    best.sort()
    print("head%d closest V row: diff=%.4f at page=%d pos=%d kvh=%d"
          % (h, best[0][0], best[0][1], best[0][2], best[0][3]), flush=True)

# sl=3 手工参考
sl3 = torch.tensor([3], dtype=torch.int32, device=dev)
out3 = m1.paged_decode(q, k_cache, v_cache, bt, sl3, scale, 16, stream, 0).float()
torch.cuda.synchronize()
K = k_cache[0, :3, 0, :].float()
V = v_cache[0, :3, 0, :].float()
for h in range(2):
    s = (q[0, 0, h, :].float() @ K.T) * scale
    p = torch.softmax(s, dim=-1)
    ref = p @ V
    print("sl=3 head%d: out=%s ref=%s" % (h, out3[0, 0, h, :4].tolist(), ref[:4].tolist()), flush=True)
print("DEBUG-DONE", flush=True)
