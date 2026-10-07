import os, sys
import torch
from vllm_metax.v1.attention.backends.fa_utils import flash_attn_with_kvcache

torch.manual_seed(20261007)
B, HKV, GQ, HQ, D, PS, PPT = 2, 2, 2, 4, 128, 16, 16
PAGES = B * PPT
X = 8                       # fp16: head_dim split factor used by FA paged layout
sl = [200, 256]
dev = "cuda"
dt = torch.float16

def step(tag):
    torch.cuda.synchronize()
    print("[step]", tag, flush=True)

step("alloc q")
q = (torch.randn(B, 1, HQ, D, device=dev, dtype=dt) * 0.5)
step("alloc caches (metax paged layout)")
# error message: kcache must have shape (num_blocks, page_block_size, num_heads_k, head_size_og)
k_cache = torch.randn(PAGES, PS, HKV, D, device=dev, dtype=dt) * 0.5
v_cache = torch.randn(PAGES, PS, HKV, D, device=dev, dtype=dt)
step("block table")
perm = torch.randperm(PAGES, device=dev)
bt = perm[: B * PPT].reshape(B, PPT).to(torch.int32)
cache_seqlens = torch.tensor(sl, dtype=torch.int32, device=dev)
scale = D ** -0.5
step("call vendor")
out = flash_attn_with_kvcache(
    q=q, k_cache=k_cache, v_cache=v_cache,
    block_table=bt, cache_seqlens=cache_seqlens,
    softmax_scale=scale, causal=True,
)
if isinstance(out, tuple):
    print("out is tuple:", [tuple(t.shape) if hasattr(t, 'shape') else t for t in out], flush=True)
    out = out[0]
step("vendor returned: %s %s" % (tuple(out.shape), out.dtype))
for name, t in [("q", q), ("k", k_cache), ("v", v_cache), ("bt", bt),
                ("sl", cache_seqlens), ("out", out)]:
    t.detach().cpu().numpy().tofile("vendor_%s.bin" % name)
    step("dumped " + name)
print("DUMPED", flush=True)
