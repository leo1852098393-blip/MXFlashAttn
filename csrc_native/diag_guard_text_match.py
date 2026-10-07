#!/opt/conda/bin/python
# -*- coding: utf-8 -*-
"""Is text_match 1/4 real divergence, or an artifact?

M1 never fires on GQ=5 (0 dispatches) -- every call raises and falls back to
the v7 extension.  So M1-UM1 and M1-UM2 should produce text IDENTICAL to v7.
1/4 means three prompts differed.  Two possible causes:

  C1  Real: the raise/fallback path perturbs state (e.g. the plugin writes
      into `out` before raising, or the exception unwinds past a partial
      write).  Then M1 output would be genuinely wrong -> a correctness bug.
  C2  Scheduling: greedy decode is deterministic per request, but continuous
      batching makes the batch composition depend on timing.  If M1 is slower
      (extra exception cost), requests finish at different times, batches
      regroup, and a borderline token flips.  -> not a kernel bug.

Distinguish: re-run the SAME config twice.  If v7 vs v7 also differs, it is C2.
Also compare token-id sequences, not just text, and check whether the M1 run
ever wrote into the output buffer.
"""
import os, sys, json, time

os.environ.setdefault("MACA_PATH", "/opt/maca")
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
os.environ["MXFLASHATTN_GRAPH_OFFICIAL_SPLIT"] = "1"
os.environ["MXFA_NATIVE_DECODE"] = "1"
os.environ["MXFLASHATTN_AUTOREGISTER"] = "1"
os.environ["MXFLASHATTN_DISPATCH_LOG"] = "/root/diag_guard_dispatch.jsonl"
sys.path.insert(0, "/root/MXFlashAttn_v1.3.0-kernel")
os.chdir("/root/MXFlashAttn_v1.3.0-kernel")

from vllm import LLM, SamplingParams

MODEL = "/data/models/Qwen2.5-14B-Instruct"
engine = LLM(model=MODEL, enforce_eager=True, max_model_len=2048)

PROMPTS = [
    "Explain paged KV cache in one paragraph.",
    "Write a Python function that reverses a linked list.",
    "What is the capital of France?",
    "Summarise why GPUs prefer matrix multiplication.",
]
SP = SamplingParams(temperature=0.0, max_tokens=48)


def run(tag):
    t0 = time.perf_counter()
    out = engine.generate(PROMPTS, SP)
    wall = time.perf_counter() - t0
    ids = [tuple(o.outputs[0].token_ids) for o in out]
    txt = [o.outputs[0].text for o in out]
    print("[%s] wall=%.2fs" % (tag, wall))
    for i, (a, b) in enumerate(zip(ids, txt)):
        print("   p%d ntok=%3d  %r" % (i, len(a), b[:70].replace("\n", " ")))
    return ids, txt


def compare(a, b):
    n = sum(x == y for x, y in zip(a, b))
    print("   -> match %d/%d" % (n, len(a)))
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            d = next((k for k, (p, q) in enumerate(zip(x, y)) if p != q), min(len(x), len(y)))
            print("      p%d first divergence at token %d (len %d vs %d)" % (i, d, len(x), len(y)))
    return n


print("=" * 96)
print("A) v7 run twice, identical config -- is the harness itself deterministic?")
print("=" * 96)
os.environ["MXFA_M1_DECODE"] = "0"
v7a_ids, v7a_txt = run("v7 pass1")
print()
v7b_ids, v7b_txt = run("v7 pass2")
print("\ncompare v7 pass1 vs v7 pass2 (same config):")
n_v7 = compare(v7a_ids, v7b_ids)

print()
print("=" * 96)
print("B) M1 config (guard rejects, falls back) vs v7")
print("=" * 96)
os.environ["MXFA_M1_DECODE"] = "1"
os.environ["MXFA_M1_UM"] = "2"
m1_ids, m1_txt = run("m1_um2")
print("\ncompare v7 pass1 vs m1_um2:")
n_m1 = compare(v7a_ids, m1_ids)

print()
print("=" * 96)
print("SUMMARY")
print("=" * 96)
print("v7 vs v7 (harness determinism) : %d/4" % n_v7)
print("v7 vs M1 (guard fallback)     : %d/4" % n_m1)
print()
if n_v7 < 4:
    print("VERDICT: the harness itself is non-deterministic at this batch size --")
    print("         the 1/4 text_match is a batching artifact, NOT a kernel bug.")
elif n_m1 < 4:
    print("VERDICT: harness is deterministic, so M1's raise/fallback path DOES")
    print("         change output -> investigate as a correctness issue.")
else:
    print("VERDICT: both deterministic and identical -- 1/4 was a one-off.")
