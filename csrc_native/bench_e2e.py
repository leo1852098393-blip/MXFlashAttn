"""批次并发 E2E 基准：一次 generate 提交全部 prompt（vLLM continuous batching）。

对比三档 decode 路径（env 热切换，无需重启引擎）：
  v7      : MXFA_M1_DECODE=0
  M1-UM   : MXFA_M1_DECODE=1, MXFA_M1_UM=1
  M1-UM2  : MXFA_M1_DECODE=1, MXFA_M1_UM=2（分桶惰性缩放）

用法（树根目录）：
  MXFLASHATTN_GRAPH_OFFICIAL_SPLIT=1 MXFA_NATIVE_DECODE=1 python bench_e2e.py \
      --output artifacts/bench-e2e.json
"""
import argparse
import json
import os
import time
from collections import Counter
from pathlib import Path

# 必须在 import vllm 前：强制 EngineCore 同进程跑，否则 os.environ 热切换
# 传不进子进程（EngineCore_DP0），三档对比会全部退化成继承的启动时环境。
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

parser = argparse.ArgumentParser()
parser.add_argument("--model", default=None)
parser.add_argument("--batches", type=int, nargs="+", default=[128, 64],
                    help="每个配置测的批大小（prompt 数）")
parser.add_argument("--max-tokens", type=int, default=256)
parser.add_argument("--max-model-len", type=int, default=2048,
                    help="引擎上下文窗口；长序列跑 2048 生成时需 4096")
parser.add_argument("--output", type=Path, default=Path("artifacts/bench-e2e.json"))
parser.add_argument("--trust-remote-code", action="store_true",
                    help="允许加载含 auto_map 自定义代码的模型仓库"
                         "（如 PaddleOCR-VL；vLLM 无对应环境变量，必须走构造参数）")
args = parser.parse_args()

import torch
from integrations.vllm.config import QUICK_MODEL
from integrations.vllm import backend as dispatch_module

MODEL = args.model or QUICK_MODEL
reg = dispatch_module.register_vllm_v1_backend()
assert reg.get("registered") == "true", reg
os.environ["MXFLASHATTN_AUTOREGISTER"] = "1"
dispatch_log = args.output.with_suffix(".dispatch.jsonl").resolve()
os.environ["MXFLASHATTN_DISPATCH_LOG"] = str(dispatch_log)

from vllm import LLM, SamplingParams

_llm_kwargs = dict(model=MODEL, enforce_eager=True, max_model_len=args.max_model_len)
if args.trust_remote_code:
    _llm_kwargs["trust_remote_code"] = True
engine = LLM(**_llm_kwargs)
TEMPLATES = [
    "Explain paged KV cache in one sentence.",
    "List three practical benefits of grouped-query attention.",
    "Give a concise checklist for validating an attention kernel.",
    "Describe how causal masking changes autoregressive decoding.",
    "Explain why variable-length attention needs cumulative offsets.",
    "Compare MQA and GQA for inference memory usage.",
    "Name two checks for a causal attention implementation.",
    "Explain the role of a block table in paged attention.",
    "Give one reason BF16 may differ slightly from FP16.",
    "Describe a safe fallback policy for an accelerator kernel.",
    "Explain why a vendor baseline must be measured separately.",
    "List the metadata needed by a decoder attention adapter.",
]


def make_prompts(n):
    return [f"{TEMPLATES[i % len(TEMPLATES)]} Case {i:04d}." for i in range(n)]


def dispatch_snapshot():
    if not dispatch_log.exists():
        return {}
    lines = [l for l in dispatch_log.read_text(encoding="utf-8").splitlines() if l]
    return Counter(json.loads(l).get("path", "?") for l in lines)


def timed_run(n, max_tokens):
    prompts = make_prompts(n)
    dispatch_module.clear_dispatch_events(truncate_log=True)
    before = dispatch_snapshot()
    sp = SamplingParams(temperature=0.0, max_tokens=max_tokens)
    t0 = time.perf_counter()
    outputs = engine.generate(prompts, sp)
    wall = time.perf_counter() - t0
    total = sum(len(o.outputs[0].token_ids) for o in outputs)
    after = dispatch_snapshot()
    delta = Counter({k: after.get(k, 0) - before.get(k, 0) for k in set(after) | set(before)})
    return {"batch": n, "max_tokens": max_tokens, "wall_s": round(wall, 3),
            "gen_tokens": total, "tokens_per_s": round(total / wall, 2),
            "dispatch_delta": dict(delta)}


results = {}
# 预热（编译/显存池/调度器稳定）
os.environ["MXFA_M1_DECODE"] = "0"
engine.generate(make_prompts(8), SamplingParams(temperature=0.0, max_tokens=8))

for label, m1, um in (("v7", "0", None), ("m1_um", "1", "1"), ("m1_um2", "1", "2")):
    os.environ["MXFA_M1_DECODE"] = m1
    if um is not None:
        os.environ["MXFA_M1_UM"] = um
    runs = []
    for n in args.batches:
        r = timed_run(n, args.max_tokens)
        runs.append(r)
        print(f"[{label}] batch={r['batch']} tps={r['tokens_per_s']} "
              f"({r['gen_tokens']} tok / {r['wall_s']}s) disp={r['dispatch_delta']}", flush=True)
    results[label] = runs

os.environ["MXFA_M1_DECODE"] = "0"

# 一致性抽查：同一 prompt 三路径的贪心输出（批内取 4 条）
os.environ["MXFA_M1_DECODE"] = "0"
ref = engine.generate(make_prompts(4), SamplingParams(temperature=0.0, max_tokens=48))
ref_texts = [o.outputs[0].text for o in ref]
for label, m1, um in (("m1_um", "1", "1"), ("m1_um2", "1", "2")):
    os.environ["MXFA_M1_DECODE"] = m1
    if um is not None:
        os.environ["MXFA_M1_UM"] = um
    out = engine.generate(make_prompts(4), SamplingParams(temperature=0.0, max_tokens=48))
    match = sum(a == b for a, b in zip(ref_texts, (o.outputs[0].text for o in out)))
    results.setdefault("text_match", {})[label] = f"{match}/4"
    print(f"[match] {label}: {match}/4", flush=True)
os.environ["MXFA_M1_DECODE"] = "0"

args.output.parent.mkdir(parents=True, exist_ok=True)
args.output.write_text(json.dumps({
    "kind": "e2e_batch_bench", "model": MODEL, "max_tokens": args.max_tokens,
    "max_model_len": args.max_model_len, "registration": reg, "results": results}, indent=2), encoding="utf-8")
print(f"Wrote {args.output}", flush=True)
