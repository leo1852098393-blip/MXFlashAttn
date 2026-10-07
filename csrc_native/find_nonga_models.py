#!/opt/conda/bin/python
# -*- coding: utf-8 -*-
"""Find GQ=6 with head_dim in {64,128}.

Key fix: many configs (Qwen2 family included) OMIT head_dim.  vLLM then
defaults to hidden_size // num_attention_heads, so derive it the same way
instead of requiring the field.
"""
import json, urllib.request, urllib.error

API = "https://www.modelscope.cn/api/v1/models/{}/repo?Revision=master&FilePath=config.json"

# (model_id, approx_download_gb) -- keep the GQ=6 hunt to things we can afford
MODELS = [
    # Qwen2 family: nQ/nKV = 40/8=5, 64/8=8, 72/8=9, 28/4=7, 48/4=12 ...
    "Qwen/Qwen2.5-14B-Instruct", "Qwen/Qwen2.5-32B-Instruct",
    "Qwen/Qwen2.5-72B-Instruct", "Qwen/Qwen2-57B-A14B",
    "Qwen/Qwen1.5-110B-Chat", "Qwen/Qwen1.5-72B-Chat",
    "Qwen/Qwen1.5-57B-Chat", "Qwen/Qwen-32B-Chat",
    # families with unusual ratios
    "baichuan-inc/Baichuan-13B-Chat", "baichuan-inc/Baichuan2-13B-Chat",
    "microsoft/Phi-3-mini-4k-instruct", "microsoft/Phi-3-medium-4k-instruct",
    "microsoft/Phi-4", "CohereForAI/aya-expanse-8b",
    "CohereForAI/aya-expanse-32b", "CohereForAI/aya-expanse-70b",
    "tiiuae/falcon3-7b-instruct", "tiiuae/falcon3-10b-instruct",
    "google/gemma-2-9b-it", "google/gemma-2-27b-it",
    "allenai/OLMo-2-1124-7B", "allenai/OLMo-2-1124-13B",
    "01-ai/Yi-6B-Chat", "01-ai/Yi-9B-Chat", "01-ai/Yi-34B-Chat",
    "moonshotai/Mixtral-8x7B-Instruct-v0.1",
    "Qwen/Qwen3-Next-80B-A3B-Instruct",
    "internlm/internlm3-8b-instruct", "internlm/internlm2_5-7b-chat",
    "THUDM/glm-4-9b-chat", "THUDM/glm-4-9b",
    "deepseek-ai/DeepSeek-V2-Lite-Chat",
    "mistralai/Mixtral-8x7B-Instruct-v0.1",
    "meta-llama/Meta-Llama-3-8B-Instruct",
    "meta-llama/Meta-Llama-3.1-8B-Instruct",
    "Qwen/Qwen3-30B-A3B", "Qwen/Qwen3-32B",
    "Qwen/Qwen3-235B-A22B",
]


def fetch(mid):
    url = API.format(mid)
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    try:
        with urllib.request.urlopen(req, timeout=25) as r:
            return json.loads(r.read().decode("utf-8", errors="replace"))
    except urllib.error.HTTPError as e:
        return {"__http__": e.code}
    except Exception as e:
        return {"__err__": type(e).__name__}


def shape_of(cfg, depth=0):
    if not isinstance(cfg, dict) or depth > 3:
        return None
    for key in ("text_config", "llm_config", "language_config"):
        sub = cfg.get(key)
        if isinstance(sub, dict):
            r = shape_of(sub, depth + 1)
            if r:
                return r
    nq = (cfg.get("num_attention_heads") or cfg.get("num_query_heads")
          or cfg.get("n_head") or cfg.get("num_heads"))
    nkv = cfg.get("num_key_value_heads")
    if nkv is None:
        nkv = cfg.get("num_kv_heads")
    if nkv is None:
        nkv = cfg.get("num_key_value_heads")
    if not nq or nkv is None:
        return None
    # head_dim: explicit, else the vLLM default hidden_size // nQ
    hd = cfg.get("head_dim")
    if hd is None:
        hd = cfg.get("qk_nope_head_dim") or cfg.get("attention_head_size")
    if hd is None:
        hs = cfg.get("hidden_size")
        if hs and nq:
            hd = hs // nq
    if not hd:
        return None
    gq = f"{nq}/{nkv}" if (nkv == 0 or nq % nkv) else nq // nkv
    return (nq, nkv, gq, hd)


print("%-44s %4s %4s %8s %5s  %s" % ("model", "nQ", "nKV", "GQ", "HD", "verdict"))
print("-" * 112)
targets = []
for mid in MODELS:
    d = fetch(mid)
    if "__http__" in d:
        print("%-44s  HTTP %s" % (mid[:44], d["__http__"]))
        continue
    if "__err__" in d:
        print("%-44s  ERR %s" % (mid[:44], d["__err__"]))
        continue
    s = shape_of(d)
    if not s:
        print("%-44s  (no GQA shape fields)" % mid[:44])
        continue
    nq, nkv, gq, hd = s
    if gq == 6:
        v = "*** GQ=6 TARGET ***" if hd in (64, 128) else "GQ=6 but HD=%s (API rejects first)" % hd
        if hd in (64, 128):
            targets.append(mid)
    elif hd not in (64, 128):
        v = "HD=%s out of support" % hd
    elif gq in (2, 4, 8):
        v = "already-covered shape"
    else:
        v = "non-pow2 GQ (guard-reject evidence)"
    print("%-44s %4s %4s %8s %5s  %s" % (mid[:44], nq, nkv, gq, hd, v))

print()
if targets:
    print("=== GQ=6 + HD in {64,128} TARGETS ===")
    for t in targets:
        print("  ", t)
else:
    print("=== none found in this candidate list ===")
