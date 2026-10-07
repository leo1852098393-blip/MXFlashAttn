#!/opt/conda/bin/python
# -*- coding: utf-8 -*-
"""Scan every model config on the instance for GQ=6 with HD in {64,128}.

GQ=6 candidates found earlier (Qwen3.5/3.8-27B) both carry head_dim=256, which
the API layer rejects BEFORE the binding-level GQ guard can be exercised.  So
neither can isolate GQ.  Sweep all configs and report (GQ, HD) for every
full-attention layer we can identify.
"""
import json, os, glob

ROOTS = ["/mnt/moark-models", "/data/models"]

rows = []
for root in ROOTS:
    if not os.path.isdir(root):
        continue
    for cfg in glob.glob(os.path.join(root, "*", "config.json")):
        name = os.path.basename(os.path.dirname(cfg))
        try:
            d = json.load(open(cfg, encoding="utf-8"))
        except Exception:
            continue
        # normalise: full-attention GQA shape may live under text_config or at top
        tcs = []
        if isinstance(d.get("text_config"), dict):
            tcs.append(("text_config", d["text_config"]))
        if isinstance(d.get("llm_config"), dict):
            tcs.append(("llm_config", d["llm_config"]))
        if isinstance(d.get("language_config"), dict):
            tcs.append(("language_config", d["language_config"]))
        tcs.append(("top", d))

        seen = set()
        for where, tc in tcs:
            nq = tc.get("num_attention_heads") or tc.get("num_query_heads") \
                or tc.get("n_head") or tc.get("num_heads")
            nkv = tc.get("num_key_value_heads") or tc.get("num_kv_heads") \
                or tc.get("n_head_kv")
            hd = tc.get("head_dim") or tc.get("qk_nope_head_dim") \
                or tc.get("attention_head_size")
            if not nq or not hd:
                continue
            if nkv is None:
                continue  # MHA: no GQA ratio
            if nkv == 0:
                continue
            key = (where, nq, nkv, hd)
            if key in seen:
                continue
            seen.add(key)
            if nkv == 0 or nq % nkv != 0:
                gq = f"nq={nq}/nkv={nkv} (non-integer)"
            else:
                gq = nq // nkv
            rows.append((name, where, nq, nkv, gq, hd))

# report: GQ=6 first, then anything with HD in the supported set
print("%-34s %-16s %5s %5s %6s %5s  %s" %
      ("model", "cfg", "nQ", "nKV", "GQ", "HD", "verdict"))
print("-" * 104)

def verdict(gq, hd):
    hd_ok = hd in (64, 128)
    if gq == 6:
        return "*** GQ=6 TARGET ***" if hd_ok else "GQ=6 but HD unsupported -> API rejects first"
    if not hd_ok:
        return "HD out of support set"
    return "supported shape" if gq in (2, 4, 8) else "non-pow2 GQ (guard reject)"

hits = [r for r in rows if r[4] == 6]
others = [r for r in rows if r[4] != 6]

for r in sorted(hits, key=lambda x: (x[5] not in (64, 128), str(x[0]))):
    print("%-34s %-16s %5d %5d %6s %5s  %s" % (r[0][:34], r[1], r[2], r[3], r[4], r[5], verdict(r[4], r[5])))

print()
print("=== all other GQA ratios with HD in {64,128} ===")
for r in sorted(others, key=lambda x: (str(x[4]), str(x[0]))):
    if r[5] in (64, 128):
        print("%-34s %-16s %5d %5d %6s %5s  %s" %
              (r[0][:34], r[1], r[2], r[3], r[4], r[5], verdict(r[4], r[5])))

print()
print("summary: GQ=6 rows=%d, of which HD in {64,128}: %d"
      % (len(hits), sum(1 for r in hits if r[5] in (64, 128))))
