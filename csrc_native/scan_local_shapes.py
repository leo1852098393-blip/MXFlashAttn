#!/opt/conda/bin/python
# Scan the local shared model library: compute (GQ, HD) for every config.json.
# Faster and more reliable than probing remote model ids.
import json, os, glob, sys

ROOTS = ["/mnt/moark-models", "/data/models"]
rows = []
for root in ROOTS:
    for cfg in glob.glob(os.path.join(root, "**", "config.json"), recursive=True):
        try:
            c = json.load(open(cfg))
        except Exception:
            continue
        if not isinstance(c, dict):
            continue
        arch = (c.get("architectures") or ["?"])[0]
        q = c.get("num_attention_heads") or c.get("n_head") or c.get("n_heads")
        kv = (c.get("num_key_value_heads") or c.get("num_kv_heads")
              or c.get("n_head_kv") or c.get("multi_query_group_num"))
        hd = c.get("head_dim")
        if hd is None and q:
            hs = c.get("hidden_size") or c.get("n_embd") or c.get("d_model")
            if hs:
                hd = hs // q
        gq = (q // kv) if (q and kv) else (1 if q else None)
        if not gq or not hd:
            continue
        rows.append((gq, hd, arch, c.get("num_hidden_layers"), cfg))

rows.sort(key=lambda r: (r[1], r[0]))
# Summary: which (GQ,HD) cells are covered
cells = {}
for gq, hd, arch, L, cfg in rows:
    cells.setdefault((gq, hd), []).append((arch, L, cfg))

print("=" * 96)
print("COVERED (GQ,HD) CELLS IN LOCAL LIBRARY")
print("=" * 96)
for (gq, hd) in sorted(cells, key=lambda k: (k[1], k[0])):
    items = cells[(gq, hd)]
    sup = "SUPPORTED" if (hd in (64, 128) and gq in (1, 2, 4, 8)) else "rejected-by-guard"
    print(f"  GQ={gq:<3} HD={hd:<4}  [{sup}]  n={len(items)}   e.g. {items[0][0]}")

print()
print("=" * 96)
print("WANTED: HD==64 WITH GQ IN {2,4,8}   (the last uncovered supported cell)")
print("=" * 96)
hit = False
for gq, hd, arch, L, cfg in rows:
    if hd == 64 and gq in (2, 4, 8):
        hit = True
        print(f"  *** {cfg}\n      arch={arch} L={L} GQ={gq} HD={hd}")
if not hit:
    print("  (none found in local library)")

print()
print("=" * 96)
print("ALL MODELS WITH HD==64  (any GQ)")
print("=" * 96)
for gq, hd, arch, L, cfg in rows:
    if hd == 64:
        mark = "OK " if gq in (2, 4, 8) else "GQ-GUARD-REJECT"
        print(f"  [{mark}] GQ={gq:<3} L={str(L):<4} {arch:<28} {cfg}")
