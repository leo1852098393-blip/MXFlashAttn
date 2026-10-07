#!/bin/bash
# scan shared model library for attention shapes; flag GQ=8 and non-2-pow GQ
for d in /mnt/moark-models/*/; do
  name=$(basename "$d")
  cfg="$d/config.json"
  [ -f "$cfg" ] || continue
  /opt/conda/bin/python - "$cfg" "$name" <<'EOF'
import json, sys
p, name = sys.argv[1], sys.argv[2]
try:
    c = json.load(open(p))
except Exception as e:
    print(f"{name}: config unreadable {e}"); sys.exit()
q = c.get("num_attention_heads") or c.get("n_head")
kv = c.get("num_key_value_heads") or c.get("num_kv_heads") or c.get("n_head_kv")
if not q:
    # multimodal wrapper: dig into text_config
    tc = c.get("text_config") or {}
    if tc:
        q = tc.get("num_attention_heads"); kv = tc.get("num_key_value_heads")
        hd = tc.get("head_dim") or (tc.get("hidden_size",0)//(q or 1))
        gq = (q//kv) if (q and kv) else None
        print(f"{name}: [text_config] layers={tc.get('num_hidden_layers')} Q={q} KV={kv} GQ={gq} HD={hd}")
        sys.exit()
    print(f"{name}: no head fields (arch={c.get('architectures')})"); sys.exit()
hd = c.get("head_dim") or (c.get("hidden_size",0)//q)
gq = (q//kv) if kv else 1
tag = ""
if gq == 8: tag = "   <<<<<< GQ=8"
elif kv and (gq & (gq-1)) != 0: tag = "   <<<<<< NON-2-POW GQ (guard must reject)"
elif kv == 0: tag = "   <<<<<< MHA (GQ=1)"
print(f"{name}: arch={(c.get('architectures') or ['?'])[0]} layers={c.get('num_hidden_layers')} "
      f"Q={q} KV={kv} GQ={gq} HD={hd}{tag}")
EOF
done