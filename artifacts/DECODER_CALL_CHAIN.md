# vLLM 0.17.0 Decoder Call Chain

## Source evidence

The C500 environment was inspected with:

```text
env MACA_PATH=/opt/maca-3.5.3 /opt/conda/bin/python - <<'PY'
import inspect
from vllm.v1.attention.backends.flash_attn import FlashAttentionImpl, FlashAttentionMetadata
print(inspect.getsource(FlashAttentionImpl))
print(inspect.getsource(FlashAttentionMetadata))
PY
```

The installed implementation defines:

- `query`: `[num_tokens, num_heads, head_size]`
- `key` and `value`: `[num_tokens, num_kv_heads, head_size]`
- `kv_cache`: `[2, num_blocks, block_size, num_kv_heads, head_size]`
- `query_start_loc`: cumulative query offsets
- `seq_lens`: one cached sequence length per request
- `block_table`: physical page indices, one row per request
- `slot_mapping`: cache update slots
- `scheduler_metadata`: optional scheduler side data
- `use_cascade`, `max_num_splits`, and `causal`: execution controls

For the ordinary decoder path, vLLM unbinds the cache into key/value pages and
passes `query_start_loc`, `seq_lens`, `block_table` and `scheduler_metadata` to
its varlen FlashAttention implementation. The v0.7 adapter covers the narrower
case where every query length is one, cascade is disabled, and the cache has
already been updated by vLLM.

## Adapter evidence

`artifacts/c500-decoder-layout.json` was produced on a C500 with:

```text
torch 2.8.0+metax3.5.3.9
vLLM 0.17.0
flash-attn 2.6.3+metax3.5.3.9torch2.8
vllm_metax 0.17.0+gd10261.d20260409.maca3.5.3.20.torch2.8
MXFlashAttn dispatch: mxmac_flash_attn
operation: flash_attn_with_kvcache
fallback_reason: null
```

The probe used two requests, query lengths `[1, 1]`, sequence lengths `[17,
16]`, a `[2, 4, 16, 2, 64]` cache and a `[2, 2]` block table. It produced an
output of `[2, 256]` without fallback.

## Boundary

This is a tensor-contract and native dispatch result, not a model generation
result. The adapter is not registered as a vLLM backend, does not cover
prefill or cascade scheduling, and has not yet been inserted into Qwen's
generation loop. v0.8 must prove that insertion before any end-to-end claim.
