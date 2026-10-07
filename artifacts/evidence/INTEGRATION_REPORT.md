# v0.5.0 Integration Report

## vLLM 0.17.0 API evidence

The 64G C500 environment reports vLLM `0.17.0`. The legacy
`vllm.attention.backends` package does not exist. The available extension point
is the v1 package:

- `vllm.v1.attention.backend.AttentionBackend`
- `vllm.v1.attention.backend.AttentionImpl`
- `vllm.v1.attention.backends.registry.register_backend`
- `vllm.v1.attention.backends.flash_attn.FlashAttentionBackend`
- `FlashAttentionImpl.forward(query, key, value, kv_cache, attn_metadata, output, ...)`

The raw probe is retained in `vllm-0.17-api-probe.json`.

## Registration result

`vllm-0.17-registration.json` records a successful isolated registration of the
`MXFLASHATTN_V017` override. The adapter uses the public MXFlashAttn API for
encoder-shaped tensors. Decoder and paged-cache calls deliberately delegate to
vLLM's original FlashAttention implementation because the scheduler metadata,
cache block layout, cascade path, and quantized-cache branches still require
dedicated validation.

This proves that the vLLM registration hook is reachable. It does not prove
end-to-end MXFlashAttn acceleration in vLLM, and no such claim is made here.

## Alternative path

The C500 image did not expose an installed SGLang package through its standard
Python environment. The independent `attention_forward` bridge remains the
stable fallback for integration testing while the decoder cache adapter is
validated in a later iteration.
