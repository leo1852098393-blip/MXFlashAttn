# CHANGELOG

## v1.2.0 (2026-10-06)

- Fixed a silent import fallback in the vLLM 0.17 plugin: the MetaX vendor backend
  class is named `MacaFlashAttentionBackend`, so importing `FlashAttentionBackend`
  from `vllm_metax` raised ImportError and the plugin silently fell back to the
  upstream vLLM FA implementation (`impl_module=vllm.v1.attention.backends.flash_attn`).
  The plugin now aliases the vendor class explicitly.
- After the fix, model-level acceptance on C500 (Qwen3-0.6B, 36 fixed prompts,
  max_tokens=16) passed 36/36 text-identical vs the vendor baseline, verified by a
  double rerun: eager 71.13 vs 79.48 tokens/s (10.5% slower) and CUDA Graph
  257.24–258.19 vs 274.09 tokens/s (about 6% slower). Decode dispatch stays on
  `vendor_direct` (840/840 samples); prefill remains delegated to the vendor.
- Graph-safe `_decode_forward_graph`, transparent prefill fallback,
  `seq_lens.reshape(-1)` shape normalization, worker dispatch JSONL persistence,
  dispatcher fallback-reason fixes, and 43 regression tests.

## v1.1.0-vendor-wiring

- Added explicit `vendor_direct` dispatch for the MetaX flash-attn wheel.
- Added `mxmac_aten_correctness` as the named ATen extension fallback.
- Vendor call and ATen fallback failures retain their reasons in `DispatchInfo` and vLLM dispatch JSONL.
- Kept v1.0 evidence as provenance; the 36-case model candidate remains slower than vendor and must be rerun on C500 after this wiring change.
