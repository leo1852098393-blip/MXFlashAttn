# vLLM Integration Status

The release candidate targets the checked C500 environment: `vllm 0.17.0` and `vllm_metax 0.17.0+gd10261.d20260409.maca3.5.3.20.torch2.8`. SGLang is not installed in the image, so the alternate integration path remains unevaluated. [`backend.py`](backend.py) contains a callable bridge and an isolated vLLM v1 registration probe for dense `[tokens, heads, head_dim]` Q/K/V tensors.

A full integration must target one explicitly pinned and C500-compatible vLLM version. The local v0.5.0 probe reached the vLLM 0.17.0 v1 registration hook, and the candidate encoder-shaped path can call the public MXFlashAttn API. Decoder paged-cache execution remains delegated to vLLM because metadata construction, slot-based KV writes, block-table reads, cascade handling, and the version's `AttentionImpl` contract still require validation. No end-to-end MXFlashAttn acceleration claim is made.

The generation script only collects a baseline by default when `--baseline-only` is passed. It refuses to run without that flag because decoder paged-cache execution has not been switched to MXFlashAttn; baseline output explicitly says that it did not exercise this project. The configured quick model is Qwen3-0.6B for the tested vLLM 0.17.0 environment; weights remain outside this repository.
