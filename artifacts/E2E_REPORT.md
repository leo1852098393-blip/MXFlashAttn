# v0.8.0 Qwen E2E Report

## Vendor baseline

`qwen-vendor-baseline.json` ran Qwen3-0.6B on the C500 with vLLM 0.17.0,
16 generated tokens and four fixed prompts. It records generated text,
prompt/generated token counts, wall latency, tokens/s and `mx-smi` memory.
vLLM 0.17.0 did not expose first-token timestamps, so TTFT is `null` and is
not inferred from wall latency.

## Candidate verification

The final candidate suite is `c500-qwen-candidate-suite.json` with its raw
dispatch stream in `c500-qwen-candidate-suite.dispatch.jsonl`. It ran four
fixed prompts for four generated tokens each on C500 and produced text for
all four requests. The result is marked:

```json
{"status": "candidate_dispatch_verified", "candidate_dispatch_verified": true}
```

The worker-side JSONL stream contains `mxflashattn_decoder` events with
`operation=flash_attn_with_kvcache` for every decode step. Prefill events are
separately labelled `vendor_prefill_or_blocked` because the adapter contract
currently accepts one-token decode only. The candidate bridge also patches the
MetaX/upstream vLLM ABI differences for FA version detection, KV cache update,
and varlen prefill calls; no PyTorch fallback was used in the decode events.

Candidate suite measurements are approximately 13.82, 45.35, 46.25 and 46.26
tokens/s (the first request includes one-time engine/prefill cost). vLLM 0.17.0
does not expose valid first-token timestamps, so TTFT remains `null`; these
numbers are an E2E dispatch proof, not a vendor-vs-candidate speedup claim.
The matched vendor/candidate model evidence and the 36-case operator matrix are
reported separately in `fair-matrix-c500.md`.
