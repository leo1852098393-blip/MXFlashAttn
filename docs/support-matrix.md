# Support Matrix

This v1.0.0 local candidate records evidence from the 64 GiB C500 instance
(32 GiB sGPU allocation) on 2026-10-05. Optimized execution uses the matching
MetaX `flash-attn` wheel; the in-repository C++ extension is an ATen
bring-up/correctness path, not a fused kernel authored by this project.

| Area | Status | Evidence |
|---|---|---|
| FP16/BF16 forward | Verified on C500 | `artifacts/fair-matrix-c500.json` |
| Head dimensions 64/128 | Verified on C500 | 36-case matrix |
| GQA/MQA and causal | Verified on C500 | 36-case matrix and API tests |
| Dense and varlen API | Verified | API tests and matrix |
| Paged KV cache API | Verified on C500 | decoder layout and matrix cases |
| vLLM 0.17.0 decoder candidate | Model generation verified with dispatch evidence | `c500-qwen-candidate-suite.dispatch.jsonl` |
| vLLM prefill | Delegated to MetaX vendor | E2E report |
| vLLM vendor/candidate comparison | 36 same-prompt model runs, kept separate from operator medians | `vllm-model-matrix-c500.md` |
| Torch SDPA baseline | Included in same-input operator matrix | `fair-matrix-c500.json` |
| TTFT | Not available from vLLM offline API | E2E report records null |
| Backward | Unsupported | Explicit validation error |
| FP8 | Unsupported in this candidate | Future milestone |
| Multi-card | Unsupported in this candidate | Future milestone |
| SGLang | Not tested in the C500 image | Future milestone |

## Performance wording

Across the 36 C500 operator cases, MXFlashAttn's median latency is 60.52%
lower than the project's PyTorch reference and 25.45% higher than the MetaX
vendor raw-call median (the v0.9 matrix recorded 32.74%). The first number is
a reference comparison; the second reflects Python wrapper overhead — the
MXFlashAttn API is currently slower than the raw vendor call on every
operator-level case. Model-level vLLM tokens/s
(eager 71.13 vs 79.48, graph 257.24–258.19 vs 274.09, both 36/36 text-identical)
are reported in a separate section because they do not share the operator
matrix's inputs.

## Fallback

Native execution is attempted only for supported FP16/BF16 accelerator tensors.
If no eligible provider is available, the API raises unless
`MXFLASHATTN_ALLOW_FALLBACK=1` is explicitly set. Fallback emits a warning and
records its reason through `get_last_dispatch_info()`. It is a correctness
reference, not a C500 performance baseline.
