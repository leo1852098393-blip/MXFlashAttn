# Support Matrix

This v1.4.0 C500 release candidate records validation completed through 2026-10-17,
with the 2026-10-07 Gate 4 and 2026-10-13 retest material retained as historical
provenance and application evidence.
The in-repository `csrc/mxmac` extension remains an ATen bring-up/correctness path;
the in-house device kernel is `csrc_native/m1_kernel.cpp`.

| Area | Status | Evidence |
|---|---|---|
| FP16/BF16 forward | Verified on C500 | Gate 4 reports and operator evidence |
| M1 head dimensions 64/128 | Verified on C500 | Real-model E2E evidence |
| M1 GQA ratios 2/4/8 | Verified on C500 | Six supported shape cells |
| M1 GQA ratios 5/7 | Must not enter M1; upper layer fallback | Historical guard evidence |
| Other unverified GQA ratios | No support commitment | Contract boundary |
| Head dimension >128 in vLLM official split | Falls back to original MetaX/vendor KV-cache path | Local submission evidence `02_evidence/c500_20261013/e2e-hd256-guardfix.json` |
| Head dimension >128 via public MXFlashAttn API | Explicit `ValueError`; no silent computation | API guard |
| Dense and varlen API | Verified | API tests and matrix |
| Paged KV cache API | Verified on C500 | vLLM E2E evidence |
| vLLM 0.17.0 M1 decoder | Verified with `m1_split_decode` dispatch events | Gate 4 E2E evidence |
| vLLM prefill | Delegated to MetaX vendor | E2E report |
| M1 vs v7 E2E | Qwen3-4B/8B, PaddleOCR-VL, TinyLlama and 0.6B | `docs/gate4/E2E_大模型与守卫验证.md` |
| Torch SDPA baseline | Included in same-input operator matrix | `fair-matrix-c500.json` |
| TTFT/P95 | Streaming harness verified; candidate had zero M1 dispatch events, so no M1 performance claim | `02_evidence/.../streaming/` |
| Near-16K real prompt | Qwen3-4B, 15,000 prompt tokens, `max_model_len=16384`; vendor prefill + M1 decode succeeded | `docs/C500_V14_VALIDATION_20261017.md` |
| Backward | Unsupported | Explicit validation error |
| FP8 | Unsupported in this candidate | Future milestone |
| Multi-card | Unsupported in this candidate | Future milestone |
| SGLang | Not tested in the C500 image | Future milestone |

## Performance wording

Only records containing `m1_split_decode` events are M1 performance evidence.
Records containing only `native_decode` are v7 fallback evidence; records containing
only `vendor_direct` are vendor or guard paths and must not be attributed to M1.

Across the 36 C500 operator cases, MXFlashAttn's median latency is 60.52%
lower than the project's PyTorch reference and 32.74% higher than the MetaX
vendor median. The first number is a reference comparison; the second means
that no vendor-speedup claim is made. Model-level vLLM tokens/s are reported in
a separate section because they do not share the operator matrix's inputs. The
primary 2026-10-13 historical result is Qwen3-4B batch=32, 1024-token decode
at about 3.59x versus the project's v7 path. The latest 2026-10-17 retest
recorded about 3.66x for Qwen3-4B and 1.37x for Qwen3-8B in the listed
workloads; these remain workload-specific C500 measurements.

## Fallback

Native execution is attempted only for supported FP16/BF16 accelerator tensors.
If no eligible provider is available, the API raises unless
`MXFLASHATTN_ALLOW_FALLBACK=1` is explicitly set. Fallback emits a warning and
records its reason through `get_last_dispatch_info()`. It is a correctness
reference, not a C500 performance baseline.
