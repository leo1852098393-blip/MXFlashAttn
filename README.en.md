# MXFlashAttn

> **Status (2026-10-13): v1.3.0 local release candidate. The in-house split-KV
> decode kernel (M1) has completed real vLLM 0.17.0 generation, dispatch, and
> fallback validation on a 64 GiB MetaX C500.**
>
> **Scope:** a C500-specific MXMACA adapter and decode optimization project,
> not a cross-platform FlashAttention replacement or an MXAttention/MXFP4 reproduction.

[中文说明](README.md) · **English**

MXFlashAttn is a FlashAttention-compatible forward operator and inference adapter for
**MXMACA / MetaX C500**. v1.3.0 ships an **in-house device kernel**
(`csrc_native/m1_kernel.cpp`, split-KV paged decode). In the 2026-10-13 retest,
Qwen3-4B reached about **3.59x** the throughput of this project's own v7 path at
batch=32 and max_tokens=1024. Gains depend on the model, batch, and KV length;
short or low-concurrency workloads are not guaranteed to improve.

## 2026-10-13 retest (primary application wording)

Environment: MetaX C500 64 GiB, MACA 3.5.3.20, driver 3.8.30, PyTorch
`2.8.0+metax3.5.3.9`, and vLLM 0.17.0. Every M1 result below has
`m1_split_decode` evidence in the dispatch log.

| Model and workload | v7 | M1-UM2 | vs v7 |
| --- | ---: | ---: | ---: |
| Qwen3-4B, batch=128, 256 tokens | 1375.59 tok/s | 4554.69 tok/s | **3.31x** |
| Qwen3-4B, batch=32, 1024 tokens | 359.82 tok/s | 1293.48 tok/s | **3.59x** |
| Qwen3-8B, batch=16, 256 tokens | 500.79 tok/s | 703.06 tok/s | **1.40x** |
| PaddleOCR-VL, batch=16, 64 tokens | 281.73 tok/s | 334.71 tok/s | **1.19x** |

These numbers apply only to the listed C500 configurations and must not be
generalized into a fixed speedup for every model or request shape. The Qwen3-4B
long-context run peaked at about 59.6 GiB. This round did not produce a unified
TTFT/P95 evaluation.

The broader 2026-10-07 Gate 4 matrix, including historical 3.67x and 6.48x
results, is retained as development provenance and is not mixed with this retest:
[`docs/gate4/Gate4_收官报告.md`](docs/gate4/Gate4_收官报告.md).

All 6 cells of the supported shape set `{GQ in {2,4,8} x HD in {64,128}}` have real-model
E2E evidence. Unsupported M1 shapes must not call the binding directly. In the
vLLM official-split path they return to the original MetaX/vendor KV-cache
implementation. On 2026-10-13, Qwen3.5-9B with head_dim=256 completed real
generation with zero M1 dispatches and 4/4 text matches for both M1-UM modes.

Full report: [`docs/gate4/Gate4_收官报告.md`](docs/gate4/Gate4_收官报告.md).

## Reproducing on C500

```bash
export MACA_PATH=/opt/maca          # without this, triton import fails with NoneType
bash csrc_native/build_m1.sh
python csrc_native/patch_plugin.py  # idempotent
python csrc_native/test_m1.py
```

End-to-end benchmark -- all four env vars matter, omit any one and the plugin silently
falls back to the vendor direct call, degrading the 3-way comparison into the same path
run three times:

```bash
export MXFLASHATTN_GRAPH_OFFICIAL_SPLIT=1 MXFA_NATIVE_DECODE=1
export VLLM_ENABLE_V1_MULTIPROCESSING=0
python csrc_native/bench_e2e.py --model /path/to/Qwen3-4B --batches 128 64 --max-tokens 256 --output artifacts/bench-q4b.json
```

**Check the `m1_split_decode` event count in `dispatch.jsonl` before trusting any
throughput number.** A count of 0 means the kernel was never dispatched.

## Layout

```text
mxflashattn/       Python API, validation, reference and dispatch
csrc/mxmac/        C++/ATen bring-up extension (naive O(n^2), correctness baseline only)
csrc_native/       in-house M1 split-KV decode kernel + v7 kernel + bench/test/build
integrations/vllm/ vLLM 0.17.0 bridge, decoder adapter and preflight
benchmarks/        fixed-config fair matrix runner and reporting
docs/gate4/        Gate 4 technical reports (raw instance logs stay local)
tests/             CPU/reference and C500-conditional tests
```

## Known limitations

- v7 path has a 4096 context ceiling (fails to start above it).
- **M1 is still below the hardware ceiling**: IR analysis shows v7 uses the fused
  `llvm.mxc.expadd.f32.i32` intrinsic for softmax rescaling while M1 emulates it with
  software shuffle reduction + `exp2`. Switching to the hardware intrinsic is the next
  step, not yet done.
- `text_match` from the E2E bench is **not** a correctness criterion under continuous
  batching (batch composition drifts). Verify text equality at a fixed batch instead.
- Dispatch events must be counted by the `(path, operation, fallback)` triple; counting
  by path alone misreports "never dispatched" when M1 never succeeded.

## License

Apache License 2.0. See [LICENSE](LICENSE). MXMACA runtime, vendor PyTorch, MetaX
wheels, vLLM, model weights and datasets each follow their own licenses.
