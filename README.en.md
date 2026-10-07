# MXFlashAttn

> **Status (2026-10-07): v1.3.0. The in-house split-KV decode kernel (M1) completed
> Gate 4 validation on MetaX C500 and actually serves decode steps inside vLLM 0.17.0.**

[中文说明](README.md) · **English**

MXFlashAttn is a FlashAttention-compatible forward operator and inference adapter for
**MXMACA / MetaX C500**. v1.3.0 ships an **in-house device kernel**
(`csrc_native/m1_kernel.cpp`, split-KV paged decode) that delivers roughly
**0.97x-6.48x** end-to-end throughput over this project's own v7 path under vLLM
continuous batching: **2.0x-6.48x** in the long-sequence / low-to-mid concurrency
range, while short sequences at low concurrency can be flat or slightly negative.

## v1.3.0 headline numbers

M1 vs the project's v7 split path, continuous-batching throughput (tok/s):

| Model | Shape | b64x256 | b32x1024 |
| --- | --- | --- | --- |
| Qwen3-4B | GQ=4 / HD=128 | 2.13x | 3.67x |
| Qwen3-8B | GQ=4 / HD=128 | 2.20x | 3.49x |
| PaddleOCR-VL | GQ=8 / HD=128 | 1.53x | 2.47x |
| TinyLlama-1.1B | GQ=8 / HD=64 | 2.96x | **6.48x** |

Two things worth calling out:

1. **The gain tracks batch x sequence shape; no clear model-size dependence was observed**
   across the four validated scales (0.6B/1.1B/4B/8B). 4B and 8B reproduce
   each other within <9% across four main scenarios. Short sequences at low concurrency
   (b16x64) are launch-latency bound and gain nothing.
2. **split-KV is not an optimization but a prerequisite** for in-house decode on large
   models: on Qwen3-4B the v7 path is 3.2x *slower* than the vendor direct call
   (1383 vs 4453 tok/s), and M1 brings it back to vendor parity (slightly ahead).

All 6 cells of the supported shape set `{GQ in {2,4,8} x HD in {64,128}}` have real-model
E2E evidence. Shapes inside the currently defined unsupported contract (non-power-of-two
GQA already exercised as GQ=5/7, head_dim > 128) are cleanly rejected by the two guard
layers without affecting generation correctness; other combinations outside that contract
carry no support commitment.

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
python csrc_native/bench_e2e.py --model /path/to/Qwen3-4B --tag q4b
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
docs/gate4/        Gate 4 validation reports and all raw evidence
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
