# Qwen3-0.6B vLLM baseline

Environment: MetaX C500 64G, vLLM `0.17.0`, PyTorch
`2.8.0+metax3.5.3.9`, MXMACA 3.5.3.20. The model was loaded once and four
sequential requests were generated with temperature 0.

The 16-token suite (`qwen3-0.6b-suite-v3.json`) reports:

| Case | Prompt tokens | Generated tokens | Wall latency | Decode tokens/s | `mx-smi` VRAM used |
|---:|---:|---:|---:|---:|---:|
| 0 | 10 | 16 | 326.00 ms | 49.08 | 58.17 GiB |
| 1 | 9 | 16 | 208.31 ms | 76.81 | 58.17 GiB |
| 2 | 10 | 16 | 205.35 ms | 77.92 | 58.17 GiB |
| 3 | 10 | 16 | 202.02 ms | 79.20 | 58.17 GiB |

A second 32-token suite is retained in `qwen3-0.6b-suite-v2.json`; its steady
state decode throughput was 79.15-80.38 tokens/s after the first request.

TTFT and request finish timestamps are `null` because vLLM 0.17.0 did not expose
those offline request metrics in this run. PyTorch allocator counters reported
zero on this MetaX runtime, so the VRAM figure comes from the device-level
`mx-smi --show-memory` sample and is labeled accordingly. These are baseline-only
measurements and do not claim MXFlashAttn integration.
