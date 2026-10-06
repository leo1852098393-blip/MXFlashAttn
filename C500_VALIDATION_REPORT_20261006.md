# C500 验收报告（2026-10-06）

> **重要更正（2026-10-06 晚）：** 本报告成文早于静默 import 回退根因的发现与修复。当时 candidate 的
> vLLM 插件实际回退到了上游 vLLM FA 实现（`impl_module=vllm.v1.attention.backends.flash_attn`），
> 因此本报告中的 candidate 侧数字（eager 61.47 / graph 263.93 等）与"dispatch 已进入 MXFlashAttn"
> 的结论**不再代表最终实现**。修复（`MacaFlashAttentionBackend as FlashAttentionBackend`）后的最终
> 验收结果为：eager 71.13 vs 79.48 tokens/s（慢 10.5%，36/36 文本一致）、graph 257.24–258.19 vs
> 274.09 tokens/s（慢约 6%，36/36 文本一致，双复跑），见 `README.md` 与 `artifacts/official-split-*`。
> 本报告的执行步骤与链路验收方法仍然有效，性能数字以 README 为准。

## 环境

- GPU：MetaX C500
- MACA：3.5.3.20
- PyTorch：2.8.0+metax3.5.3.9
- vLLM：0.17.0
- 模型：`/root/models/Qwen3-0.6B`（通过 hf-mirror 下载，离线加载）
- 初始链路测试：1 条固定 prompt，max_tokens=16；随后完成 36 条固定 prompt 的 eager/graph 对照复测。

## 调用链验收

`cand-eager-1.json` 和 `cand-graph-1.json` 均满足：

- `candidate_dispatch_verified=true`
- dispatch 路径包含 `vendor_direct`
- plugin log 包含 `plugin_imported` 和 `impl_constructed`
- prefill 记录 `vendor_prefill_or_blocked`，decode 记录 `vendor_direct`

这证明真实 vLLM worker 的 decode 已进入 MXFlashAttn，并调用 MetaX vendor kernel。

## 小规模性能结果

| 模式 | vendor baseline tok/s | candidate tok/s | candidate 状态 |
|---|---:|---:|---|
| eager | 49.2086 | 38.8989 | verified，含 `vendor_direct` |
| CUDA Graph | 154.1437 | 147.1360 | verified，含 `vendor_direct` |

单次运行的 candidate 相对 baseline 约为：eager 慢 20.95%，graph 慢 4.55%。这些不是最终 36-prompt 中位数，只是当前离线模型环境下的链路验收结果。

## 36-prompt 公平复测

固定条件：Qwen3-0.6B、36 条 prompt、max_tokens=16。candidate 使用 `MXFLASHATTN_DISPATCH_SAMPLE=2000`，避免证据写盘污染热路径。

| 模式 | vendor median tok/s | candidate median tok/s | candidate 状态 |
|---|---:|---:|---|
| eager | 79.2559 | 61.4747 | verified，含 `vendor_direct` |
| CUDA Graph | 274.0861 | 263.9327 | verified，含 `vendor_direct` |

相对 vendor median，candidate eager 约慢 28.2%，candidate graph 约慢 3.7%。两组均为同模式对比；不能将 graph 相对 eager 的提升归因于 MXFlashAttn。

## 自动化测试

在正式副本 `/root/MXFlashAttn_v1.2.0` 上运行：

```text
43 passed in 7.92s
```

## 证据文件

- `artifacts/base-eager-1.json`
- `artifacts/cand-eager-1.json`
- `artifacts/base-graph-1.json`
- `artifacts/cand-graph-1.json`
- `artifacts/cand-eager-1.dispatch.jsonl`
- `artifacts/cand-graph-1.dispatch.jsonl`
- `artifacts/cand-eager-1.plugin.log`
- `artifacts/cand-graph-1.plugin.log`
- `artifacts/base-eager-36.json`
- `artifacts/cand-eager-36-sampled.json`
- `artifacts/base-graph-36.json`
- `artifacts/cand-graph-36-sampled.json`
- `artifacts/cand-eager-36-sampled.dispatch.jsonl`
- `artifacts/cand-graph-36-sampled.dispatch.jsonl`
- `artifacts/cand-eager-36-sampled.plugin.log`
- `artifacts/cand-graph-36-sampled.plugin.log`

## 当前代码状态

正式目录 `MXFlashAttn_v1.2.0` 已包含：

- 图安全 `_decode_forward_graph`；
- transparent prefill fallback；
- `seq_lens.reshape(-1)` 形态规整；
- worker dispatch JSONL 直接持久化；
- dispatcher fallback reason 修复；
- 43 项回归测试修复。

## 限定

本报告已完成真实链路验收和单 prompt eager/graph 对照。最终发布前仍应补跑 36-prompt 多次中位数矩阵，并单独保存 correctness 误差矩阵；不能用本报告的单次 tok/s 替代正式性能结论。
