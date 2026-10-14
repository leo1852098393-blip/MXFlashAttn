# MXFlashAttn v1.3.0 Progress

更新时间：2026-10-13

## 当前状态

- M1 split-KV decode 已在 64G C500 上完成真实 vLLM 0.17.0 生成、dispatch
  和 fallback 回归，并接入 vLLM 0.17.0。
- 当前复测证据覆盖 Qwen3-4B、Qwen3-8B、PaddleOCR-VL GQ=8，以及
  Qwen3.5-9B head_dim=256 回退路径；历史 Gate 4 证据另行归档。
- M1 支持矩阵的 6 个已声明格子为 `GQ={2,4,8} x head_dim={64,128}`；GQ=5/7 与
  `head_dim>128` 不进入 M1。vLLM official split 对这些形状回退到原始
  MetaX/vendor 路径；直接调用公共 API 仍显式拒绝。
- 所有性能结论必须同时检查 `dispatch.jsonl`；没有 `m1_split_decode` 事件的记录不计入 M1。

## 当前主结果

- Qwen3-4B batch=32、1024 token：相对 v7 为约 3.59x。
- Qwen3-8B batch=16、256 token：约 1.40x。
- PaddleOCR-VL GQ=8 batch=16、64 token：约 1.19x。
- Qwen3-4B 长上下文复测峰值显存约 59.6 GiB。

## 证据入口

- `docs/gate4/Gate4_收官报告.md`：Gate 4 总结。
- `docs/gate4/E2E_大模型与守卫验证.md`：跨模型、守卫和 fallback 证据。
- `02_evidence/model_benchmarks/E2E_SUMMARY.md`：本地整理的 30 条 E2E 记录分类。
- `02_evidence/model_benchmarks/EVIDENCE_INDEX.json`：机器可读索引。

## 当前不承诺

TTFT/P95、backward、FP8、多卡和 SGLang 尚未验证；短序列低并发可能无收益或略慢。
当前证据没有形成统一的 TTFT/P95 结论。阈值调度仍需在 C500 上校准，当前本地策略
只负责候选形状判断和安全回退。
