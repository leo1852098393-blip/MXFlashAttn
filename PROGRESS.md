# MXFlashAttn v1.4.0-dev Progress

更新时间：2026-10-17

## 当前状态

- 本开发线从已冻结的 `v1.3.0` public candidate 创建。
- v1.3.0 的 M1 split-KV decode 能力、C500/vLLM 0.17.0 证据和回退边界保持不变。
- v1.4.0-dev 尚未宣称 Prefill 优化、自适应阈值、vLLM 0.30、FP8 或多卡支持。
- 新功能先在本地完成正确性原型和测试，再安排一次有明确矩阵的 C500 验证。

## 第一阶段进度

- ✅ 已完成 Prefill correctness-first 原型：`mxflashattn.prefill`
- ✅ 已覆盖 GQA、causal、短 chunk、FP16/BF16 和非法参数测试
- ✅ 原型暂不接入 runtime dispatcher，不改变 v1.3.0 基线行为
- ✅ 已完成长上下文误差摘要、候选调度策略和 vLLM 版本兼容分类
- ✅ 已固定 C500 实验矩阵并生成待填写 manifest
- ⏳ 尚未进行 C500 性能测试，也未宣称 Prefill 加速

## 租 C500 前准备状态

- 计划：`benchmarks/configs/v14_c500_prefill.json`
- 实验说明：`docs/C500_V14_EXPERIMENT_PLAN.md`
- 清单生成：`scripts/prepare_c500_v14.py`
- 结果字段：Prefill/TTFT/P95、Decode 吞吐、峰值显存、误差分布、dispatch/fallback
- vLLM 兼容：0.17.0 已验证；0.30.x 仅 probe_required

## v1.3.0 基线结果

- Qwen3-4B batch=32、1024 token：相对 v7 为约 3.59x。
- Qwen3-8B batch=16、256 token：约 1.40x。
- PaddleOCR-VL GQ=8 batch=16、64 token：约 1.19x。
- Qwen3-4B 长上下文复测峰值显存约 59.6 GiB。

## 基线证据入口

- `docs/gate4/Gate4_收官报告.md`：Gate 4 总结。
- `docs/gate4/E2E_大模型与守卫验证.md`：跨模型、守卫和 fallback 证据。
- `02_evidence/model_benchmarks/E2E_SUMMARY.md`：本地整理的 30 条 E2E 记录分类。
- `02_evidence/model_benchmarks/EVIDENCE_INDEX.json`：机器可读索引。

## v1.4.0-dev 当前不承诺

TTFT/P95、backward、FP8、多卡和 SGLang 尚未验证；短序列低并发可能无收益或略慢。
当前证据没有形成统一的 TTFT/P95 结论。阈值调度仍需在 C500 上校准，当前本地策略
只负责候选形状判断和安全回退。
