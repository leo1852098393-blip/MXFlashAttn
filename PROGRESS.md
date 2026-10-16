# MXFlashAttn v1.4.0 Progress

更新时间：2026-10-17

## 当前状态

- 本开发线从已冻结的 `v1.3.0` public candidate 创建。
- v1.3.0 的 M1 split-KV decode 能力、C500/vLLM 0.17.0 证据和回退边界保持不变。
- v1.4.0 已完成本轮新实例上的 C500 验证，但尚未宣称 Prefill 生产加速、
  自适应阈值、vLLM 0.30、FP8、多卡或 SGLang 支持。
- C500 原始证据保存在上层工作区 `02_evidence/`，不随公开代码提交。

## 第一阶段进度

- ✅ 已完成 Prefill correctness-first 原型：`mxflashattn.prefill`
- ✅ 已覆盖 GQA、causal、短 chunk、FP16/BF16 和非法参数测试
- ✅ 原型暂不接入 runtime dispatcher，不改变 v1.3.0 基线行为
- ✅ 已完成长上下文误差摘要、候选调度策略和 vLLM 版本兼容分类
- ✅ 已固定 C500 实验矩阵并生成 manifest
- ✅ 已在约 63.6 GiB 显存的 C500 实例上完成环境部署、扩展编译和回归
- ✅ C500 完整测试通过：`79 passed`
- ✅ 已完成真实 vLLM 0.17.0 生成验证：Qwen3-4B、Qwen3-8B
- ✅ 已完成 M1 的 BF16、8192 上下文、overflow 和不支持 GQA guard 验证
- ✅ 已完成 Qwen3.5-9B `head_dim=256` 的 fallback/guard 验证
- ✅ Prefill prototype 已完成正确性和性能定位
- ⏳ Prefill prototype 未进入默认 dispatch，也未宣称 Prefill 加速
- ✅ official split 的 `invalid argument` 窄化 vendor 回退保护已实现，并加入单元回归测试
- ✅ 真实 15K prompt / max_model_len=16384 已在 C500 完成 vendor prefill + M1 decode 回归

## C500 本轮验证结果

- 计划：`benchmarks/configs/v14_c500_prefill.json`
- 实验说明：`docs/C500_V14_EXPERIMENT_PLAN.md`
- 清单生成：`scripts/prepare_c500_v14.py`
- 新实例环境：MACA `3.5.3.20`、MetaX PyTorch `2.8.0+metax3.5.3.9`、
  `flash-attn 2.6.3+metax3.5.3.9torch2.8`、vLLM `0.17.0`
- Qwen3-4B batch=32、生成 1024 token：M1-UM2 相对 v7 约 `3.66x`
- Qwen3-8B batch=16、生成 256 token：M1-UM2 相对 v7 约 `1.37x`
- Prefill prototype 正确，但比 vendor 慢约 `8.2x`--`32.3x`
- 详细结果：`docs/C500_V14_VALIDATION_20261017.md`
- vLLM 兼容：0.17.0 已验证；0.30.x 仍为 `probe_required`

## v1.3.0 基线结果（历史对照）

- Qwen3-4B batch=32、1024 token：相对 v7 为约 3.59x。
- Qwen3-8B batch=16、256 token：约 1.40x。
- PaddleOCR-VL GQ=8 batch=16、64 token：约 1.19x。
- Qwen3-4B 长上下文复测峰值显存约 59.6 GiB。

## 基线证据入口

- `docs/gate4/Gate4_收官报告.md`：Gate 4 总结。
- `docs/gate4/E2E_大模型与守卫验证.md`：跨模型、守卫和 fallback 证据。
- `02_evidence/model_benchmarks/E2E_SUMMARY.md`：本地整理的 30 条 E2E 记录分类。
- `02_evidence/model_benchmarks/EVIDENCE_INDEX.json`：机器可读索引。

## 尚未完成或暂不承诺

严格 token-level 长尾一致性、backward、FP8、多卡、SGLang、vLLM 0.30.x
兼容性尚未验证；流式 TTFT/P95 采集链路已在 C500 打通，但 AsyncLLMEngine
candidate 尚未复用已验证的同步 M1 adapter，因此不能把该流式 candidate 数字
写成 M1 TTFT/P95 提升；
此前 16K official split wrapper 真实 prefill 触发过 `invalid argument`，当前代码已
加入窄化 vendor 重试保护；新实例 15K prompt 回归未复现该错误，且 M1 decode
真实命中；
短序列低并发可能无收益或略慢。当前 vLLM 离线接口没有提供首 token 和完成时间戳，
此前本轮只归档 wall latency、吞吐、显存和同步 dispatch 结果；流式 baseline
和 candidate 诊断结果已归档到 `02_evidence/.../streaming/`，严格 token-level
长尾一致性仍需更大样本验证。

## 当前收口判断

v1.4.0 当前可作为完成 C500 验证的发布候选版本继续整理申报材料。核心 M1
Split-KV decode 已在真实 Qwen3-4B/8B vLLM 生成链路复现；Prefill prototype
仅作为 correctness-first 研究原型保留，不接入生产 dispatch。PyPI 发布继续等
版本范围、文档和发布策略最终确认后再决定。
