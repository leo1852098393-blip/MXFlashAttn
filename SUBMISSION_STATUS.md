# 申报材料状态（2026-10-06 更新）

## 项目简介

MXFlashAttn 是面向 MXMACA/C500 的 FlashAttention 兼容前向 API 与大模型解码适配层，
提供 dense、varlen、paged KV cache、显式 dispatch/fallback、vLLM 适配和可复现 benchmark
工具链。当前优化执行依赖匹配的 MetaX `flash-attn` wheel；仓库内的 C++ ATen 扩展是
朴素实现的 bring-up / 正确性对照路径，不表述为自研融合 kernel。

## 当前真实状态（先说没做完的）

- **v1.1.0-vendor-wiring 已在本地完成接线，尚未在 C500 上复跑。** dispatch 顺序改为
  `vendor_direct`（MetaX wheel）→ `mxmac_aten_correctness`（ATen 扩展）→ `reference`（显式开启）。
  本地只能跑通 `compileall`，无法跑 `pytest`（无 torch 环境），因此**本版没有任何新的性能结论**。
- 复跑前的性能数据一律沿用 v1.0 C500 实测值。复跑步骤见 `docs/C500_VALIDATION_RUNBOOK.md`。

## 技术路线

1. 以公共 API 和统一校验隔离 reference、vendor 与 MXMACA dispatch。
2. 以 paged KV cache 和 vLLM decoder contract 接入单卡 C500 解码。
3. 以固定 seed、同输入、同设备矩阵比较 reference、SDPA、vendor 与 API。
4. 以原始 JSON、dispatch JSONL、环境清单和 Markdown 报告保证复现。

## 阶段成果（均有 C500 原始证据）

- C500 36/36 公平算子矩阵完成，无异常；API 对 reference 最大绝对误差 0.00390625。
- 算子级：MXFlashAttn 相对 PyTorch reference 中位延迟下降 60.52%；
  相对 MetaX vendor 直调中位延迟上升 32.74%（**如实写出，不宣称对 vendor 加速**）。
- Qwen3-0.6B candidate 生成完成，decoder dispatch 证据显示 `mxflashattn_decoder`；
  prefill 明确委托 vendor。
- vendor/candidate 使用同一组 36 个固定 prompt 完成模型级配对运行：
  vendor 中位 76.46 tokens/s，candidate 中位 46.15 tokens/s（差距约 39.7%）。
  两者单独列示，不把模型 tokens/s 混入算子级 median。

## 已知的硬缺口（评审问到必须承认）

- **性能：当前慢于官方 vendor 直调**（算子级 +32.74%，模型级 -39.7%）。v1.1 接线就是为了收敛这个差距，复跑前不做提速宣称。
- **未测量 TTFT**：vLLM 0.17.0 离线接口不提供首 token 时间戳，尚未做外部计时。
- **模型规模小**：只在 Qwen3-0.6B 上验证，计划中的 Qwen2.5-1.5B / 7B 未跑。
- **backward / FP8 / 多卡 / SGLang 均未支持**。
- **尚未公开发布**：无 release、无 GitLink 镜像、无演示视频、无外部用户与贡献者。
- **自研 kernel 尚未开始**：仓库不含任何 MACA device kernel。

## 评审口径

可申报的工程价值是国产算力兼容层、paged-KV 解码适配、可观测 fallback 和验证工具链，
以及"把国产生态的适配过程做成可复现证据"这个方法论。

不能写成：vendor 超越、TTFT 已测、backward/FP8/多卡已支持、自研融合 kernel 已完成、
已完成公开发布迭代、已有外部用户。

## 后续路线

1. C500 上完成 v1.1 复跑，量化 candidate 与 vendor 的差距（最高优先级）。
2. 外部计时补 TTFT，扩展更多 vLLM 兼容版本。
3. 评估 backward、FP8、多卡与 SGLang。
4. 研究不依赖 vendor wheel 的自研 kernel。

每一项都必须先有真实运行证据，才能进入公开说明。
