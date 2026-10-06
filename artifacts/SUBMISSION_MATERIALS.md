# 申报摘要（v1.0.0 候选稿）

## 项目简介

MXFlashAttn 面向 MXMACA / MetaX C500，提供 FlashAttention 兼容的
FP16/BF16 forward API、varlen 与 paged KV cache 推理适配，并以显式
dispatch/fallback、固定 seed 矩阵和原始 JSON 结果保证复现性。当前优化
执行依赖匹配的 MetaX `flash-attn` wheel；项目贡献集中在兼容 API、国产
算力适配、paged-KV 解码桥接和验证工具链。

## 已验证阶段成果

- C500 36 组固定 seed、同输入公平矩阵全部完成，覆盖 reference、Torch
  SDPA、MetaX vendor 和 MXFlashAttn API；无异常，最大 API 对 reference
  绝对误差为 `0.00390625`。
- MXFlashAttn API 相对项目 PyTorch reference 的中位延迟下降 `60.52%`；
  相对 MetaX vendor 中位延迟上升 `32.74%`，因此没有 vendor 加速声明。
- Qwen3-0.6B 在 vLLM 0.17.0 上完成 candidate 生成，dispatch JSONL 中
  出现 `mxflashattn_decoder`；prefill 委托 MetaX vendor。
- vendor/candidate 使用同一组 36 个固定 prompt 完成模型级配对运行；结果
  单独列示，不混入算子级 median。

## 明确限制

vLLM 0.17.0 离线接口没有提供可验证 TTFT，因此 TTFT 保持为空；backward、
FP8、多卡和 SGLang 端到端接入未完成。项目不声称自研融合 kernel，也不
声称超过 MetaX vendor。

## 后续路线

补充外部计时 TTFT、扩展 vLLM 兼容版本、优化 API 与 vendor 差距，再分别
评估 backward、FP8、多卡和 SGLang。任何新结论先保存真实日志和 JSON，
再进入公开版本。
