# MXFlashAttn v1.0.0 Submission Candidate

更新时间：2026-10-05

## 已验证证据

- v0.7：C500 vLLM 0.17.0 decoder tensor/cache layout 与 dispatch contract。
- v0.8：Qwen3-0.6B candidate 生成，JSONL 中出现 `mxflashattn_decoder`，并
  明确记录 prefill vendor 委托；TTFT 保持为空。
- v0.9：36 组固定 seed、同输入、同设备的 reference/SDPA/vendor/API 矩阵，
  36/36 无异常，MXFlashAttn 对 reference 最大绝对误差 0.00390625。
- v0.9 模型级补充：同一组 36 个固定 prompt 分别完成 vendor 与 candidate
  运行，candidate 36/36 保留 `mxflashattn_decoder` dispatch 证据。

## 关键文件

- `artifacts/fair-matrix-c500.json`：36 组原始 C500 数据。
- `artifacts/fair-matrix-c500.md`：公平矩阵报告。
- `artifacts/fair-matrix-summary.json`：机器可读结论，vendor 加速声明为 false。
- `artifacts/c500-qwen-candidate-suite.json`：模型级 candidate 证据。
- `artifacts/c500-qwen-candidate-suite.dispatch.jsonl`：实际 dispatch 事件。
- `artifacts/C500_ENVIRONMENT.json`：驱动、MXMACA、PyTorch 和 vLLM 版本。

## 尚未完成或不承诺

TTFT、backward、FP8、多卡、SGLang 和超过 MetaX vendor 的性能均未验证。
这些限制必须保留在申报材料和公开说明中。
