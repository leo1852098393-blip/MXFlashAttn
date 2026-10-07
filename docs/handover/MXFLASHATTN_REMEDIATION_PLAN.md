# MXFlashAttn 评审问题整改计划

更新时间：2026-10-05

## 目标

针对外部评审指出的文档与代码不一致、vLLM 注册探测被误读为端到端集成、版本目录缺少真实演进、smoke test 未调用 MXFlashAttn、baseline 不够公平以及项目技术价值边界不清等问题，建立可复现、可审计的整改路径。

公开 `MXFlashAttn/`、GitHub 和 GitLink 在整改完成前保持不变。每个整改阶段使用独立本地目录，不把未验证能力写入公开材料。

## 严重程度与处理原则

### P0：端到端路径和事实一致性

1. **修复 v0.6 代码与文档的一致性**
   - 候选目录必须包含它声称已验证的探针、配置和适配代码。
   - README、支持矩阵、PROGRESS、申报摘要和 JSON 结果必须使用同一套状态词。
   - “registration probe passed”只能表示注册 API 可达，不能写成“MXFlashAttn 已用于模型生成”。
   - 验收：源码哈希、文档声明、探针 JSON 和运行日志互相对应；敏感文件审计通过。

2. **完成真实 decoder 生成验证**
   - 在 C500 的 vLLM 0.17.0 环境中确认 Qwen3-0.6B 的 decoder attention 调用链。
   - 将 vLLM 的 query、key、value、KV cache、block table、sequence lengths 和 scheduler metadata 正确转换到 MXFlashAttn API。
   - decoder 路径不得调用 `super().forward()` 作为最终实现；必须记录实际调用的 `get_last_dispatch_info()`。
   - 使用同一模型、同一 prompt、同一采样参数分别运行 vendor baseline 和 MXFlashAttn candidate。
   - 验收：生成文本、token 数、TTFT、完成延迟、decode tokens/s、峰值显存、backend 名称和 fallback 状态全部落盘；backend 必须显示 MXFlashAttn 路径。

3. **明确失败边界**
   - 如果 paged KV 接入在 vLLM 0.17.0 上不稳定，不得保留“已完成”措辞。
   - 记录具体阻塞点：cache layout、block table、cascade、量化 cache、scheduler metadata 或 vLLM ABI 版本差异。
   - 同时评估独立推理 demo 或 SGLang 路径，但只有真实运行结果才能写入“已验证”。

### P1：版本和证据可信度

4. **建立真实的本地迭代链**
   - `MXFlashAttn_v0.7.0-decoder-probe`：只做调用链追踪和最小 decoder adapter。
   - `MXFlashAttn_v0.8.0-qwen-e2e`：只做 Qwen 端到端生成和 baseline 对比。
   - `MXFlashAttn_v0.9.0-fair-benchmark`：扩展 Torch SDPA、vendor provider 和 candidate 的完整矩阵。
   - `MXFlashAttn_v1.0.0-submission`：汇总已验证材料并做发布前审计。
   - 每个目录必须有独立 `SCOPE.md`、`CHANGELOG.md`、测试结果和原始 JSON。
   - 不把旧 README 原样复制后当作新版本说明；每个版本必须说明新增代码、测试和限制。

5. **避免把证据目录当成代码实现**
   - v0.6 只能汇总真实存在于候选目录中的代码和结果。
   - 证据索引必须标明：测试命令、环境、commit/hash、backend、是否 fallback、是否模型级。
   - 任何没有运行日志支持的数字不得进入申报摘要。

### P1：公平性能证据

6. **扩大公平 baseline**
   - 在相同 C500、相同 seed、相同输入布局和相同 dtype 下比较：
     - PyTorch reference
     - Torch SDPA
     - MetaX vendor FlashAttention
     - MXFlashAttn API
     - vLLM vendor baseline
     - vLLM MXFlashAttn candidate（完成后）
   - 覆盖 dense、varlen、paged KV、prefill、decode、FP16、BF16、GQA/MQA、head dimension 64/128。
   - 至少形成 30 组完整矩阵，并单独报告 warmup、重复次数、同步方式和显存采样方法。
   - 性能结论必须区分“相对朴素 reference 的收益”和“相对 vendor baseline 的收益”。

7. **模型级指标规范化**
   - TTFT 使用运行时可验证的时间戳；如果 vLLM 0.17.0 不提供，则使用明确的外部计时方案并说明测量边界。
   - 记录 prompt tokens、generated tokens、TTFT、端到端延迟、decode latency、tokens/s、显存、模型路径、vLLM/MXMACA/PyTorch 版本。
   - candidate 结果必须证明调用了 MXFlashAttn，而不是只运行 vendor FlashAttention。

### P2：项目定位和申报材料

8. **准确描述技术贡献**
   - 将项目定位为 MXMACA/C500 上的 FlashAttention 兼容层、paged KV 推理适配、dispatch/fallback、验证和 benchmark 工具链。
   - 不声称自研融合 kernel，除非后续确实提交并验证新的 kernel 实现。
   - 明确 vendor wheel 是当前快速路径依赖，说明项目自身提供的接口、适配和工程价值。

9. **重写申报材料中的状态表**
   - 已完成：有原始 JSON、日志和可重复命令。
   - 已验证探针：只证明 API 或注册点可达。
   - 进行中：代码存在但没有真实模型证据。
   - 未支持：明确列出 backward、FP8、多卡、SGLang 等限制。
   - 只有 v1.0.0-submission 完成后，才考虑合并公开仓库和同步 GitHub/GitLink。

## 迭代验收表

| 阶段 | 目录 | 必须产物 | 通过条件 |
|---|---|---|---|
| 调用链 | `MXFlashAttn_v0.7.0-decoder-probe` | adapter、调用日志、失败原因 | 明确 decoder tensor/cache layout，能定位实际 backend |
| 模型生成 | `MXFlashAttn_v0.8.0-qwen-e2e` | vendor/candidate JSON、生成文本、TTFT、tokens/s、显存 | candidate dispatch 显示 MXFlashAttn，结果可重复 |
| 公平 benchmark | `MXFlashAttn_v0.9.0-fair-benchmark` | 30+ 组 JSON、报告和图 | vendor 与 candidate 在同环境公平比较 |
| 发布候选 | `MXFlashAttn_v1.0.0-submission` | 双语 README、支持矩阵、PROGRESS、申报材料、审计 | 测试、敏感信息、许可证和声明全部通过 |

## 当前不得宣称

- 不得宣称“vLLM 已完成 MXFlashAttn 端到端集成”。
- 不得宣称“Qwen 生成已使用 MXFlashAttn”，除非 dispatch 证据显示 candidate backend。
- 不得把相对项目 reference 的加速直接写成相对 vendor baseline 的加速。
- 不得把多个本地目录直接包装成多个公开 release。
- 不得把 TTFT 为空的结果写成已获得首 token 延迟。

## 目标模式提示词

将下面内容作为目标模式的目标：

```text
在 C:\Users\86191\Downloads\mx 继续整改 MXFlashAttn，公开 MXFlashAttn/、GitHub 和 GitLink 保持不变。严格按 MXFLASHATTN_REMEDIATION_PLAN.md 分阶段推进，每个阶段使用独立本地目录，不把不同主题混在一起。

第一阶段创建 MXFlashAttn_v0.7.0-decoder-probe：检查 vLLM 0.17.0 的 Qwen3-0.6B decoder 调用链，确认 query/key/value、paged KV cache、block table、cache_seqlens 和 scheduler metadata 的真实布局；实现最小 adapter 或记录明确阻塞点。禁止把注册探测写成端到端集成。必须保存源码变更、运行日志、backend dispatch 信息、测试结果、SCOPE.md 和 CHANGELOG.md。

第二阶段创建 MXFlashAttn_v0.8.0-qwen-e2e：只有在 decoder adapter 可验证时，分别运行 vendor baseline 和 MXFlashAttn candidate。记录生成文本、prompt tokens、generated tokens、TTFT、端到端延迟、decode tokens/s、峰值显存、vLLM/MXMACA/PyTorch 版本和 fallback 状态。若 candidate 没有实际调用 MXFlashAttn，必须判定未完成并记录原因。

第三阶段创建 MXFlashAttn_v0.9.0-fair-benchmark：完成至少 30 组相同输入下的 PyTorch reference、Torch SDPA、MetaX vendor、MXFlashAttn API、vLLM vendor 和 vLLM candidate 对比，区分 reference 加速与 vendor 加速，保存 JSON 和 Markdown 报告。

第四阶段创建 MXFlashAttn_v1.0.0-submission：汇总已验证结果，更新双语 README、支持矩阵、PROGRESS、申报材料和证据索引，运行所有本地测试、许可证/依赖/敏感文件审计。未验证能力必须写成限制。只有完成发布审查后，才向我报告是否适合合并公开仓库；未经我明确要求不要 push 或同步 GitHub/GitLink。

每次工作先检查当前目录和前一版本结果，不能依赖记忆。完成一个阶段后报告：改动文件、实际命令、测试输出、原始证据路径、未完成项和是否可以进入下一阶段。不要为了让指标看起来完成而伪造模型集成、TTFT、vendor 加速或版本演进证据。
```

## 当前执行状态（2026-10-05）

- v0.7 decoder probe：已完成，保存 C500 tensor/cache layout 与 dispatch 证据。
- v0.8 Qwen E2E：已完成，Qwen3-0.6B candidate 生成和 `mxflashattn_decoder` 事件已落盘；TTFT 仍不可用。
- v0.9 fair benchmark：已完成，C500 36/36 组无错误；相对 reference 的中位延迟下降 60.52%，相对 MetaX vendor 的中位延迟上升 32.74%，因此未声明 vendor 加速。
- v1.0 submission：已建立本地候选目录 `MXFlashAttn_v1.0.0-submission`，34 passed、8 skipped，许可证/敏感文件审计通过。
- 公开 `MXFlashAttn/`、GitHub 和 GitLink 未修改，尚未执行 push 或镜像同步。
