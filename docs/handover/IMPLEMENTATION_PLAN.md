# MXFlashAttn 完整执行计划

更新时间：2026-10-05

## 一、目标

建设一个面向 MXMACA/C500 的 Apache-2.0 开源项目，提供 FlashAttention 风格的前向 API、GQA/MQA、varlen、paged KV cache 和可复现 benchmark，并逐步接入 vLLM 或 SGLang。

项目分为两个交付层次：

1. **首版可发布交付**：API、C500 算子验证、benchmark、文档和开源仓库。
2. **后续增强交付**：真实大模型通过 vLLM/SGLang 在 C500 上完成端到端文本生成。

端到端生成是第一优先级的后续工作，但不会把尚未验证的能力写成已完成成果。

## 二、当前状态

### 已完成

- Python API：`flash_attn_func`、`flash_attn_varlen_func`、`flash_attn_with_kvcache`。
- 输入校验、dtype/shape/设备检查和明确错误信息。
- PyTorch reference backend 和显式 fallback：`MXFLASHATTN_ALLOW_FALLBACK=1`。
- dispatch 层和 MetaX FlashAttention provider 识别。
- dense、varlen、GQA/MQA、causal、paged KV cache 场景。
- C++ ATen extension 的 C500 编译与验证路径。
- 384 组 benchmark 配置、运行器、对比工具和报告工具。
- Apache-2.0 许可证、README、贡献指南、支持矩阵和申报书草稿。
- C500 环境记录及 384 组结果文件。
- C500 MetaX provider 路径 37 项测试通过。
- C500 强制 ATen 路径 37 项测试通过。
- 本地测试 30 项通过，8 项 C500 专项测试跳过。

### 尚未完成

- vLLM 或 SGLang 端到端模型生成。
- vLLM AttentionBackend 正式注册。
- Qwen2.5-1.5B/7B 的完整生成 smoke test。
- 项目自研融合 MXMACA FlashAttention kernel。
- GitHub 和 GitLink 的正式公开发布。
- 演示视频和最终申报提交材料。

## 三、第一阶段：发布前收尾

目标：把当前代码整理成可审阅、可复现、可公开发布的首版。

### 任务

- 在 `requirements-c500.txt` 中加入 `setuptools>=77`。
- 确认 README 的安装命令与 C500 构建要求一致。
- 让 benchmark 绘图和统计工具过滤 `backend=reference` 或存在 `fallback_reason` 的样本。
- 为 fallback 过滤补回归测试。
- 用最新 runner 重新生成 384 组 C500 JSON 和 Markdown 报告。
- 更新 `PROGRESS.md`、`docs/seed_application.md` 和 README 中的测试数量、seed、版本和结果口径。
- 检查仓库不包含密码、SSH 信息、模型权重或实例信息文件。

### 验收标准

- 本地 pytest 全部通过。
- C500 MetaX provider 测试全部通过。
- C500 ATen extension 测试全部通过。
- benchmark JSON 包含设备、显存、PyTorch、MXMACA、驱动、FlashAttention、vLLM、seed、warmup、repeats 和 fallback 状态。
- fallback 样本不会进入 native 性能图和 native 性能汇总。
- 384 组结果与当前代码 schema 一致。

## 四、第二阶段：GitHub 发布

目标：先建立公开主仓库，形成可审阅版本。

### 发布内容

- `README.md`
- `LICENSE`
- `CONTRIBUTING.md`
- `mxflashattn/`
- `csrc/mxmac/`
- `benchmarks/`
- `integrations/`
- `tests/`
- `docs/`
- CI 配置和依赖清单

### 发布说明必须明确

- C500 已验证的功能范围。
- benchmark 基线是 PyTorch reference。
- 当前优化执行依赖 MetaX flash-attn wheel。
- ATen extension 不是自研融合 kernel。
- vLLM/SGLang 端到端生成尚未完成。
- fallback 只能通过环境变量显式开启。

### 验收标准

- GitHub 仓库可以从干净环境安装。
- README 示例可执行。
- CI 通过本地 CPU/reference 测试。
- GitHub 仓库不包含凭据和大模型权重。

## 五、第三阶段：GitLink 镜像

目标：将与 GitHub 相同的首版代码同步到 GitLink，满足项目申报和成果沉淀要求。

### 任务

- 确认 GitLink 仓库名称、组织和默认分支。
- 从 GitHub 镜像同一版本。
- 检查 LICENSE、README、benchmark 结果和申报材料均可访问。
- 在申报材料中填写 GitHub/GitLink 地址。

### 验收标准

- GitLink 与 GitHub 的提交版本一致。
- GitLink 可以正常浏览源码和文档。
- 不上传实例信息、密码、模型权重或临时日志。

## 六、第四阶段：vLLM/SGLang 端到端生成

目标：使用真实模型证明 MXFlashAttn 能参与完整推理流程。

### 优先顺序

1. 检查 C500 当前 vLLM 版本是否能加载目标模型。
2. 如果版本兼容，先完成 vLLM 适配和 Qwen2.5-1.5B 生成 smoke test。
3. 如果 vLLM 受运行时限制，记录版本和错误原因。
4. 检查 SGLang 是否可安装并支持当前 MXMACA/C500 环境。
5. 如果 SGLang 可用，优先完成 SGLang 的最小生成 demo。
6. 最后使用 Qwen2.5-7B-Instruct 做展示性验证。

### 必须记录

- 模型名称和版本。
- 推理框架版本。
- prompt 和生成结果。
- 首 token 延迟。
- decode latency。
- tokens/s。
- 峰值显存。
- 实际使用的 attention backend。
- 失败时的完整兼容性原因。

### 验收标准

- 至少一个真实模型能在 C500 上稳定生成文本。
- 生成过程确认调用了 MXFlashAttn 或明确记录未调用的原因。
- 提供可重复的命令和 smoke test 脚本。
- 结果加入 README、支持矩阵和申报材料。

## 七、第五阶段：性能和内核增强

目标：减少当前未达到 20% 延迟下降的场景，并逐步替换 ATen bring-up 路径。

### 优先方向

- decode 小 query length 场景。
- varlen 非 causal 场景。
- paged KV 的页面访问和 cache 更新。
- GQA/MQA 的 head 映射。
- C500 原生 fused kernel。

### 后续路线

- backward。
- FP8。
- 多卡。
- 更完整的 vLLM paged metadata 适配。
- SGLang 原生 backend。

这些能力在完成真实验证前，只能列为路线图，不能写入已完成成果。

## 八、最终申报材料

申报材料应按以下顺序整理：

1. 项目简介。
2. 国产算力 MXMACA/C500 适配价值。
3. FlashAttention 兼容 API 和 paged KV 能力。
4. 384 组可复现实验和 C500 环境信息。
5. 精度、延迟、吞吐和显存结果。
6. vLLM/SGLang 集成现状，明确已完成和未完成部分。
7. GitHub/GitLink 地址。
8. 阶段成果截图。
9. 端到端生成演示视频。
10. 后续 backward、FP8、多卡和 fused kernel 路线。

## 九、发布决策

当前建议采用以下决策：

- **先完成第一阶段收尾。**
- **随后发布 GitHub 首版。**
- **GitHub 版本确认无误后同步 GitLink。**
- **发布后立即进入 vLLM/SGLang 端到端生成工作。**
- **所有尚未实测的能力继续标记为后续目标。**

这样可以尽快形成真实、可审阅的开源成果，同时避免把尚未完成的端到端能力包装成已交付功能。
