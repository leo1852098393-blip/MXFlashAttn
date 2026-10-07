# MXFlashAttn 持续维护与分阶段发布计划

更新时间：2026-10-05

## 一、发布原则

MXFlashAttn 的 GitHub 和 GitLink 已经完成首版公开发布。后续测试、性能补测和代码改进先在本地工作区与 C500 实例中积累，不在每次短周期实验后立即同步到公开仓库。

公开仓库要反映真实的工程维护过程：每次发布都应包含一组有明确主题的改动、对应测试证据、文档更新和变更说明。发布节奏以实际开发周期为准，不为了制造提交数量而频繁提交空洞变更。

## 二、当前公开基线

- GitHub：`https://github.com/leo1852098393-blip/MXFlashAttn`
- GitLink：`https://gitlink.org.cn/Leo77/MXFlashAttn`
- 当前公开版本提交：`a02052d`
- 当前公开能力：API、reference、显式 fallback、MetaX provider、ATen bring-up 路径、384 组 C500 结果、双语 README 和 CI。
- 当前公开限制：vLLM AttentionBackend 尚未注册，没有端到端 Qwen 生成结果，仓库 ATen 路径不是自研融合 FlashAttention kernel。

当前基线不因后续本地实验自动改变。任何新增结果先保存在外层材料目录或实例的 `artifacts/` 中。

## 三、本地迭代与公开发布的边界

### 本地版本目录命名

公开基线固定放在 `MXFlashAttn/`。每个后续阶段先复制为独立的本地版本目录，目录名使用：

```text
MXFlashAttn_<版本号>-<阶段>
```

示例：

```text
MXFlashAttn_v0.2.0-dev       64G 长上下文和 paged KV 阶段
MXFlashAttn_v0.3.0-model     真实模型 baseline 阶段
MXFlashAttn_v0.4.0-vllm      vLLM/SGLang 适配阶段
```

版本目录只用于本地开发和验证，不直接作为公开仓库的嵌套目录，也不把多个版本目录一起提交到 GitHub/GitLink。每个版本目录都应保留对应的实验结果、环境记录和变更说明。

一个阶段完成后，按以下顺序处理：

1. 在版本目录完成代码、测试、benchmark 和文档。
2. 复核与公开基线的差异，删除临时文件、模型权重和敏感信息。
3. 将已验证改动合并到 `MXFlashAttn/`。
4. 在 `MXFlashAttn/` 完成本地测试和发布前检查。
5. 形成一个有主题的提交和版本标签。
6. 同步 GitHub 与 GitLink，并记录 release note。
7. 保留本地版本目录作为归档，或在确认结果已归档后压缩保存；不要让多个开发副本长期混在项目根目录。

### 本地阶段

- 在 `MXFlashAttn/` 工作区开发和测试。
- 在 64G C500 实例运行补测、模型实验和兼容性验证。
- 将原始 JSON、Markdown、日志和截图保存到外层 `64G_results/` 或本地归档目录。
- 记录实验日期、代码提交、硬件、驱动、MACA、PyTorch、vLLM、模型版本、seed 和命令。
- 实验结果未经过复核前，不写入公开 README、申报书或公开 benchmark 目录。

### 发布阶段

满足以下条件后，才形成一次公开小版本：

- 一个主题的功能或验证已经完成，而不是只有单次试验结果。
- 本机测试和目标 C500 测试均有结果。
- 新增 benchmark 已有可复现配置和原始数据。
- README、支持矩阵、PROGRESS 和申报材料的状态口径一致。
- 已检查敏感文件、凭据、模型权重和实例信息没有进入提交。
- 变更能够写成清晰的 release note。

## 四、建议迭代节奏

### Iteration 1：64G 硬件验证

建议在积累一段完整实验周期后发布，内容包括：

- 64G 长上下文 paged KV 结果。
- FP16/BF16、head_dim 128、batch 1/2 的显存和延迟数据。
- 误差统计改进：绝对误差、RMS/P95/P99，以及有效分母相对误差。
- 与当前 32G 结果的环境和指标对照。

### Iteration 2：真实模型 baseline

- Qwen2.5-1.5B 或兼容的本地模型生成记录。
- TTFT、decode tokens/s、总延迟、峰值显存和生成配置。
- Torch SDPA 或 MetaX vendor baseline 对比。
- 明确区分 baseline、独立算子 demo 和 MXFlashAttn candidate。

### Iteration 3：推理引擎适配

- 固定一个实际可用的 C500 vLLM 或 SGLang 版本。
- 完成平台版本检查、KV cache 元数据和 paged block table 适配。
- 通过至少一个真实模型 smoke test 后再声称集成完成。
- 适配未完成前，只更新兼容性记录，不改变 README 的能力口径。

### Iteration 4：工程与性能优化

- 优化未达到目标的 decode 组合。
- 改进 benchmark 统计和报告可视化。
- 增加回归测试、安装文档和故障排查说明。
- 自研融合 MXMACA kernel 作为独立的大版本路线，不与短期适配混在一起。

## 五、版本与分支规则

- `main` 只接收经过验证、可公开复现的阶段版本。
- 新实验先保留在本地分支或未发布提交中，例如 `experiment/c500-64g`、`experiment/vllm-baseline`。
- 一个迭代主题完成后合并为一个或少量有意义的提交，并创建版本标签，例如 `v0.2.0-c500-64g`。
- 每次公开发布同时更新 `PROGRESS.md`、`docs/support-matrix.md` 和对应结果报告。
- GitHub 和 GitLink 在同一个阶段版本完成后一起同步，避免两个平台内容不一致。

## 六、评委可见的维护证据

持续维护应由真实工程证据体现：

- 随时间增加的 issue、PR、测试和 benchmark 报告。
- 每次版本的环境信息和复现命令。
- 对已知限制和失败实验的公开记录。
- 小而明确的功能改进，而不是重复修改 README 或制造无意义提交。
- 版本之间可追溯的性能、精度和兼容性变化。

## 七、当前暂不公开同步的内容

以下内容先留在本地，等对应迭代完成后统一整理：

- 64G 长上下文结果：外层 `64G_results/`。
- Qwen/vLLM 实验日志和模型运行缓存。
- 未复核的 baseline 数据。
- 实例环境、连接信息、密码、SSH 密钥和私人令牌。
- 仅用于排错的临时补丁和编译产物。

本文件位于项目外层，不提交到 GitHub 或 GitLink。
