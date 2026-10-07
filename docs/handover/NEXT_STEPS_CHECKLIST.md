# MXFlashAttn 后续待做清单

更新时间：2026-10-05

这份清单用于 64G C500 新实例的逐项执行和核对。公开代码仍位于 `MXFlashAttn/`；本文件、申报书和项目计划放在外层，不提交到 GitHub/GitLink。

后续实验先本地归档，按主题完成一个小迭代后再统一发布。发布节奏和版本边界见外层 [`MAINTENANCE_AND_RELEASE_PLAN.md`](MAINTENANCE_AND_RELEASE_PLAN.md)。

## 一、64G 实例推荐配置

### 租用时选择

| 配置项 | 推荐值 | 说明 |
| --- | --- | --- |
| 计算平台/框架 | **vLLM** | 先复用已有 C500 环境，目标是完成真实模型生成和推理指标 |
| vLLM 版本 | **优先选择 0.17.0** | 之前 32G 实例已知可运行；不要先升级到 0.30.0 |
| 编程语言 | **Python 3.10** | 与之前实例保持一致，减少依赖和 ABI 变量 |
| 底层驱动/MACA | **MACA 3.5.3.307** | 按平台下拉选项使用；进入实例后必须记录实际 MXMACA/驱动版本 |
| PyTorch | **平台预装的 MXMACA vendor PyTorch** | 不要安装上游 CUDA 或 CPU wheel 覆盖它 |
| GPU 显存 | **64G** | 用于长上下文、paged KV 和峰值显存补测 |
| 其他平台 | 暂不选 SGLang、FlagOS、Megatron、Paddle、DeepSpeed、TileLang 等 | 这些会引入新的运行时或训练变量，后续单独评估 |

### 版本选择原则

之前实例记录为 `vLLM 0.17.0 / Python 3.10 / MACA 3.5.3.307`。新实例优先保持完全一致；如果平台只提供其他 vLLM 版本，不要直接认定兼容，先记录版本并做 preflight、import、短生成和算子测试。vLLM 0.30.0 仍是仓库的上游适配目标，但不是当前 C500 已验证版本。

## 二、实例启动后的部署顺序

- [x] 记录实例 ID、GPU 型号、显存、操作系统、Python、驱动、MACA、PyTorch、vLLM 和 `flash-attn` 版本。
- [x] 确认 `torch.cuda.is_available()`、设备名称和可用显存。
- [x] 确认 `flash-attn` 是 MetaX/MACA 构建版本，不安装通用 CUDA wheel。
- [x] 从 GitHub 或 GitLink 克隆 `MXFlashAttn`，不要复制外层申报材料到实例。
- [x] 执行 `python -m pip install --no-build-isolation -e .`；同时修复了 native extension 的相对路径安装问题。
- [x] 先运行本地/reference 测试，再运行 C500 专项测试。
- [x] 构建并验证 ATen extension，保留编译日志；不要把日志中的账号、IP 或凭据提交仓库。
- [ ] 将所有环境信息保存为本地 `artifacts/environment-64g.txt`，只提交脱敏后的结果。

## 三、第一批必须完成的重点级证据

### 1. 真实模型生成

- [ ] 下载 `Qwen/Qwen2.5-1.5B-Instruct`，权重不进入仓库。
- [ ] 先完成 vLLM 0.17.0 baseline 生成。
- [ ] 记录 prompt、输入长度、输出长度、batch size、随机参数和模型 commit/version。
- [ ] 记录 TTFT、decode tokens/s、总延迟、峰值显存和错误日志。
- [ ] 如果 MXFlashAttn 尚未注册为 vLLM AttentionBackend，必须将结果标记为 baseline 或独立算子 demo，不得写成 MXFlashAttn 端到端结果。
- [ ] 若 vLLM 集成无法在该版本运行，记录具体兼容性错误，再评估 SGLang 或独立生成 demo。

### 2. 64G 长上下文与 paged KV 补测

优先补测 8 至 12 组，不必重跑全部 384 组：

- [x] decode：`q_len=1`，KV length 4K/8K/16K。
- [x] paged KV：GQA 比例 1/4。
- [x] head dimension 128，FP16 和 BF16 各测一组或两组。
- [x] batch size 1/2，记录 tokens/s、decode latency 和峰值显存。
- [ ] 对比 PyTorch reference、Torch SDPA 或平台 vendor baseline；明确写出 baseline 类型。
- [x] 保留完整 JSON、运行命令、seed 和环境版本；结果已导出到外层 `64G_results/`。

### 3. 精度统计修正

- [ ] 报告最大绝对误差、RMS/P95/P99 绝对误差。
- [ ] 相对误差只统计 `abs(reference) > 1e-3` 的有效样本。
- [ ] 对近零分母导致的极大 max-relative-error 做单独说明，不把它作为主要精度结论。
- [ ] 统一 32G 和 64G 的误差阈值、seed 和比较方式。

## 四、申报材料更新

- [ ] 将 GitLink 状态更新为“已同步”：`https://gitlink.org.cn/Leo77/MXFlashAttn`。
- [ ] 将 GitHub 地址写入材料：`https://github.com/leo1852098393-blip/MXFlashAttn`。
- [ ] 更新 64G 实例的驱动、MACA、PyTorch、vLLM 版本。
- [ ] 加入 Qwen 生成结果、TTFT、decode tokens/s 和显存截图。
- [ ] 加入 64G 长上下文/paged KV 结果图。
- [ ] 明确写出：当前优化路径依赖 MetaX `flash-attn` wheel，仓库 ATen extension 是 bring-up/correctness 路径，不是自研融合 kernel。
- [ ] 明确写出：vLLM backend 是否已注册；没有注册就不能写“已完成 vLLM 集成”。
- [ ] 更新 `PROGRESS.md` 中“GitLink 未完成”等过期表述。
- [ ] 准备 2 至 3 分钟演示视频：环境信息、安装、API/benchmark、真实模型生成或清晰的兼容性说明。

## 五、完成判定

### 最低可提交版本

- [ ] 64G 环境信息完整。
- [ ] C500 专项测试通过。
- [ ] 至少一组真实模型生成或独立推理 demo 有完整日志。
- [ ] 至少 8 组 64G 长上下文/paged KV 数据。
- [ ] GitHub/GitLink 内容一致，许可证和敏感文件检查通过。
- [ ] 申报书所有数字、版本和仓库状态一致。

### 重点级增强版本

- [ ] Qwen2.5-1.5B 完成 MXFlashAttn candidate 与 baseline 对比。
- [ ] 有 TTFT、decode tokens/s、显存和端到端生成截图。
- [ ] 有 Torch SDPA 或 MetaX vendor baseline，而不只有 PyTorch reference baseline。
- [ ] 64G 长上下文结果显示显存收益或稳定性收益。
- [ ] vLLM 或 SGLang 至少有一个 C500 兼容路径被实际验证。

## 六、当前不作为本轮阻塞项

- [ ] 自研融合 MXMACA FlashAttention kernel：列入后续版本路线，不阻塞本次申报。
- [ ] backward、FP8、多卡：列入后续路线，不在新实例第一轮部署。
- [ ] Megatron、DeepSpeed、Paddle、ComfyUI、N8N、OpenClaw 等其他平台：与本项目当前推理目标无直接关系，暂不部署。

## 七、每一步的交付物

每完成一项，保留以下证据：

1. 命令或配置文件。
2. 原始终端日志。
3. 脱敏后的环境信息。
4. JSON/Markdown 结果。
5. 必要时的截图或短视频。

凭据、SSH 私钥、私人令牌、实例密码和完整公网地址不得写入仓库或申报材料。
