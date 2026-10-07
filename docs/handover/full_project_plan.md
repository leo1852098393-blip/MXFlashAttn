# MXFlashAttn 完整项目计划

更新时间：2026-10-05

## 1. 项目定位

MXFlashAttn 是一个 Apache-2.0 开源项目，面向沐曦 MXMACA/C500 提供 FlashAttention 兼容的 Attention 前向 API、KV cache 推理适配和可复现实验工具。

首期聚焦单卡 C500、FP16/BF16、GQA/MQA、causal、varlen 和 paged KV cache。backward、FP8、多卡和完整推理引擎接入不作为当前已交付能力。

## 2. 目标

1. 提供 `flash_attn_func`、`flash_attn_varlen_func` 和 `flash_attn_with_kvcache`。
2. 统一 dtype、shape、设备、序列长度、KV cache 和不支持参数校验。
3. 仅在 `MXFLASHATTN_ALLOW_FALLBACK=1` 时使用 PyTorch reference。
4. 在 C500 上验证 dense、varlen、GQA/MQA、causal 和 paged KV 场景。
5. 用固定配置、固定 seed 和公开脚本复现延迟、精度、显存和版本信息。
6. 为 vLLM/SGLang 后续接入保留稳定 API。

## 3. 当前状态

### 已完成

- 三个公共 API、参数校验、reference backend 和显式 fallback。
- MetaX `flash-attn` provider 调度和版本识别。
- C++ ATen extension 的 C500 构建与验证路径。
- dense、varlen、GQA/MQA、causal、paged KV 和 zero scale 测试。
- 384 组固定 seed C500 benchmark、Markdown 报告和性能汇总图。
- Apache-2.0、README、贡献指南、支持矩阵、申报书和 CI。
- 源码包清单、敏感文件忽略规则和发布前检查。

### 验证结果

本机测试：31 项通过，8 项 C500 专项测试跳过；Python 编译检查和 source distribution 构建通过。

C500 测试：MetaX provider 路径 39 项通过，强制 ATen extension 路径 39 项通过。384/384 组 benchmark 完成，全矩阵中位延迟下降 53.77%，decode 中位延迟下降 51.64%，185/192 组达到 20% 目标，最大绝对误差 0.0078125。

测试环境：MetaX C500、32 GiB sGPU 配额、MXMACA 3.5.3.20、驱动 3.8.30、PyTorch 2.8.0+metax3.5.3.9、MetaX `flash-attn` 2.6.3+metax3.5.3.9torch2.8。

## 4. 目录职责

### 核心 API

- `mxflashattn/api.py`：三个公共 API。
- `mxflashattn/validation.py`：输入和参数校验。
- `mxflashattn/reference.py`：PyTorch reference 实现。
- `mxflashattn/dispatch.py`：MetaX provider、ATen extension 和 fallback 调度。

### C500 后端

- `csrc/mxmac/attention_forward.cpp`：Attention forward 的 ATen 实现。
- `csrc/mxmac/paged_kv.cpp`：paged KV cache 相关实现。
- `csrc/mxmac/workspace.cpp`：扩展模块和工作区。
- `csrc/mxmac/attention.h`、`CMakeLists.txt`：接口和构建配置。

### Benchmark

- `benchmarks/configs/c500.yaml`：384 组测试矩阵和 seed。
- `benchmarks/run.py`：执行测试并记录结果。
- `benchmarks/compare.py`：比较不同环境的结果。
- `benchmarks/report.py`：生成 Markdown 报告。
- `benchmarks/plot_summary.py`：生成性能汇总图。

### 推理引擎集成

- `integrations/vllm/backend.py`：dense Q/K/V callable bridge。
- `integrations/vllm/config.py`：版本和模型配置。
- `integrations/vllm/smoke_generate.py`：生成测试入口。
- `integrations/vllm/README.md`：兼容性和限制说明。

### 测试与工程文件

- `tests/`：API、dispatch、C500、benchmark、绘图和 vLLM bridge 测试。
- `pyproject.toml`：包、依赖、pytest 和可选依赖配置。
- `setup.py`：可选 C++ extension 构建配置。
- `requirements-c500.txt`、`requirements-vllm.txt`：环境依赖。
- `.github/workflows/ci.yml`：持续集成。
- `MANIFEST.in`：源码包文件清单。

### 申报材料

- `seed_application.md`：专项基金申报书草稿。
- `MXFlashAttn/docs/support-matrix.md`：功能和硬件支持矩阵。
- `full_project_plan.md`：本文档。
- `PROGRESS.md`：项目进度和验证结果。
- `MXFlashAttn/docs/results/`：C500 原始数据、报告和图片。

## 5. 技术路线

### 阶段一：API 与 reference

完成公共 API、统一校验和 reference backend。默认没有 native backend 时抛错；只有显式设置 fallback 环境变量才使用 reference，并记录原因。

### 阶段二：C500 bring-up

优先使用带 MetaX 标识的 `flash-attn` wheel。ATen extension 用于 C500 编译、正确性和调度验证。显式 `softmax_scale=0` 强制走已验证的 ATen 路径，因为测试中的 MetaX wheel 对该值会产生 NaN。

### 阶段三：benchmark

固定 batch size、query length、KV length、head dimension、GQA 比例、布局、dtype 和 causal 配置。每组记录延迟、吞吐、误差、decode latency、显存、backend、fallback、软件版本、seed、warmup 和 repeats。

### 阶段四：推理引擎

目标版本固定为 vLLM 0.30.0，接入时处理 AttentionBackend 注册、paged KV、slot 写入、request metadata、prefill/decode 调度和 Qwen2.5-1.5B 生成 smoke test。如果 C500 运行时不支持该版本，则记录原因并评估 SGLang 或独立推理 demo。

## 6. 发布前工作

1. 确认公开仓库名称、组织和仓库地址。
2. 检查 README、支持矩阵、申报书和 benchmark 数字一致。
3. 检查 Apache-2.0 和第三方依赖说明。
4. 确认源码包不含 SSH 信息、密码、模型权重和临时文件。
5. 在干净环境执行本地测试和源码包安装测试。
6. 推送到 GitHub、Gitee 或 GitLink，并确认仓库可访问。
7. 准备项目简介、技术路线、阶段成果、benchmark 截图和演示视频。
8. 在申报表单中明确 vLLM/SGLang 尚未完成端到端验证。

## 7. 后续版本路线

### v0.2：推理引擎接入

- 完成 vLLM 适配或 SGLang 替代方案。
- 完成 paged KV 和请求元数据联调。
- 完成 Qwen2.5-1.5B 端到端生成。
- 记录首 token latency 和 decode latency。

### v0.3：性能扩展

- 优化当前未达到 20% decode 目标的 7 个场景。
- 增加更长 KV sequence、更大 batch 和更多 page size。
- 建立跨版本回归 benchmark。

### v0.4：自研融合 kernel

- 设计 MXMACA 原生 tiled/fused attention kernel。
- 与 MetaX provider、ATen extension 和 reference 分层比较。
- 增加 workspace、编译选项和 kernel 性能回归。

### 更后续版本

- backward；
- FP8；
- 多卡通信和并行；
- 更完整的训练和推理框架适配。

## 8. 验收标准

1. 三个公共 API 可导入，并对非法组合给出明确错误。
2. fallback 默认关闭，启用时有 warning 和 backend 记录。
3. C500 上 FP16/BF16、head dimension 64/128、GQA/MQA、causal、varlen、paged KV 测试通过。
4. 至少 30 组公开 benchmark；当前已完成 384 组。
5. benchmark 数据包含完整环境元数据和固定 seed。
6. 至少一个真实模型完成生成 smoke test 后，才宣称模型级推理支持。
7. 发布包不包含凭据、模型权重和临时编译目录。

## 9. 风险与应对

- **vLLM 版本不兼容：** 锁定版本并保留 preflight；无法运行时切换 SGLang 或独立 demo。
- **性能结果波动：** 使用固定 seed、warmup、repeats 和环境元数据，保留逐组结果。
- **reference 被误当作优化后端：** 报告明确标注 backend 和 fallback，性能图排除 fallback 样本。
- **native wheel 行为差异：** 对 zero scale 等边界参数单独测试，必要时选择 ATen 或明确报错。
- **发布泄密：** `.gitignore` 忽略实例信息，发布前检查源码包和 Git 状态。

## 10. 结论

MXFlashAttn 已达到“可审阅、可测试、可申报”的首期工程状态。发布前主要工作是仓库同步、材料整理和演示准备；vLLM/SGLang 端到端集成、自研融合 kernel、backward、FP8 和多卡属于后续版本，不应在当前发布中宣称已经完成。
