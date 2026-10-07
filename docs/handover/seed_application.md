# MXFlashAttn 青年开源专项基金种子计划申报书（草稿）

> **材料状态：项目阶段申报草稿（2026-10-05 更新）。** 本机测试 31 项通过，另有 8 项 C500 专项测试在本机跳过；远端 C500 上 MetaX provider 与仓库 C++ ATen extension 两条路径各 39 项通过。C500 全矩阵 384/384 组已运行并留存原始 JSON。该结果使用 MetaX 构建的 `flash-attn` wheel；本项目的 C++ ATen 扩展不是融合 FlashAttention kernel。decode 场景中位延迟下降 51.64%，185/192 组达到 20% 下降，仍有单项未达标。当前没有 vLLM/SGLang 端到端生成结果，也未完成 GitLink 镜像。

## 一、项目简介

MXFlashAttn 是一个面向沐曦 MXMACA/C500 的 FlashAttention 兼容 API 与推理适配项目，聚焦大模型推理中的 Attention 前向计算与 KV cache 访问。当前已提供参数校验、PyTorch reference 和显式 fallback，并优先适配经 C500 验证的 MetaX `flash-attn` wheel；仓库内 C++ ATen extension 已在 C500 构建并单独执行测试。项目贡献集中在兼容 API、后端选择、缓存语义统一和可复现验证，不将 ATen 路径表述为自研融合内核。

首期目标范围为单卡 C500、FP16/BF16、head dimension 64/128，并围绕 causal attention、GQA/MQA、变长序列和 paged KV cache 建立逐步交付的支持矩阵。反向传播、FP8 和多卡能力不纳入首期承诺。

**已实现并验证：** 三个公开入口、参数校验、reference 执行、显式回退、dense/varlen/paged KV 语义已实现。本机 31 项测试通过；C500 上 MetaX provider 与 C++ ATen extension 强制路径分别运行 39 项测试并通过。C500 基准对比 reference 的最大绝对误差为 0.0078125，测试容差为 `atol=0.04, rtol=0.04`。优化执行使用 MetaX wheel；项目尚未自研融合内核。

**待推进：** 当前 384 组基准的 decode 中位数下降 51.64%，185/192 组达到 20% 目标，但仍有 7 组未达目标。vLLM 运行时版本不匹配，SGLang 未安装；两者的平台适配与模型端到端生成均未完成。

## 二、项目痛点与需求

Attention 是大模型 prefill 与 decode 阶段的核心算子之一。序列增长和 KV cache 访问会带来显存带宽与延迟压力；GQA/MQA、变长批处理和 paged KV 等常见推理场景也要求算子与上层推理引擎协同工作。

国产算力生态建设还需要可复用、可验证的算子适配和公开的兼容性证据。当前项目已在 C500 上运行 dense、varlen、GQA/MQA 和 paged KV 场景，并通过全量 384 组 benchmark 记录性能、误差与显存。报告同时标明基线是本项目 PyTorch reference，且逐项保留性能回退结果；该证据不等同于自研融合内核或完整模型推理成果。

## 三、建设目标与创新点

### 建设目标

1. 持续完善与 FlashAttention 常用调用方式相兼容的前向 Python API，并对 dtype、shape、设备和暂不支持的参数进行明确校验；三个公开入口和本地校验/reference 路径已实现。
2. 在 MXMACA/C500 上验证首期 Attention 前向及 decode 路径，首批 FP16/BF16、causal、GQA/MQA、varlen 和 paged KV 场景已通过 MetaX provider 与 ATen bring-up 路径测试。
3. 维护并验证由用户显式开启的 PyTorch reference 回退（`MXFLASHATTN_ALLOW_FALLBACK=1`）；该回退已实现，须保持可观测并在评测结果中注明原因，不支持静默改变执行路径。
4. 优先验证固定版本的 vLLM 集成；若 C500 运行时条件阻碍集成，则保留相同 API，并记录原因后评估 SGLang 或独立推理 demo。
5. 扩展已实现的 benchmark 工具与 384 组配置矩阵，至少完成 30 组目标硬件实测，并报告误差、吞吐、首 token 延迟、decode latency、峰值显存、回退状态及完整软件/硬件版本信息。

### 预期创新点

- **国产算力适配：** 在 MXMACA/C500 上适配 MetaX 优化 wheel 与项目 ATen 扩展，产出可复现的硬件验证和支持矩阵；项目自研融合内核仍属后续研发方向。
- **兼容接口与清晰回退：** 对外提供熟悉的 FlashAttention 风格 API，以校验和显式回退改善上层调用的可预期性。
- **面向推理路径：** 首期优先覆盖 decode、GQA/MQA 和 KV cache 相关场景，并推进主流推理引擎集成。
- **证据随代码交付：** benchmark 脚本、报告工具、384 组原始 C500 数据和摘要已进入仓库，记录精度、延迟、decode、显存和环境版本。

## 四、技术路线

1. **API 与校验层（已实现）：** `mxflashattn/api.py`、`validation.py` 提供三个公开入口和输入检查，对暂不支持的参数明确报错；31 项本机测试通过。
2. **参考实现与调度（已实现）：** PyTorch reference 覆盖 dense、varlen 和 dense/paged KV cache 参考执行。dispatch 默认不静默回退，只有设置 `MXFLASHATTN_ALLOW_FALLBACK=1` 才运行 reference，并记录 backend 与 fallback 原因。
3. **C++/MXMACA 路径（已完成首轮验证）：** `csrc/mxmac/` 中的 C++ ATen extension 已用 C500 工具链构建，并在 `MXFLASHATTN_BACKEND=aten` 下通过 39 项测试。它通过 ATen 运算，不是融合 FlashAttention kernel。
4. **推理引擎适配（部分实现）：** `integrations/vllm/` 有 dense Q/K/V callable bridge 和版本 preflight，但未注册为 vLLM AttentionBackend。测试镜像含 vLLM 0.17.0 / vllm_metax 0.17.0，而仓库目标固定为 vLLM 0.30.0；SGLang 未安装。paged metadata 和模型生成 smoke test 尚未完成。
5. **自动化评测（已实测）：** `benchmarks/configs/c500.yaml` 定义并运行了 384 组组合，报告包括误差、吞吐、延迟、decode latency、reference/candidate 显存、fallback 状态和软件版本。算子 microbenchmark 的首 token 延迟为 null，需要后续模型生成测试补充。

## 五、实施里程碑（2026-10-04 至 2026-10-31）

| 时间 | 计划交付 | 状态口径 |
| --- | --- | --- |
| 10 月 4 日—6 日 | 接入 C500 环境；记录驱动、MXMACA、PyTorch、编译工具链版本；建立可复现 baseline | **已完成首轮环境记录和 384 组 baseline/candidate 对比**；设备、驱动和工具链版本见结果文件 |
| 10 月 7 日—12 日 | 完成公开 API、参数校验、reference backend、显式 fallback、基础测试与 CI | **API/reference 原型已实现；本机 31 项测试通过** |
| 10 月 13 日—19 日 | 完成 C500 前向执行验证，检查 FP16/BF16、head dimension 64/128、causal、GQA/MQA 与 varlen 支持子集 | **MetaX provider 全 384 组运行；ATen extension 构建并通过专项测试**；项目融合内核仍未实现 |
| 10 月 20 日—24 日 | 完成 paged KV/decode 路径的目标设备验证，推进 vLLM 平台适配；若运行时不兼容，记录原因并评估 SGLang 或独立 demo | Python reference 已有 paged KV；**C500 验证、vLLM 注册适配及生成 smoke test 仍为阶段目标** |
| 10 月 25 日—28 日 | 运行至少 30 组硬件 benchmark；整理复现配置、误差/性能报告、文档和版本 | **提前完成 384 组 C500 benchmark；decode 整体中位改善达标，部分单项未达 20%** |
| 10 月 29 日—31 日 | 完成 GitLink 镜像、申报材料、阶段成果说明和演示材料 | **阶段目标；发布前核对实际仓库、硬件评测和集成证据** |

## 六、验收指标与证据

| 验收项 | 目标/判定方式 | 当前状态 |
| --- | --- | --- |
| API 与输入校验 | 三个公开入口可导入；支持范围有测试；非法或暂不支持组合明确报错 | 已实现；本机 31 项测试通过，C500 两条 provider 路径各 39 项通过 |
| fallback | 默认不静默回退；仅显式设置环境变量后启用；输出/报告记录 fallback 与原因 | 已实现并有 CPU 与 C500 专项测试 |
| C++/ATen extension | 源码能够在目标 MXMACA/C500 环境构建运行；不将通用 ATen 路径表述为融合内核 | C500 构建和 ATen 强制路径测试通过；不是融合 FlashAttention kernel |
| 正确性 | 支持矩阵内各场景与 reference 对比，记录最大绝对误差、相对误差及判定容差 | 384 组 C500 记录；最大绝对误差 0.0078125，C500 测试容差 `atol=0.04, rtol=0.04`；相对误差按 reference 值 1e-6 floor 计算，近零项会放大 |
| 覆盖度 | 至少 30 组公开、可重复的配置，覆盖计划列出的场景维度 | 384/384 组已运行，覆盖 batch 1/2、query 1/64、KV 128/512、head_dim 64/128、GQA 1/4、三种布局、两种 dtype 和 causal |
| 性能 | 记录 tokens/s、首 token 延迟、decode latency、峰值显存；重点 decode 场景目标为相对同环境 reference 延迟降低 20% 以上 | decode 中位降低 51.64%，185/192 组达到 20%；7 个未达标单项保留。算子 microbenchmark 不提供模型首 token 延迟 |
| 环境可追溯 | 每条记录包括 C500/驱动、MXMACA、PyTorch、推理引擎版本、配置、fallback 状态和运行命令 | 原始 JSON 记录 MetaX C500、驱动 3.8.30、MXMACA 3.5.3.20、PyTorch 2.8.0+metax3.5.3.9、vLLM 0.17.0 和 fallback 状态 |
| vLLM/SGLang 集成 | 注册可用的推理引擎 attention backend，处理平台元数据和 KV cache，并通过模型生成 smoke test | 仅有 dense vLLM callable bridge 与版本 preflight；C500 运行时 vLLM 0.17.0 与目标 0.30.0 不同，SGLang 未安装；未注册 backend，暂无生成 smoke 成果 |
| 端到端演示 | 至少一个模型完成生成 smoke test；优先 Qwen2.5-1.5B 快速验证，条件允许时演示 Qwen2.5-7B-Instruct；权重不进入仓库 | 阶段目标；未完成前不得宣称通过 |
| 开源交付 | Apache-2.0 许可证、安装与使用文档、测试/benchmark、已知限制、贡献说明；完成 GitLink 镜像 | LICENSE、README、测试、benchmark 和贡献说明已在工作区；GitLink 镜像及最终发布材料仍待完成 |

![C500 benchmark summary](results/c500-speedup-summary.png)

## 七、风险与应对

- **工具链或运行时不兼容：** 先锁定并公开实际环境版本，逐步构建最小算子；遇阻时保留错误日志、缩小支持矩阵，并按计划评估 SGLang 或独立 demo。
- **单项性能目标未达到：** 当前 384 组中有 10 组低于 20% 延迟下降，其中 decode 有 7 组低于 20%；报告保留逐项数据，后续优先优化这些场景。
- **数值误差或功能组合复杂：** 逐项扩展功能矩阵，reference 对照并测试边界输入；未通过的组合保持禁用或明确报错。
- **周期紧、硬件访问有限：** 先完成 API、测试与可复现工具，再安排硬件验证；无法在申报节点前验证的能力列为后续目标。
- **上游接口变化：** 固定首期推理引擎依赖版本，在兼容性文档中记录版本和测试结论；升级通过独立验证后再纳入支持范围。

## 八、开源治理与成果管理

- 采用 Apache-2.0 许可证；第三方代码、依赖和引用实现逐项核对许可证并保留声明。提交前检查许可证兼容性，避免复制未授权代码。
- 项目公开托管于 GitHub/Gitee 等代码平台，并按计划同步至 GitLink；仓库公开支持矩阵、版本、构建步骤、benchmark 原始记录和已知限制。
- 设至少一名核心维护者负责发布、代码审查和问题响应；团队人数按申报要求控制在 3 人以内。贡献通过 issue/PR 留痕，关键变更需审查和测试。
- 不将模型权重、用户数据、访问凭据或远程环境密钥提交仓库。性能报告注明设备及软件环境，任何数字均可追溯至配置和运行记录。
- 发布节奏采用阶段版本：API/reference 验证版、C500 原生验证版、推理集成与 benchmark 版本。每版列出已验证能力、限制和兼容环境，不把路线图功能标作已交付。

## 九、当前状态与阶段成果口径

- **已实现并验证：** Python API、参数校验、PyTorch reference、显式 fallback、MetaX FlashAttention provider、C++ ATen extension C500 构建和运行测试；本机 31 项通过，C500 两条 provider 路径各 39 项通过。
- **硬件成果：** 已完成 384 组 C500 benchmark；decode 中位延迟降低 51.64%，185/192 项达到 20%，最大绝对误差 0.0078125。原始数据与报告位于 `MXFlashAttn/docs/results/`。
- **仍待完成：** 项目自研融合 MXMACA kernel、vLLM AttentionBackend 注册、Qwen 模型生成 smoke test、GitLink 镜像、申报书最终版和演示材料。
