# MXFlashAttn 攻关过程交代文档（2026-10-06）

> 本文只记录从开始到现在**实际做了什么、试了什么、结果是什么**，不含任何建议、预判或后续规划。给接手 AI 作为事实背景。

---

## 0. 项目是什么

- MXFlashAttn = MetaX（沐曦）C500 GPU 上的 `flash-attn` 兼容层。对外提供与原版 flash-attn 一致的 API（`flash_attn_varlen_func` / `flash_attn_with_kvcache` 等），底层把活派给 MetaX 官方 vendor wheel（`metax_vendor` / `_C.so`）或 ATen 兜底。
- 目的是让在 NVIDIA 上写好的 vLLM / SGLang 注意力代码，能在国产 C500 上不改代码跑起来（dispatch 兼容）。
- 申报背景：CCF / 木兰开源社区 / 沐曦联合的"青年开源专项基金种子计划"（基础级 4000×50、重点级 12000×30，第二批窗口 2026-09-01 ~ 2026-10-31）。
- 项目由 Codex（AI）生成，非用户手写。

---

## 1. 实机环境（C500 容器）

| 项 | 值 |
|---|---|
| 访问 | 远程 C500 实例（SSH 端点与凭据不入库；本机实例信息文件与驱动脚本同样不入库，见 .gitignore 凭据类规则） |
| GPU | MetaX C500，63.59 GiB |
| MACA | 3.5.3.20，驱动 3.8.30 |
| PyTorch | 2.8.0+metax3.5.3.9 |
| vLLM | 0.17.0 |
| flash-attn | 2.6.3+metax3.5.3.9torch2.8 |
| 解释器 | 远程 `/opt/conda/bin/python`（PATH 里没有裸 `python`） |
| 后端 | `is_available()=True`，dispatch 优先级 `vendor_direct → ATen 兜底 → reference` |

**实测中确认的环境事实（复跑需满足，否则结果无效）：**
- 远程没有裸 `python`，只有 `/opt/conda/bin/python`。
- 跑 vLLM 前必须 `export MACA_PATH=/opt/maca`（triton 的 metax backend 读 `MACA_PATH`，不是 `MACA_HOME`），否则起不来。
- vLLM 0.17 的 `EngineCore` 是子进程，不继承主进程 `PYTHONPATH`。插件要在子进程生效，必须 `export PYTHONPATH=/root/<工作目录>` 再起 LLM。否则插件未加载，测出的是裸跑 vLLM 的"假持平"。
- 跑测量脚本前必须确认 `artifacts/*.plugin.log` 出现 `plugin_imported`。

---

## 2. 工作日志（按时间阶段，纯事实）

### 阶段 A：仓库审查（未上机）
- 锐评 GitHub 仓库 `leo1852098393-blip/MXFlashAttn`。
- 用户确认项目由 Codex 生成。
- 审查 `mx.zip` 多个快照，观察到的客观事实：
  - 代码库含 11 个版本文件夹（v1.0.0 至 v1.1.0），文件内容基本未变。
  - RELEASE_NOTES 写"性能 +32.74%"，实测（后上机）为慢 39.7%，文档与代码不一致。
  - 仓库含空 `.git`，commit 时间戳与代码改动不对应。

### 阶段 B：上机实机验证 + 性能优化（工作目录 `/root/MXFlashAttn_v1.1.0-vendor-wiring`）
- 原始 v1.1 端到端（Qwen3-0.6B，36 prompt × 16 token，中位）：candidate 47.52 vs vendor 79.48 → 慢 40.21%。
- 逐轮修改与结果（证据：`RESULTS_20261006.md` / `HOTPATH_OPTIMIZATION.md`）：

| 轮次 | 修改内容 | candidate 中位 | 差值 |
|---|---|---|---|
| v1.1 原始 | — | 47.52 | 慢 40.21% |
| 第一轮 | 去掉设备→主机同步校验（`.cpu().tolist()`） | 59.89 | 慢 24.64% |
| 第二轮 | 去掉 `.clone()` / 不必要的 `.to()` / import / 写盘 | 71.70 | 慢 9.78% |

- 修改清单（均已落在工作目录）：
  - `mxflashattn/api.py`：深校验改为 `MXFLASHATTN_STRICT_VALIDATION=1` 才启用，默认只做廉价形状/dtype 校验。
  - `integrations/vllm/decoder_adapter.py`：新增 `fast_layout()`（不回读主机）；去掉 `cache_seqlens.clone()`；`.to()` 仅在 device/dtype 不匹配时调用；`DecoderLayout` 加 `__slots__`。
  - `mxflashattn/dispatch.py`：保留 `add_()` 原地语义（`k is not None` 分支才改，`k is None` 的 decode 路径不触发）。
  - `integrations/vllm/vllm_plugin.py`：热路径 `__import__` 改为模块级 import。
  - `integrations/vllm/backend.py`：事件序列化缓存；新增 `MXFLASHATTN_DISPATCH_SAMPLE=N` 采样写盘；`clear_dispatch_events(truncate_log=)` 参数。
  - `integrations/vllm/smoke_generate.py`：验收断言改认 v1.1 事件名 `vendor_direct` / `mxmac_aten_correctness`；修复多 prompt 重复计入事件（文件从 40MB/279720 行降到 2.2MB/16128 行）；每条 prompt 只读取自己新追加的日志行。
- 探针实测根因（fp16 / B=1 / H=16 / KVH=8 / D=128）：`seq_lens` / `block_table` 本就在 `cuda:0` 且为 int32，`.to()` 是 no-op；最大单笔开销是 `cache_seqlens.clone()`（0.045–0.072 ms/次）。
- 可观测性开销实测：同代码，完整逐事件写盘 = 62.53（慢 21.32%），采样（1/2000）= 71.70（慢 9.78%）。
- 分层微基准（优化后）：vendor 直调 0.0634 ms / 经 API 0.0783 ms / 经 adapter 0.1003 ms。
- 算子级公平矩阵（36 组）：MXFlashAttn API 比 vendor 裸调慢 25.45%（中位 0.1338 vs 0.1067 ms），36/36 组均不快于 vendor。
- 回归：38 passed / 4 failed（4 个为 v1.0 时代写死断言，与优化无关）。

### 阶段 C：反超实验（工作目录 `/root/MXFlashAttn_v1.2.0-cudagraph-exp`，从 v1.1.0 复制，隔离，未发布）
- 新增开关：`MXFLASHATTN_EAGER=0`（开 CUDA graph）、`MXFLASHATTN_GRAPH_DECODE=1`（图安全 decode 路径）、`MXFLASHATTN_TRANSPARENT=1`（透传）。
- 观察到的事实：`smoke_generate.py` 第 125 行 `LLM(..., enforce_eager=True, ...)`——vendor 基线（vllm-vendor-36.json）和 candidate 此前均为 eager 模式，CUDA graph 从未启用。
- 实验时间线（证据：`experiment-graph/EXPERIMENT_REPORT.md`）：

| 配置 | tok/s | 关键事实 |
|---|---|---|
| 原生 baseline eager（原发布数字） | 79.48 | 基准 |
| 我们 candidate eager（对照） | 71.13 | `verified=True`，走我们 decode，慢 10.51% |
| 我们 candidate 开 graph（旧 decoder_adapter） | 263.81 | `verified=False`，dispatch 仅 1 条 `vendor_prefill_or_blocked`，decode 退回原生 native |
| 原生 baseline 开 graph | 274.58 | — |
| 我们 candidate 图安全路径 `cand-graph2` | 270.31 | `verified=True`，dispatch `{vendor_direct:1}`，走我们 decode |
| 我们 candidate 去 `.contiguous()` `cand-graph3` | 99.29 | dispatch 退回 `mxmac_aten_correctness` |
| 原生 baseline graph 重跑 `base-graph2` | 275.56 | 原生自身 run-to-run 方差约 0.36% |

- 关键事实记录：
  - 旧 decoder_adapter 开 graph 跑出 263.81，但 `verified=False`，因 `decode_forward` 的 `max_query_len != 1` 硬断言把 graph 抓图（dummy 形状 `max_query_len=15`）拒了，`MXFlashAttnImpl.forward` 捕获异常后 `super().forward()` 退回原生 native。263.81 实为"原生 native + 我们 fallback 多一层 `output.copy_`"。
  - 探针（/tmp/graph_probe.log）抓到抓图期 metadata：`max_query_len=15, query_shape=(16,16,128), seq_lens_shape=(1,)`。
  - 图安全路径 `_decode_forward_graph`：batch 取 `query.shape[0]`（恒定）；`cache_seqlens = seq_lens.expand(batch).contiguous()`、`block_table = block_table.expand(batch,B).contiguous()`。结果 `cand-graph2` 的 `verified=True`，确认走我们 backend。
  - 去 `.contiguous()` 的 `cand-graph3` dispatch 退回 `mxmac_aten_correctness`，说明 `flash_attn_with_kvcache` 拒非连续张量。
- `MXFLASHATTN_TRANSPARENT=1` 直接崩，报错 `cache_seqlens must be one-dimensional`，未修复。

### 阶段 D：当前状态（事实，未执行后续）
- 申报材料（seed_application.md / RELEASE_NOTES / VIDEO_SCRIPT 等）仍写旧数字（+32.74% / 慢 39.7% / 46.15 vs 76.46），未回填为实测值（+25.45% 算子层 / 慢 9.78% 端到端 / 71.70 vs 79.48）。
- 4 个过时断言未更新，CI 为红。
- 明文 SSH 凭据仍在 `Downloads\mx\64G实例.txt`，未删。
- 发布动作（release/tag/GitLink/视频/真实 commit）未做。
- 实验目录 `/root/MXFlashAttn_v1.2.0-cudagraph-exp` 留在带 `.contiguous()` 的最佳状态，未发布、未合入 v1.1.0。

---

## 3. 试过的方法与客观结果

| # | 方法 | 结果 | 客观根因 |
|---|---|---|---|
| 1 | 去 `.cpu().tolist()` 设备同步校验 | 有效：慢 40.21% → 慢 24.64% | 每次调用 2 次 H2D 同步，移除即生效 |
| 2 | 去 `cache_seqlens.clone()` | 最有效单笔：慢 24.64% → 慢 9.78% | clone 白花（decode 路径 `k is None`，`add_()` 不触发） |
| 3 | `.to()` 仅不匹配时调用 | 微效 | seq_lens/block_table 本就在 GPU 且 int32，`.to()` 多为 no-op |
| 4 | 事件采样写盘 `MXFLASHATTN_DISPATCH_SAMPLE` | 有效且保证据 | 完整写盘慢 21.32%，采样慢 9.78% |
| 5 | CUDA graph（旧 decoder_adapter） | 假象 263.81 | 抓图被 `max_query_len!=1` 硬断言拒，退回原生 |
| 6 | 图安全 decode 路径 `_decode_forward_graph` | 真抓上 graph：270.31，`verified=True` | `query.shape[0]` + `expand().contiguous()` 广播 dummy 形状 |
| 7 | 去 `.contiguous()` 看能否抹平 1.56% | 失败 99.29 | kernel 拒非连续张量，退回 ATen |
| 8 | 透传 `MXFLASHATTN_TRANSPARENT=1` | 崩 `cache_seqlens must be one-dimensional` | 透传路径未处理 seq_lens 的 dtype/维度形态 |

---

## 4. 关键事实与约束（客观，供核对）

- **dispatch 第一优先级 `vendor_direct`**：底层调的就是原厂 kernel，耗时 = vendor kernel + 包装开销。数学上 ≥ vendor 裸调。
- **算子层 36/36 全输**：MXFlashAttn API 比 vendor 裸调慢 25.45%（中位 0.1338 vs 0.1067 ms），最好一组也慢 15.8%。
- **口径事实**：所有 ">79.48" 的数字均为 graph-vs-eager 对比（vLLM 图红利，两边同涨约 3.4 倍）。公平对比只有 graph-vs-graph（我们 270.31 vs 原生 274.58~275.56，慢 1.5~2%）和 eager-vs-eager（我们 71.13 vs 79.48，慢 10.5%）。
- **符号约定**：本文所有"快/慢 X%"指耗时变化，正号=更慢、负号=更快。
- **仓库事实**：11 个版本文件夹 + 空 `.git`，commit 时间戳与代码改动不对应。CCF 规则要求如实披露贡献、不得伪造提交记录。
- **明文凭据**：`Downloads\mx\64G实例.txt` 含 root 密码，已被使用。

---

## 5. 当前代码状态

| 目录 | 状态 | 说明 |
|---|---|---|
| `/root/MXFlashAttn`（原 git 仓库，GitLink remote） | 备份于 `/root/_backup_MXFlashAttn_20261006` | 非当前工作副本 |
| `/root/MXFlashAttn_v1.1.0-vendor-wiring` | 发布版工作目录 | 已合入阶段 B 全部优化（eager 慢 9.78%） |
| `/root/MXFlashAttn_v1.2.0-cudagraph-exp` | 隔离实验目录，未发布 | 阶段 C，含图安全 decode、TRANSPARENT 开关，已生效 270.31 |
| 本地 `mx_submission/MXFlashAttn/` | 干净提交包 | 从 Downloads\mx 拷出过滤垃圾 |
| 本地 `mx_submission/c500_evidence_20261006/` | 证据归档 | RESULTS / HOTPATH / ENVIRONMENT / fair-matrix / experiment-graph |

实验目录开关：
- `MXFLASHATTN_EAGER=0` 开 CUDA graph（默认 `enforce_eager=True`）
- `MXFLASHATTN_GRAPH_DECODE=1` 走 `_decode_forward_graph`
- `MXFLASHATTN_TRANSPARENT=1` 透传（当前崩）
- `MXFLASHATTN_STRICT_VALIDATION=1` 开深校验（默认关）
- `MXFLASHATTN_DISPATCH_SAMPLE=N` 每 N 条写盘（默认 1/2000）

---

## 6. 复跑方法（操作事实）

本地工具 `ssh_run.py`（paramiko），读一个 `.txt` 命令清单逐行在远程执行，输出落 `ssh_out.txt`：
```
& "C:\Users\seer\anaconda3\python.exe" "C:\Users\seer\WorkBuddy\2026-10-05-05-20-54\ssh_run.py" "<命令清单.txt>" <超时秒>
```
命令清单模板：
```
export MACA_PATH=/opt/maca
export LD_LIBRARY_PATH=/opt/conda/lib:$LD_LIBRARY_PATH
export PATH=/opt/conda/bin:$PATH
export PYTHONPATH=/root/MXFlashAttn_v1.2.0-cudagraph-exp
cd /root/MXFlashAttn_v1.2.0-cudagraph-exp
MXFLASHATTN_EAGER=0 MXFLASHATTN_GRAPH_DECODE=1 python integrations/vllm/smoke_generate.py --model Qwen3-0.6B --prompts 36 --max-tokens 16 --out /tmp/cand-graph.json
```
**长命令注意（实测事实）**：一次跑两个 vLLM（合计约 8 分钟）在 540s 外层超时下会 RC=1 且写盘不落盘。拆成短命令逐个跑，各自独立落盘。探活用 `echo ALIVE_MARKER > /tmp/alive` 验证 SSH 写盘。
下载证据：本地 `ssh_exp_download2.py`（paramiko，拉 `/root/.../vllm-*.json` 到本地 `experiment-graph/`）。
测量口径：Qwen3-0.6B，36 条固定 prompt（须与 vendor 基线同套），max_tokens=16，取中位 tok/s；必须以 `candidate_dispatch_verified=True` + dispatch 含 `vendor_direct` 确认走我们 backend（否则是退回原生假象）。

---

## 7. 当前未完成事项（事实清单，无优先级）

- 申报材料旧数字未回填（+25.45% / 慢 9.78% / 71.70 vs 79.48）。
- 4 个 v1.0 过时断言未更新，CI 红。
- 明文 SSH 凭据未删、未改密。
- 发布动作未做（release/tag/GitLink/视频/真实 commit）。
- 图安全 decode 路径（`_decode_forward_graph`）未合入发布版 v1.1.0，发布版开 graph 时 decode 退回原生。
- `MXFLASHATTN_TRANSPARENT=1` 崩溃未修复。
- 11 个假版本文件夹 + 空 `.git` 未整理为真实可追溯 commit。
