# Gate 4 参考文献与调研记录（M1 decode 内核优化）

日期：2026-10-07
范围：MXFlashAttn v1.3.0 paged-KV decode 内核（MetaX C500 / MACA）在 split-KV、
shuffle 通路、统一 max 三个方向上的资料依据。只列实际查阅并影响了实现决策的资料。

---

## 一、学术论文

**[1] Ke Hong, Guohao Dai, Jiaming Xu, Qiuli Mao, Xiuhong Li, Jun Liu, Kangdi Chen,
Yuhan Dong, Yu Wang. "FlashDecoding++: Faster Large Language Model Inference with
Asynchronization, Flat GEMM Optimization, and Heuristics."**
*Proceedings of Machine Learning and Systems (MLSys 2024)*, pp. 148–161.
arXiv:2311.01282（2023）.
PDF: https://proceedings.mlsys.org/paper_files/paper/2024/file/5321b1dabcd2be188d796c21b733e8c7-Paper-Conference.pdf

采纳点：
- **统一 max（asynchronized softmax with unified max value，第 3.1 节）**——直接催生本内核
  mode 10（UM）。论文指出 online-softmax 的逐迭代重缩放（m→o 循环携带依赖）占注意力
  计算约 20% 开销；用预设 φ 取代运行时 max 后各 chunk 可独立异步计算。我们的实现取
  φ=0（p=2^s 直接累加，循环末 o·2^−m 一次换算），在 C500 上全 12 配置提速 2–4%
  （split 路径 s1024 1.20→1.15ms，lowB NP32 0.141→0.137ms）。
- **溢出与回退机制（recompute）**——论文统一 max 的溢出保护思路（超界即重算）。
  我们的 φ=0 边界为 |s·log2e|<126，gate 数据 |s|<50 安全；生产化的 dirty 回退列为后续工作。
- **decode 是 flat-GEMM 型记忆受限负载 + 启发式数据流**——支撑了 split-KV 的 NP
  启发式（按 B·HKV 自动定 chunk 数，目标总 block ≈512）。

## 二、工程实践与社区资料

**[2] Engininja2. "cuda : use amd wave sharing intrinsics for warp_reduce functions."
llama.cpp Pull Request #6522（2024-05）.**
https://github.com/ggml-org/llama.cpp/pull/6522

采纳点：该 PR 实测确认 **AMD 上 `__shfl_xor_sync` 编译为 `ds_bpermute`（LDS/本地数据
共享通路，需 setup 指令、吃 LDS 带宽和延迟），而 DPP 寄存器级指令近乎免费**——与我方
微探针结论（C500 上 `__shfl_xor_sync` ~62–66ns/条 = FMA 的 20 倍）互相印证，说明这是
wave64 血统硬件的通病而非 C500 独有。也解释了为何 vendor 把归约放进 smem。

**[3] AMD GPUOpen. "AMD GCN Assembly: Cross-Lane Operations."**
https://gpuopen.com/learn/amd-gcn-assembly-cross-lane-operations/

采纳点：DPP（Data-Parallel Primitive）与 permute 两类跨 lane 指令的语义和成本差异，
即 PR #6522 的引用出处；`mov_shfl` 的控制码行为（组内绝对广播/相对移位）与此对照确认。

**[4] AMD ROCm Blog. "Llama.cpp Meets Instinct: A New Era of Open-Source AI
Acceleration."** https://rocm.blogs.amd.com/ecosystems-and-partners/llama-cpp/README.html

采纳点：wave64（wavefront=64）与 NVIDIA warp=32 的差异背景；早期 llama.cpp 硬编码
warp=32 导致 Instinct 上性能不佳的教训——支撑了我们"所有 shuffle 假设必须按 64-lane
重新验证"的测试原则（shfl_probe：indexed shuffle 的 src≥32 行为、xor 跨 32 边界行为）。

## 三、源码级资料（本地逆向，非公开文献）

**[5] MACA SDK 设备函数头文件** `/opt/maca/include/__clang_maca_device_functions.h`（远程实例）

发现点：
- `__shfl_xor_sync` = `__builtin_mxc_bsm_bpermute(index<<2, var)`——BSM/LDS 往返，
  微探针实测 ~66ns/条；
- **`__builtin_mxc_mov_shfl(var, ctrl, 0xf, 0xf, 0/1)`：寄存器级 16-lane 交换**，~16ns/条
  （4 倍快）。控制码 0x150+X = 组内绝对广播（封装 `__shfl_sync_16`）、0x111+D = 组内
  相对移位（封装 `__shfl_up_sync_16`），16-lane 组边界恰与 K 行线程分组（BDX=16）对齐。
- AMDGCN wave_reduce 系 builtins 在 xcore1000 后端不存在（排除一条捷径）。

**[6] MXFlashAttn v1.3.0 vendor decode 内核源码**（`csrc/` 下 decode.cuh，远程实例）

采纳点：SharedKV 结构复刻（v3 内核）——`__builtin_mxc_arrive_gvmcnt(N)` 计数等待语义
（等最老组完成）、页表 bt 预载 smem、K 块在 QK 后发射与 Vupdate 重叠。这些是 mode 3
达到全数学 1.19× vendor 的结构基础。

**[7] 自制 wavelang ISA 探针**（`wl_kernel.cpp` / `wl_binding.cpp` / `wl_test.py`，本目录）

用途：在无法用 llvm-dis 反汇编 vendor bitcode（"Invalid bitcode signature"）的情况下，
用 wavelang 内核直译 ISA 指令，实测 mov_shfl scan+广播 82ns vs BSM 263ns，坐实语义
与成本，替代反汇编证据。

## 四、检索路径备忘

- FlashDecoding++ 检索词：`flash decoding unified max softmax attention synchronization`；
  定位到 MLSys 2024 正式版（arXiv 版标题略异，引用以 MLSys 版为准）。
- AMD shuffle 通路检索词：`llama.cpp AMD wavefront shuffle ds_bpermute PR` → #6522 →
  GPUOpen Cross-Lane Operations。
- MACA 内置函数清单：直接读 SDK 头文件 + 在内核里逐个编译探测（AMDGCN builtins
  全部试探性编译一遍，确认存在性）。

## 五、结论对照表（资料 → 决策）

| 资料 | 原假设 | 决策 | 结果 |
|---|---|---|---|
| [1] FlashDecoding++ 统一 max | 逐迭代重缩放是主要开销之一 | mode 10 UM（φ=0）→ mode 11 分桶惰性缩放 | 全配置小胜 2–4%；mode 11 零代价消除 |s|<126 溢出边界（ovf 配置 UM1 F(nan)、UM2 P） |
| [2][3] AMD DPP vs ds_bpermute | shuffle 走 LDS 是 C500 特例 | 确认为 wave64 通病，转向 mov_shfl | 探针证实 4 倍差；但换归约仅 +2%（调度崩坏才是主因） |
| [4] wave64 背景 | NVIDIA warp 语义可平移 | 全部 shuffle 行为按 64-lane 重测 | 发现 src≥32 与发散 UB 两个坑 |
| [5] MACA 头文件 | 只能用 __shfl_xor_sync | 发现 mov_shfl 寄存器级通路 | 5 条 mov_shfl 替 4 条 BSM |
| [6] vendor 源码 | 调度差异无法解释 | 结构逐项对齐复刻 | 全数学 1.19× vendor，锁定剩余差距为编译器调度 artifact |
| [7] wavelang 探针 | 需要 llvm-dis | 自建 ISA 级证据 | 无反汇编也能定 mov_shfl 语义 |
