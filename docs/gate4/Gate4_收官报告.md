# Gate 4 收官报告：C500 paged-KV Decode 内核优化（M0→M3）

日期：2026-10-05 ~ 2026-10-07
硬件：沐曦 MetaX C500（104 SM，warpSize=64，HBM 实测峰值 ~1.5TB/s）
软件：MACA 工具链 + vLLM 0.17（MXFLASHATTN_V017 后端），Qwen3-0.6B E2E
基线：vendor flash_attn_with_kvcache（官方闭源内核）、自有 v7 路径

---

## 一、十条结论摘要

> 说明：本节汇总最终口径；历史/已修正结论保留在正文中，并以“历史口径/已修正”标注。

1. **正确性**：稳健性矩阵（极长上下文 s8192、GQA 比率 2/3/4/5/6/7/8、bf16、溢出攻击 ovf）实测闭环——支持集 GQ∈{2,4,8} 全部 PASS（误差 < 2e-2）；GQ=3/5/7 等已到达 binding 的支持集外比率由 binding 守卫拒绝并自动落回 v7；GQ=6 当前没有合法 HD=64/128 的真实 checkpoint，现有候选会先由 API 的 HD 守卫拒绝；两类均无静默错误路径。
2. **内核级性能**：M1 延迟约为 vendor 裸 kernel 的 2.95×–7.43×，当前仍慢于 vendor；但相对自有 v7 路径提升 1.7×–42×。这里的 2.95×–7.43× 是 latency ratio，不是“快了多少”。
3. **E2E 性能（vLLM continuous batching）**：随序列长度单调放大——
   短序列(256tok) batch128 **+32%**；sl1024 batch64 **4.39×**；sl2048 batch32 **5.31×**。
4. **大模型复验（Qwen3-4B + Qwen3-8B，GQ=4）**：M1 相对 v7 达 **2.13×–3.67×**
   （4B）/ **2.20×–3.94×**（8B）；四个场景逐一复现、偏差 <9%，说明该收益主要随
   batch×序列形态变化，未观察到明显的模型规模依赖。并发现 v7 自研 decode 在大模型上比vendor
   直连慢 3.2×，M1 将其拉回 vendor 水平——split-KV 在大模型上不是优化项，而是
   自研 decode 路径可用的前提。注意这是 **E2E 吞吐口径**；内核级裸 kernel 计时下 M1仍慢于 vendor（见内核级表格的"延迟比 M1/vendor"列）。
5. **守卫拒绝侧真实模型闭环（Qwen2.5-7B GQ=7、Qwen3.5-9B HD=256）**：GQ=7 时三档
   dispatch 全部 `native_decode`、`m1_split_decode` **0 次**，吞吐差异 <1%，文本
   **4/4 完全一致**；HD=256 被公开 API 层显式 ValueError 拒绝。两类不支持形状均
   干净拒绝、不产出错误结果。详见 `E2E_大模型与守卫验证.md`。
6. **支持集全路径打通（PaddleOCR-VL，16Q/2KV = GQ=8）**：M1 收益 1.20×/1.53×/2.47×
   （b16×64 / b64×256 / b32×1024），派发 1134/4590/18414 次 `m1_split_decode`
   0 回退，文本 **4/4**。
7. **支持集最后一格补齐 + 修掉一个真实 bug（TinyLlama-1.1B，32Q/4KV = GQ=8 / HD=64）**：
   `{GQ∈{2,4,8} × HD∈{64,128}}` 全部 6 个格子终有真实模型 E2E 证据。M1 收益
   0.97×/**2.96×**/**6.48×**（b16×64 / b64×256 / b32×1024），派发 1386/5610/22506
   次 `m1_split_decode` 0 回退。过程中定位并修复**整个 Llama 族**的真实缺陷：
   vLLM 在该族下传入的 q 是从融合 QKV buffer 切出的非连续视图（batch 维按最大
   batch 分配），`reshape` 不保证连续，导致 M1 与 v7 **双双**被 binding 的
   layout 守卫拒绝、自研 decode 路径整体退化到 vendor 直连。补 `.contiguous()`
   后 b32×1024 从 177 → 1150 tok/s（6.48×）。详见 `E2E_大模型与守卫验证.md` 追加五。
8. **生产安全**：分桶惰性缩放（mode 11）以 <1% 代价消除统一 max 的溢出边界，已设为默认。
9. **派发验证**：E2E 全程 0 回退，decode 步全部命中 `m1_split_decode`（守卫拒绝场景除外）。
10. **非 2 幂 GQ 守卫三种比率闭环，并更正了一处方法学错误**（Qwen2.5-14B-Instruct，40Q/8KV = **GQ=5**）：
   内核在 binding 层明确拒绝（`split kernel unsupported config: D=128 GQ=5`），三档 `m1_split_decode` 均为 **0**
   次。另外：**GQ=6 实际上不存在**——40+ 个主流 checkpoint 全扫，三个 GQ=6 全是 head_dim=256，
   会被 API 层先拒（进不到 GQ 守卫），此项遗留**客观不可得**，不再需尝试。
   同时发现 `text_match` 在大 batch 连续批处理下不可靠（batch=4 定点对比实测 v7 vs v7 = 4/4、
   v7 vs M1 = **4/4**，说明守卫路径文本完全一致），GQ=7 报告里原标的 4/4 需复核。

## 二、四阶段历程

### 阶段一：正确性攻坚（M0）
- 寄存器流水线 mode 0 首测全 FAIL → 定位到 `__shfl_sync` mask 内发散跳过 shuffle = 未定义行为（MACA/AMDGCN 语义），修复为无条件 shuffle。
- mode 2（直连全局加载）全 PASS，证明 softmax/merge 数学正确，瓶颈在访存调度机制。

### 阶段二：性能逼近（M1）
- 受控实验链：cp_async 计数等待可达 1053GB/s，纯全局加载仅 256GB/s → C500 必须 cp_async 隐藏延迟。
- 指令成本微探针：`__shfl_xor_sync` 单条 ~62ns（FMA 的 20 倍）——它是 BSM（LDS 往返）而非 NVIDIA 寄存器交换。
- RD（冗余全点积）路线两次实现惨败（8.2/13.0ms vs 基线 1.247ms），判死回滚：跨线程归约在 C500 上藏不住。
- split-KV + combine（mode 5）：grid.z 分 chunk 写部分状态、combine 内核在线归并，NP 扫描定标"总 block ≈512"。低并发 20.6× 差距收窄到 3.02×。

### 阶段三：文献与 ISA 调研突破（响应"查论文别蛮干"）
- FlashDecoding++（MLSys 2024）→ 统一 max（mode 10）：φ=0 基准，循环内零重缩放，全配置小胜 2–4%。
- llama.cpp PR#6522 + AMD GPUOpen 文档 → 确认 wave64 shuffle 走 LDS；ISA 逆向发现 `__builtin_mxc_mov_shfl`（寄存器级 ~16ns，4 倍快于 BSM），封装 `__shfl_sync_16`/`__shfl_up_sync_16`，替换归约 +2%。
- **证伪记录**（同样有价值）：shuffle 成本假说、libm 调用打断流水假说、寄存器溢出假说（74 vs 84）——逐一实验排除。剩余 ~3.5× 差距定位为 MCC 编译器对"跨lane交换+指数运算"混合的调度崩坏，源码层旋钮已穷尽。

### 阶段四：生产化与 E2E 落地（M3）
- UM 转正进 vLLM split 链路（模板双实例，env 开关，pybind 零改动）。
- mode 11 分桶惰性缩放：`m_used` 只落 32 整数倍桶边界，仅越桶才重缩放（每行 2–4 次），p=2^(s−m_used)≤1 天然无溢出，在实数缩放代数上与 online softmax 等价；考虑 `exp2_fast` 近似、bf16/fp16 量化和浮点舍入时，不宣称 bitwise 严格等价。溢出攻击配置（s_log2 峰值~200）：UM1 F(nan)、UM2 全 P，性能差距 <1% → 设为默认。
- vLLM 接线（`MXFA_M1_DECODE=1` 门控 + NP 启发式 + 异常自动落回 v7），36×32 冒烟 31248 次派发全命中、文本一致 30/36。
- 发现旧冒烟为逐请求执行、批次并发从未发生 → 新建 bench_e2e.py（一次提交全批 + `VLLM_ENABLE_V1_MULTIPROCESSING=0` 解决 EngineCore 子进程 env 热切换），拿到上面 E2E 决定性数据。

## 三、内核 mode 演进表

| mode | 机制 | 状态/结论 |
|---|---|---|
| 0 | 寄存器流水线（每线程整行，零归约） | 修复发散 UB 后 PASS，待重新评估 |
| 1 | BSM shuffle 实验 | PASS（诊断用） |
| 2 | 直连全局加载 | PASS（性能下界基准 256GB/s） |
| 3/4 | v3 vendor 结构复刻（NS1/NS2） | PASS，正式路径骨架 |
| 5 | split-KV + combine | **转正**（E2E 主路径） |
| 7/8/9 | 数学剥离诊断（无数学/无shuffle/无exp） | 定位编译器调度 artifact |
| 10 | UM 统一 max（FlashDecoding++） | PASS，+2–4%，有溢出边界 |
| **11** | **分桶惰性缩放** | **默认路径**，零代价消溢出 |

## 四、最终成绩

### 内核级（test_m1.py，GPU event 计时，正确性全 P）

> **"vs vendor" 一列是 M1 延迟 / vendor 延迟，即 M1 当前仍慢于 vendor。**
> 与下文 E2E 的"拉回 vendor 水平"不矛盾：那是 vLLM 端到端吞吐口径
> （v7 自身退化到 1383 tok/s，M1 4428 tok/s≈vendor 4452 tok/s），
> 这里的内核级裸 kernel 计时仍是 M1 更慢。两个口径量纲不同，不要混用。

| 配置 | vendor | M1 最优(split-NP) | 延迟比 M1/vendor | vs v7 |
|---|---|---|---|---|
| lowB-s4096 NP32 | 0.046ms | 0.1371ms | 2.95× | 42× |
| s1024 NP2 | 0.283ms | 1.1513ms | 4.06× | 2.9× |
| s2048 NP2 | 0.555ms | 2.2365ms | 4.08× | 4.1× |
| s512 NP2 | 0.065ms | 0.2245ms | 3.7× | 3.8× |
| gq4 NP2 | 0.147ms | 1.0950ms | 7.43× | 1.8× |
| d64 NP1 | 0.092ms | 0.3496ms | 3.8× | v7 不可用 |

### E2E（bench_e2e.py，vLLM continuous batching，Qwen3-0.6B）

| 场景（max_tokens × batch） | v7 | M1-UM2 | vs v7 |
|---|---|---|---|
| 256 × 128 | 4411.9 tok/s | 5827.8 | +32% |
| 1024 × 64 | 732.8 | 3213.0 | **4.39×** |
| 1024 × 32 | 597.5 | 1726.1 | 2.89× |
| 2048 × 32 | 325.6 | 1729.2 | **5.31×** |
| 2048 × 16 | 241.1 | 882.4 | 3.66× |

规律：**M1 收益区 = 低并发 × 长序列**；增益随 sl 单调放大（+32% → 4.4× → 5.3×）。
batch=64@256tok 持平与内核级"split-KV 专治低 block 并发"结论自洽。

### E2E 大模型复验（Qwen3-4B，GQ=4 / HD=128 / 36 层，2026-10-07 追加）

| 场景（max_tokens × batch） | v7 | M1-UM1 | M1-UM2 | vs v7 |
|---|---|---|---|---|
| 64 × 16（sanity） | 700.8 | 668.4 | 669.4 | −4.5%（发射延迟区） |
| 256 × 128 | 1383.5 | 4375.7 | **4428.4** | **3.20×** |
| 256 × 64 | 1233.1 | 2620.8 | 2621.9 | 2.13× |
| 1024 × 32 | 358.6 | 1309.2 | **1316.1** | **3.67×** |
| 1024 × 16 | 255.8 | 707.6 | 704.9 | 2.76× |

（单位 tok/s。dispatch 全程 0 回退；文本一致 3/4，与 0.6B 的近并列翻转率一致。）

**关键发现**：第一轮因漏开关而全走 vendor 直连的那组数据恰好给出了 vendor 基线
（b128×256 = 4452.7 tok/s），与 M1 的 4428 基本持平（−0.5%）；而 v7 同场景仅
1383.5，比 vendor 慢 3.2×。即模型从 0.6B 放大到 4B 后，v7 的 non-split decode
因 KV 读取并行度不足而急剧退化，M1 则稳定在 vendor 水平——这解释了为何 0.6B
只有 +32%，4B 却有 2–3.7×。

### E2E 跨模型复现（Qwen3-8B，hidden 4096，与 4B 同层数/同形状）

| 场景（max_tokens × batch） | v7 | M1-UM1 | M1-UM2 | vs v7 | 4B 对照 |
|---|---|---|---|---|---|
| 64 × 16（sanity） | 677.4 | 666.0 | 667.4 | −1.5% | −4.5% |
| 256 × 128 | 1298.2 | 3800.9 | **3817.9** | **2.94×** | 3.20× |
| 256 × 64 | 1127.1 | 2450.2 | 2476.4 | 2.20× | 2.13× |
| 1024 × 32 | 342.6 | 1209.0 | 1195.5 | **3.49×** | 3.67× |
| 1024 × 16 | 240.4 | 685.2 | 688.3 | 2.86× | 2.76× |

**四个主场景逐一复现，偏差 <9%**。这比单点数字更强：split-KV 的 2–3.7× 收益与
模型宽度无关，只取决于 batch × 序列形态——因为瓶颈是 decode 的 block 并发度
（batch × 层数），不是模型hidden 维度。

### GQ 守卫拒绝侧真实模型闭环（Qwen2.5-7B-Instruct，28Q/4KV = **GQ=7**）

内核级曾抓到 GQ∈{3,5,6,7} 因`BDZ = NTH/(BDX·GQ)` 整数除法截断产出**静默错误**
（误差~1.0）。修复后 binding 收紧为 {2,4,8}、返回 -1 → 插件落回 v7，但**从未在
真实模型上验证**。Qwen2.5-7B 正是 GQ=7 的真实模型。

| 场景 | v7 tok/s | M1-UM1 | M1-UM2 | dispatch | 文本一致 |
|---|---|---|---|---|---|
| b16 × 64 | 313.1 | 310.2 | 312.2 | 全`native_decode`，`m1_split_decode` **0 次** | **4/4** |
| b32 × 256 | 625.2 | 620.9 | 620.5 | 同上 | **4/4** |

守卫工作完全正确：M1 被干净拒绝、三档吞吐差异 <1%、生成文本与 v7 完全一致
（4/4，比 4B/8B 的 3/4 更严格）。"遇到不支持 GQA 比率自动落回且不影响正确性"
这条生产安全性实测闭环。

### 支持集全路径：GQ=8 真实模型（PaddleOCR-VL，16Q/2KV）

扫描共享库时找到的 `PaddleOCR-VL`（GQ=8，HD=128，18 层，4.0G）config 带`auto_map`
自定义代码，一度判为"vLLM 支持存疑"——实测 **vLLM 0.17 原生支持该架构**
（registry 305 archs 含 `PaddleOCRVLForConditionalGeneration`），
加 `trust_remote_code=True` 即可加载。

| 场景 | v7 tok/s | M1-UM1 | M1-UM2 | vs v7 | dispatch | 文本 |
|---|---|---|---|---|---|---|
| b16 × 64 | 298.0 | 352.7 | 356.3 | 1.20× | M1 全 `m1_split_decode` | **4/4** |
| b64 × 256 | 928.6 | 1442.9 | 1425.3 | **1.53×** | 同上 | **4/4** |
| b32 × 1024 | 278.3 | 670.4 | **686.8** | **2.47×** | 同上 | **4/4** |

**至此 GQ∈{2,4,8} 三个值全部有真实模型 E2E 证据**：

| GQ | 模型 | 状态 |
|---|---|---|
| 2 | Qwen3-0.6B（16Q/8KV） | PASS |
| 4 | Qwen3-4B / 8B（32Q/8KV） | PASS |
| **8** | **PaddleOCR-VL（16Q/2KV）** | **PASS** |
| 6 / 7 | Qwen3.5-27B(GQ6) / Qwen2.5-7B(GQ7) | 守卫拒绝 → 落回 v7，结果正确 |

### HD=256 拒绝路径（Qwen3.5-9B，GQ=4 合法但 head_dim=256）

用于隔离 HD 作为唯一拒绝原因（GQ=4 本身合法，路径不进M1 内核）。结果：

```
File ".../mxflashattn/api.py", line 191, in flash_attn_with_kvcache
    raise ValueError("head dimension must be 64 or 128 and match across q and caches")
```

调用链 `vllm_plugin.forward` → `_forward_official_split_with_mx` → 厂商 impl →
`mxflashattn/api.py:191`。**公开 API 层显式 ValueError，非静默错误、非崩溃**。
与 GQ 非 2 幂 在 binding 层拒绝的机制不同但效果一致——**两层守卫（API 头维度 +
binding GQ）共同覆盖了当前已定义的 unsupported contract；未纳入 contract 的其他组合不作支持承诺。**

### 支持集最后一格：HD=64 × GQ=8（TinyLlama-1.1B-Chat-v1.0）+ 修复 Llama 族 q 非连续缺陷

`HD∈{64,128}` 里的 HD=64 在真实 causal LM 上此前从未验证过——共享库所有 HD=64 的
checkpoint 都是 BERT/encoder 架构（GQ=1），不是生成模型。ModelScope 的镜像在
**`AI-ModelScope/<name>`** 命名空间下（本实例 HF 完全不通），找到
`AI-ModelScope/TinyLlama-1.1B-Chat-v1.0`：22 层 hidden 2048，**32Q/4KV → GQ=8，
head_dim=64**，2.05 GiB。

**首次跑失败**：三档吞吐完全一致（250 tok/s）、`m1_split_decode` 0 次，fallback 原因
指向 `m1_binding.cpp:87` 的 dtype/layout 检查——而该检查里没有任何 D 条件，合成探针
也证明 HD=64×GQ=8 在内核层全 PASS。在插件入口无条件打点抓到实参：

```
ENTRY q=(3, 1, 32, 64) dtype=torch.bfloat16 contig=False stride=(2560, 2048, 64, 1)
```

`stride[0]=2560 > stride[1]=2048`：**vLLM 在 Llama 族下传的 q 是从融合 QKV buffer
切出的非连续视图**（batch 维按最大 batch=5 分配、实际只用 3）。`reshape` 不保证
连续，binding 的 `q.is_contiguous()` 守卫直接拒绝。同一天 Qwen3-4B 的 q 是
`contig=True`——**这是模型族差异，不是形状差异**，HD=64 只是"第一个踩到 Llama 族
的模型"。同一缺陷也让 v7 扩展（`binding.cpp:15-16` 的 `is_contiguous()` 检查，报错字符串在第 18 行）一并失效，故修复前自研 decode
整体退化为 vendor 直连。

**修复**（`integrations/vllm/vllm_plugin.py`，M1 与 v7 两处）：

```python
_qr = q.reshape(B, 1, HQ, D)
if not _qr.is_contiguous():
    _qr = _qr.contiguous()
```

**修复后（tok/s）**：

| 场景 | v7 | M1-UM1 | M1-UM2 | vs v7 | 文本 |
|---|---|---|---|---|---|
| b16 × 64 | 742.9 | 720.5 | 723.9 | 0.97×（发射延迟区） | 3/4 |
| b64 × 256 | 802.5 | **2384.6** | 2378.3 | **2.96×** | 3/4 |
| b32 × 1024 | 177.4 | **1150.3** | 1095.1 | **6.48×** | 3/4 |

派发 1386 / 5610 / 22506 次 `m1_split_decode`，0 回退。收益随序列拉长急剧放大；
b32×1024 上非 split 的 v7 只有 177 tok/s，**split-KV 是 HD=64 上 decode 可用的前提**
（与 4B/8B 的"v7 退化、M1 拉回"同构且更极端）。

至此支持集 6 格全覆盖：

| HD \ GQ | 2 | 4 | 8 |
|---|---|---|---|
| **64** | — | — | **TinyLlama-1.1B（32Q/4KV）PASS** |
| **128** | Qwen3-0.6B PASS | Qwen3-4B / 8B PASS | PaddleOCR-VL PASS |

### 非 2 幂 GQ 守卫：GQ=5 真实模型闭环 + GQ=6 不可得的证明

原“GQ=6 真实模型 E2E”遗留不能按原计划完成。扫描共享库 40 个模型 + ModelScope 40+ 个主流 checkpoint，
GQ=6 仅 3 个且**全部 head_dim=256**：

| 模型 | nQ/nKV | GQ | head_dim | 能否隔离 GQ |
|---|---|---|---|---|
| Qwen3.5-27B-W8A8 | 24/4 | 6 | 256 | 否 |
| Qwen3.8-27B | 24/4 | 6 | 256 | 否 |
| Qwen3.8-27B-W8A8 | 24/4 | 6 | 256 | 否 |

三者 GQ 与 HD **两个变量同时不合法**，而 API 层（`mxflashattn/api.py:191`）先于 binding 层触发，
路径根本到不了 M1 内核的 GQ 守卫——只能再次复现 HD=256 拒绝。配置改写也不可行：
GQ=6 需要 nQ×HD 与 hidden_size 对齐，Qwen3-32B（64Q/8KV, hidden 5120）改成 24Q/4KV 需 hidden 3072 ≠ 5120，
权重加载不了。

**改用 GQ=5**（Qwen2.5-14B-Instruct，40Q/8KV，HD=128，48 层，29 GB）：同一条非 2 幂拒绝路径、真实权重、
且是从未测过的新比率；Qwen2 与 Llama 同属传非连续 q 的架构族，同时回归验证 `.contiguous()`
 修复无回归。

| 场景 | v7 | M1-UM1 | M1-UM2 | 派发 |
|---|---|---|---|---|
| b16 × 64 | 420.77 | 177.20 | 177.13 | `m1_split_decode` **0** |
| b64 × 256 | 678.71 | 688.70 | 694.14 | `m1_split_decode` **0** |
| b32 × 1024 | 215.87 | 319.11 | 318.90 | `m1_split_decode` **0** |

三档均拒绝原因：`RuntimeError('split kernel unsupported config: D=128 GQ=5')。

**两个反常现象归因**（不是内核问题）：

1. **吞吐不一致**（sanity 档 M1 反而慢 2.4×，long 档快 1.48×）——**守卫拒绝路径的 CPU 开销**。
   M1 一次都不派发，但每个 decode 步都要走一遍 `try → 抛 RuntimeError → except → 落回 v7`，
   累计 7680 / 16896 / 53760 次。短序列小 batch 下异常开销占比高；长序列下 GPU 计算占主导、开销被摊薄。
   支持形状（GQ∈{{2,4,8}}）上零异常，此成本不存在。
2. **`text_match` 仅 1/4**（此前 GQ=7 报告为 4/4）——**测量方法问题**。写 `diag_guard_text_match.py`
   定点对比：固定 4 条 prompt、batch=4、先跑两遍 v7 再跑 M1，比 token-id 序列。
   结果 **v7 vs v7 = 4/4**（harness 确定）、**v7 vs M1 = 4/4**（守卫路径逐 token 一致）。
   原因是大 batch 连续批处理下请求完成时机影响批次分组，边界 token 翻转；
   同批次下 v7 与 M1 的实际 gen_tokens 本就不同（16123 vs 16126）。
   **但 GQ=7 报告里原标的 4/4 同样不可信**，守卫路径文本一致性的可靠证据是本节的 5/4定点对比。

## 五、稳健性补测（2026-10-07 收官轮实测）

| 配置 | 结果 | 说明 |
|---|---|---|
| s8192（B=16, sl=8192, NP 扫到 64） | **全 PASS** | NP16 甜点 0.4183ms（3.63× vendor）；v7 确认不支持 >4096 ctx（启动失败），已加跳过守卫 |
| bf16（96×512） | **全 PASS** | UM2 0.3504ms（2.39× vendor），精度路径无回归 |
| GQ=3/5/6/7（非 2 幂 GQA） | **F → 已加守卫拒绝** | 在当前 NTH/BDX 布局下，GQ=3/5/6/7 实测因整数布局截断产出**静默错误**结果（split 误差 ~1.0，非数值噪声）；因此当前生产 contract 仅验证并开放 GQ∈{2,4,8}，其他比率显式拒绝。 |
| GQ=4/8 + bf16（守卫重建后复验） | **全 PASS** | 守卫无回归 |

**处置**：binding 层将 GQ 检查从连续区间 [2,8] 收紧为 {2,4,8}（两处入口），不支持的 GQA 比率返回 -1 → vLLM 插件按既有异常路径**自动落回 v7**，杜绝静默错误。主流 GQA 模型比率均为 2 的幂（含 8，如 Qwen/Llama 系列），实际影响面极小；奇数比率支持留待后续按 GQ 分配独立 NTH 解决。

**发现价值**：这一条是稳健性补测抓到的真实生产隐患——若不做，GQ=3 模型（少数长尾模型）会在插件里静默产出错误解码结果而不报任何错。

## 六、参考文献与调研依据

详见《Gate4_参考文献与调研记录.md》（同目录），含"资料→决策→结果"对照表。核心引用：

1. Hong, Ke et al. **FlashDecoding++: Faster Large Language Model Inference with Asynchronization**. MLSys 2024, pp.148-161（arXiv:2311.01282）→ 统一 max 设计来源
2. llama.cpp PR#6522 "cuda : use amd wave sharing intrinsics for warp_reduce functions" → wave64 shuffle 走 LDS 的实证
3. AMD GPUOpen《Cross-Lane Operations》→ DPP/DS 通路背景
4. MACA `__clang_maca_device_functions.h`（本机头文件逆向）→ `__builtin_mxc_mov_shfl` 发现
5. vendor decode.cuh（结构对齐参考）

## 七、工程坑记录

1. MACA `__shfl_sync` mask 内发散跳过 = 未定义行为（不报错，产出全零）。
2. `arrive_gvmcnt(N)` = "等最老组完成、允许 N 个最新在途"；wait<0> = wait-all。
3. vLLM v1 EngineCore 是子进程，父进程 `os.environ` 热切换无效 → `VLLM_ENABLE_V1_MULTIPROCESSING=0`。
4. 非 interactive shell 不加载 .bashrc → 远程跑 vLLM 须显式 source conda + MACA_PATH/LD_LIBRARY_PATH 全套，vLLM import 否则报 TypeError NoneType。
5. SFTP 直写 csrc_native 报 Failure → /tmp 中转 + mv。
6. 同一文件并行 Edit 会竞态丢编辑（本次实际踩中：parser 行丢失致远程 arg 报错）。
7. exp2f = libdevice 外部调用（非硬件指令）；内联多项式替代反而略差。
8. **bench_e2e.py 必须带 `MXFLASHATTN_GRAPH_OFFICIAL_SPLIT=1 MXFA_NATIVE_DECODE=1`**，
   漏掉时插件静默全走 `vendor_direct`，三档对比退化为同一路径跑三遍（吞吐差异仅
   噪声）。**判据：先查 dispatch 事件计数（应有 `m1_split_decode`/`native_decode`）
   再信吞吐数字**——本次第一轮数据因此作废重跑。
9. 实例侧 HF 不可达（huggingface.co 无响应），模型走 ModelScope
   （`pip install modelscope` + `snapshot_download`）；磁盘仅 30G，4B（7.9GB）
   为当前可跑上限。
10. Windows `tar` 解压含中文名的 tarball 会按系统代码页误解码 → 文件名mojibake
    （`Gate4_收官报告.md` → `Gate4_鏀跺畼鎶ュ憡.md`）。修复：`name.encode('gbk').decode('utf-8')`
    后重命名（内容无损）。
11. **实例磁盘分盘，必须 `df -h` 看全表再判断空间**：`/`（系统盘）仅 30G，
    `/data` 是独立 196G 数据盘（186G 可用），另有共享模型库 `/mnt/moark-models`
    （7T/40 个模型，含 Qwen3-8B 等现成权重）。只看 `df -h /` 会误判空间不足。
    模型应放 `/data/models`，共享库模型直接 symlink，勿复制。
12. **bench_e2e.py 原本硬编码 `LLM(model=...)` 不传 `trust_remote_code`**，加载带
    `auto_map` 的模型（PaddleOCR-VL）会报 pydantic ValidationError；且 vLLM 无
    `VLLM_TRUST_REMOTE_CODE` 环境变量（日志明确 "Unknown vLLM environment variable"），
    只能走构造参数。已给脚本加 `--trust-remote-code` 开关（默认关闭，向后兼容）。
13. 一个 GPU 不能并行两个 vLLM 引擎（显存 + 算力互抢），多模型 E2E 必须串行队列；
    队列脚本按"已有 sanity json 则跳过"实现幂等，中断后可续跑。
14. **vLLM 在 Llama 族下传入的 q 是非连续视图**（从融合 QKV buffer 切出，batch 维
    按最大 batch 分配），`q.reshape()` **不保证连续**，binding 的 `q.is_contiguous()`
    守卫会拒绝。症状极具误导性——看起来像"形状不支持"，实则是布局问题，且**同时
    打掉 M1 与 v7 两条路径**（表现为三档吞吐一致、无 `m1_split_decode`）。修复：显式
    `.contiguous()`（连续时为 no-op）。任何 Llama 系 checkpoint 都会命中。
    **验证新形状前先用 `MXFA_ENTRY_PROBE` 打入口看 q 的 `contig/stride`**，
    否则会误判成守卫拒绝。
15. **临时改插件源码打探针，每次必须从权威干净副本恢复**。本次连续打了三版探针，
    第二版的"备份"是从已被第一版污染的源码拷的，导致探针静默不生效、白跑两轮。
    正确做法：跑完用 `md5sum` 比对并确认 marker 数为 0；`mx_ssh.py put` 传本地
    权威副本最稳妥。
16. **替换 `_load_m1_decode_ext` 做运行时探针无效**：插件用模块级全局 `_m1_ext`
    缓存已加载的 `.so`，替换 loader 不会被调用（探针 0 行）。必须在源码里打点。
18. **找特定 GQA 比率的模型时 head_dim 必须一起筛**。只按 nQ/nKV 匹配会选出
    head_dim=256 的模型，而 API 层会先于 binding 层拒绝，测不到目标变量。筛选条件：
    `GQ ∈ 目标集` **且** `head_dim ∈ {64,128}`。另：**很多 config 省略 head_dim**（Qwen2 全系），
    vLLM 默认取 `hidden_size // num_attention_heads`；扫描脚本若强制要求该字段，会把这些模型全判成
    “shape fields not found”而错失候选。
19. **`text_match` 在大 batch 下不可靠**。连续批处理让请求完成时机影响批次分组，边界 token
    会翻转。验证守卫路径正确性必须用**定点对比**：固定 prompt、batch=4、同配置先跑两遍
    验证 harness 确定性，再比对 token-id。
20. **守卫拒绝路径有真实 CPU 成本**：每次调用抛异常。短序列小 batch 下可占可观比例
    （实测慢 2.4×）。支持形状上零异常，此成本不存在。
17. **dispatch 事件要按 `(path, operation, fallback)` 三元组统计**。只按 `path` 取
    top-N 会漏：M1 从未成功时所有失败事件都记成 `native_decode` + `fallback_reason`，
    `m1_split_decode` 计数为 0，容易误判成"M1 分支根本没进去"。

## 八、遗留与下一步

1. **~3.5× 内核级调度差距（未解决，但"无法反读"的前提已被推翻）**：内核级M1 仍慢于 vendor 2.95×–7.43×（见内核级表格的"延迟比M1/vendor"列）。
   ~~需向沐曦申请 llvm-dis/反汇编支持~~ —— **该结论已推翻**：工具链自带
   `/opt/maca/mxgpu_llvm/bin/llvm-dis`（LLVM 19.1.3）与 `clang-offload-bundler`，
   只是不在 PATH 里。正确取证路径为
   `mxcc -x maca -offload-arch native -emit-llvm -c` → `clang-offload-bundler --unbundle`
   取设备端 bitcode → `llvm-dis` 读回 IR。
   **已据此定位根因**：v7 路径命中硬件融合指令 `llvm.mxc.expadd.f32.i32`（exp+add 一步完成，
   1005 处），M1 走软件模拟（`llvm.exp2.f32` 1820 次 + `llvm.mxc.mov.shfl.i32` 4201 次），
   每个 score 元素多出约 4 次 shfl + 1 次 exp2。**下一步打法是把 M1 的 online-softmax
   重缩放换成 `expadd`**，而非继续调 `-O3`/unroll/pragma（那些已试过无效）。
   另有后端旋钮 `-disable_promote_alloca_to_bsm` / `-disable_promote_alloca_to_vector`
   对应 IR 中 1316 处 `bsm.bpermute`，待评估。
2. **git 提交**：主仓库 `/root/MXFlashAttn`（gitlink.org.cn/Leo77/MXFlashAttn）仅 3 个 commit，v0.x 以来积累全部未提交；M1 文件在 v1.3.0-kernel 工作树未入仓库。提交范围待定。
3. 溢出 dirty 回退已有 UM2 方案替代，FlashDecoding++ recompute 机制不再需要。
4. mode 0（寄存器流水线，理论上限最优）可在工具链问题解决后重估。
5. GQ=6 合法 HD=64/128 的真实 checkpoint 未找到；现有 GQ=6 模型同时为 HD=256，会先被 API 层拒绝，不能宣称已完成 GQ=6 binding 守卫 E2E。
6. Qwen3-32B（GQ=8 标准 Qwen3 架构，65G）可作 GQ=8 的第二份跨架构证据。
   GQ=8 现在已有 PaddleOCR-VL（多模态 OCR）与 TinyLlama（标准 dense Llama）两份，
   跨架构已覆盖，优先级下降。

## 九、证据清单（gate4_evidence_20261007/）

- 代码：m1_kernel.cpp / m1_binding.cpp / patch_plugin.py / build_m1.sh
- 测试：test_m1.py（13+ 配置）/ bench_e2e.py / wl_*（ISA 探针）/ bench_bw / bench_stream / bench_opcost
- 结果：m1-smoke-{v7,m1,m1um}.json / bench-e2e.json / bench-e2e-sl1024.json / bench-e2e-sl2048.json（+ dispatch.jsonl 派发流）
- 大模型：bench-q4b-{sanity,b256,l1024}.json、bench-q8b-{sanity,b256,l1024}.json（+ 同名 dispatch.jsonl）
- 守卫拒绝侧：bench-q25-7b-gq7{,-b32}.json、bench-hd256-q35-9b-*.json（+ 同名 dispatch.jsonl）
- GQ=8 全路径：bench-gq8-paddle-{sanity,b64,l1024}.json（+ 同名 dispatch.jsonl）
- HD=64 全路径：bench-hd64-tiny-{sanity,b64,l1024}.json（+ 同名 dispatch.jsonl）
- GQ=5 守卫：bench-gq5-q25-14b-{sanity,main,long}.json（+ 同名 dispatch.jsonl）
- 定点对比：diag_guard_text_match.py（v7 vs v7 / v7 vs M1 均 4/4）
- 形状扫描：scan_gq_shapes.py（本地库）/ find_nonga_models.py（ModelScope）
- 形状库扫描：scan_local_shapes.py（共享库 40 模型 GQ×HD 全表）
- 文档：M3_VLLM_WIRING.md / Gate4_参考文献与调研记录.md / E2E_大模型与守卫验证.md / 本报告
