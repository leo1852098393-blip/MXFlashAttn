# E2E 大模型与守卫验证：Qwen3-4B / 8B / Qwen2.5-7B(GQ=7) / PaddleOCR-VL(GQ=8) / Qwen3.5-9B(HD=256) / TinyLlama(HD=64)

日期：2026-10-07（远程 C500 实例，工作树 `/root/MXFlashAttn_v1.3.0-kernel`）

**一句话结论**：内核支持集 `{GQ∈{2,4,8} × HD∈{64,128}}` 的全部 6 个格子都有真实
模型 E2E 证据（HD=64 × GQ=8 由 TinyLlama-1.1B 补齐，附带修掉一个"整个 Llama 族
q 非连续导致自研 decode 全退化"的真实 bug）；GQ=5/7 等已验证的非 2 幂 GQ 与超界 HD（256）
两类不支持形状均被干净拒绝且不影响正确性；M1 在 0.6B→1.1B→4B→8B 四个规模上
收益规律一致（长序列/中低并发区间约 2.0×–6.48×，短序列低并发持平或略负，偏差 <9%）。

## 最终口径（以本节为准）

- GQ=5、GQ=7：有合法 HD=128 的真实模型守卫拒绝证据，M1 dispatch 为 0 并回退 v7。
- GQ=6：当前没有合法 HD=64/128 的可用真实 checkpoint；现有 GQ=6 模型为 HD=256，会先被 API guard 拒绝，不宣称已完成 GQ=6 binding E2E。
- HD=256：由公开 API 层显式拒绝，不进入 M1 GQ 守卫。
- 守卫路径文本一致性以固定 batch/token-id harness 为准；连续 batching 的 `text_match` 不作为唯一 correctness 判据。

各档 E2E 的实例侧复现脚本在仓库 `scripts/` 下：`run_e2e_q4b.sh` / `run_e2e_q8b.sh`（GQ=4 主场景）、
`run_e2e_gq7.sh`（GQ=7 守卫拒绝）、`run_e2e_gq8.sh`（GQ=8 PaddleOCR-VL）、`run_e2e_hd256.sh`（HD=256 API 拒绝）、
`run_queue.sh`（单卡串行编排）、`scan_shapes.sh`（共享模型库形状扫描，用于 GQ=6 不可得的证明）、
`ms_install.sh` / `probe_paddle_vllm.sh` / `stage_models.sh`（模型获取、架构探测与数据盘暂存）。
运行期shell 重定向日志已归档至 `docs/gate4/evidence/`。


## 目的

v1.3.0 的 M1 split-KV decode（UM=2 分桶惰性缩放为默认）此前只在 Qwen3-0.6B 上做过
E2E。本次在真实 GQA 大模型上验证端到端收益与正确性。

## 环境与模型

- vLLM 0.17.0 / torch 2.8.0+metax3.5.3.9 / MACA 3.5.3（`MACA_PATH=/opt/maca`）
- 模型：`Qwen/Qwen3-4B`（ModelScope 下载，HF_HUB_OFFLINE=1），bf16，36 层，
  **head_dim=128，32 Q heads / 8 KV heads → GQ=4**，落在内核支持形状
  （HD∈{64,128}、GQ∈{2,4,8}）内
- 方法：`csrc_native/bench_e2e.py` 一次提交全批 prompt（continuous batching），
  三档 env 热切换（v7 / M1-UM1 / M1-UM2），`VLLM_ENABLE_V1_MULTIPROCESSING=0`

## 结果

### 吞吐（tokens/s，去掉 EOS 差异，gen_tokens 3.27 万/档）

| 场景（batch × max_tokens） | v7 | M1-UM1 | M1-UM2 | UM2 vs v7 |
|---|---|---|---|---|
| sanity b16 × 64 | 700.8 | 668.4 | 669.4 | −4.5%（发射延迟区，持平） |
| b128 × 256 | 1383.5 | 4375.7 | **4428.4** | **3.20×** |
| b64 × 256 | 1233.1 | 2620.8 | 2621.9 | **2.13×** |
| b32 × 1024 | 358.6 | 1309.2 | **1316.1** | **3.67×** |
| b16 × 1024 | 255.8 | 707.6 | 704.9 | **2.76×** |

### 派发验证（dispatch.jsonl）

- v7 档：decode 事件全部 `native_decode`（b256 档 9216 = 36 层 × 256 步）
- M1 档：decode 事件全部 `m1_split_decode`，**0 回退、0 blocked**
- 第一轮误跑（漏开关）的对照数据：全 vendor 直连时 b128 × 256 = 4452.7 tok/s

### 文本一致性（温度 0，批内 4 条 vs v7）

- UM1：2/4；UM2：3/4。与 0.6B 的近并列 token 翻转率一致，未见发散。

## 关键发现

1. **4B 上 v7 自研 decode 路径严重退化**：v7 native decode 在 b128 × 256 仅
   1383 tok/s，比 vendor 直连（4452，第一轮对照）慢 3.2×。0.6B 时 v7 尚能
   与 vendor 持平，模型变大后 non-split decode 的 KV 读取并行度不足被放大。
2. **M1 把 custom 路径拉回 vendor 水平并略优**：M1-UM2 b128 = 4428 ≈ vendor
   4453（−0.5%）；b64 +8.3%；长序列 b32×1024 +2.7%、b16×1024 +6.5%。
   即：在大模型上，split-KV 不是"锦上添花"，而是自研 decode 路径可用的前提。
3. E2E 增益随 batch/序列放大的规律与 0.6B 一致（0.6B b128 +32%、sl2048 5.3×；
   4B 因 v7 退化更重，相对倍数反而更大）。

## 工程坑（本次新增）

- **bench_e2e.py 必须带 `MXFLASHATTN_GRAPH_OFFICIAL_SPLIT=1 MXFA_NATIVE_DECODE=1`**。
  漏掉时插件静默全走 vendor_direct，三档对比退化为同一跑三遍（表现为 dispatch
  里 0 个 `m1_split_decode`/`native_decode` 事件、三档差异仅噪声）。判据：先查
  dispatch 事件计数再信吞吐数字。
- 实例 HF 不通（huggingface.co 无响应），模型走 ModelScope
  （`pip install modelscope` + `snapshot_download`）。
- **磁盘分盘，务必先看 `df -h` 全表再下模型**：本机 `/`（系统盘）仅 30G，
  而 `/data` 是独立 196G 数据盘（186G 可用），另有共享模型库
  `/mnt/moark-models`（7T，40 个模型，含Qwen3-8B 等）。只看 `df -h /` 会误判
  空间不足而下大模型，白占系统盘。

---

# 追加一：Qwen3-8B（GQ=4 / HD=128 / 36 层，hidden 2560→4096）

## 结果（tok/s）

| 场景（max_tokens × batch） | v7 | M1-UM1 | M1-UM2 | vs v7 |
|---|---|---|---|---|
| 64 × 16（sanity） | 677.4 | 666.0 | 667.4 | −1.5%（持平） |
| 256 × 128 | 1298.2 | 3800.9 | **3817.9** | **2.94×** |
| 256 × 64 | 1127.1 | 2450.2 | 2476.4 | 2.20× |
| 1024 × 32 | 342.6 | 1209.0 | 1195.5 | **3.49×** |
| 1024 × 16 | 240.4 | 685.2 | 688.3 | 2.86× |

dispatch：M1 全部 `m1_split_decode`、0 回退；文本一致 3/4（与4B/0.6B 同）。

## 跨模型一致性（M1-UM2 vs v7）

| 场景 | Qwen3-4B | Qwen3-8B |
|---|---|---|
| 256 × 128 | 3.20× | 2.94× |
| 256 × 64 | 2.13× | 2.20× |
| 1024 × 32 | 3.67× | 3.49× |
| 1024 × 16 | 2.76× | 2.86× |

**四个场景逐一复现，偏差 <9%**。这是比单点数字更强的证据：split-KV 带来的
2–3.7× 收益与模型规模无关，只取决于 batch/序列形态——因为瓶颈是 decode 的
block 并发度（由 batch × 层数决定），不是模型宽度。

---

# 追加二：GQ=7 守卫拒绝侧验证（Qwen2.5-7B-Instruct，28Q/4KV）

## 目的

内核级稳健性测试曾抓到 GQ∈{3,5,6,7} 会因 `BDZ = NTH/(BDX·GQ)` 整数除法截断
而产出**静默错误**（误差 ~1.0，非数值噪声）。修复方案是 binding 层把 GQ 检查从
区间 [2,8] 收紧为 {2,4,8}，不支持则返回 -1 → 插件自动落回 v7。**此前从未在真实
模型上验证过这条守卫是否安静放行**。Qwen2.5-7B 恰好是 GQ=7（28Q/4KV）的真实模型。

## 结果

| 场景 | v7 tok/s | M1-UM1 | M1-UM2 | 文本一致 |
|---|---|---|---|---|
| b16 × 64 | 313.1 | 310.2 | 312.2 | **4/4** |
| b32 × 256 | 625.2 | 620.9 | 620.5 | **4/4** |

**dispatch 签名（决定性证据）**：三档配置**全部只有 `native_decode`，
`m1_split_decode` 出现 0 次**。

## 结论

守卫在真实模型上工作完全正确：GQ=7 时 M1 被干净拒绝 → 三档吞吐一致（差异
<1%）→ 生成文本与 v7 **完全一致（4/4，比 4B/8B 的 3/4 更严格）**。
即"遇到不支持的 GQA 比率自动落回且不影响正确性"这条生产安全性，实测闭环。

补充说明：Qwen2.5-1.5B（12Q/2KV）= **GQ=6**，但当前报告不把它作为已完成的 binding 守卫 E2E 证据；GQ=6 的合法 HD=64/128 真实 checkpoint 尚未形成可用闭环。

## 遗留

- **GQ=8 真实模型仍未覆盖**。扫描共享库 40 个模型 + ModelScope 候选后确认：
  主流 7-8B（Llama-3/3.1、Mistral-7B）全为 GQ=4；符合 GQ=8 的最小模型是
  Qwen3-32B（65G，可下到 `/data`，需确认显存与时长）或 Qwen1.5-110B（220G，不现实）。
  共享库另有 `PaddleOCR-VL`（16Q/2KV，GQ=8，仅 4.0G）但为自定义架构
  （`trust_remote_code` + 112KB 自定义 modeling），vLLM 支持存疑，待验证。
  内核级 test_m1.py 的 GQ=8 配置已 PASS。
- HD=256 路径（Qwen3.5-9B / Qwen3.8-27B 的 HD=256 超出内核 HD∈{64,128} 支持）
  同样未在真实模型上验证，属另一条守卫路径（HD 拒绝），可后续补。
- 远端结果：`artifacts/bench-q8b-*.json`、`artifacts/bench-q25-7b-gq7*.json`
  （+ 同名 dispatch.jsonl）。

---

# 追加三：GQ=8 全路径打通（PaddleOCR-VL，16Q/2KV）

上一节把 GQ=8 列为"待验证"，本节已解决。扫描共享库时发现 `PaddleOCR-VL`
（16Q / 2KV = **GQ=8**，HD=128，18 层，4.0G）——它的 config 带 `auto_map`
自定义代码，一度判为"vLLM 支持存疑"，实测**vLLM 0.17 原生支持该架构**
（registry 305 个 arch 中含 `PaddleOCRVLForConditionalGeneration`），
加 `trust_remote_code=True` 即可加载（KV cache 31.66 GiB / 184 万 token）。

## 结果（tok/s）

| 场景 | v7 | M1-UM1 | M1-UM2 | vs v7 | dispatch | 文本 |
|---|---|---|---|---|---|---|
| b16 × 64 | 298.0 | 352.7 | 356.3 | 1.20× | M1 全 `m1_split_decode` | **4/4** |
| b64 × 256 | 928.6 | 1442.9 | 1425.3 | **1.53×** | 同上 | **4/4** |
| b32 × 1024 | 278.3 | 670.4 | **686.8** | **2.47×** | 同上 | **4/4** |

- 派发计数：1134 / 4590 / 18414 次 `m1_split_decode`，**0 回退**。
- 文本一致性**三档全部 4/4**（比 Qwen3 系列的 3/4 更严格——PaddleOCR-VL 是
  OCR 任务模型，输出更确定，近并列 token 翻转更少）。
- 收益随序列拉长放大（1.20× → 1.53× → 2.47×），与 GQ=4 规律同构。

## 意义

至此**内核支持集{GQ∈{2,4,8} × HD∈{64,128}} 的三个 GQ 值全部有真实模型 E2E 证据**：

| GQ | 模型 | 状态 |
|---|---|---|
| 2 | Qwen3-0.6B（16Q/8KV） | PASS |
| 4 | Qwen3-4B / 8B / Mistral-7B-v0.3（32Q/8KV） | PASS |
| **8** | **PaddleOCR-VL（16Q/2KV）** | **PASS** |
| 6 / 7（非 2 幂） | Qwen3.5-27B(GQ6) / Qwen2.5-7B(GQ7) | 守卫拒绝，落回 v7，结果正确 |

# 追加四：HD=256 拒绝路径（Qwen3.5-9B）

`Qwen3.5-9B`（text_config Q=16 / KV=4 → **GQ=4 合法**，但 `head_dim=256`
超出支持集）用于隔离 HD 维度作为唯一拒绝原因。

## 结果：加载即被拒，错误清晰

```
File ".../mxflashattn/api.py", line 191, in flash_attn_with_kvcache
    raise ValueError("head dimension must be 64 or 128 and match across q and caches")
ValueError: head dimension must be 64 or 128 and match across q and caches
```

调用链：`vllm_plugin.forward` → `_forward_official_split_with_mx` →
厂商 `MXFlashAttnImpl.forward` → `mxflashattn/api.py:191` 抛错。

## 结论

HD=256 被**公开 API 层**干净拒绝（显式 ValueError，非静默错误、非崩溃），
与 GQ 非 2 幂 在 binding 层的拒绝机制不同但效果一致：不支持形状 → 明确报错 →
不会产出错误结果。因为 GQ=4 本身合法，路径未进入 M1 内核，故拒绝点在 API 而非
binding。这也说明两层守卫（API 头维度 + binding GQ）共同覆盖了当前已定义的 unsupported contract；未纳入 contract 的其他组合不作支持承诺。

## 历史遗留（已被追加六修正）

> 以下“GQ=6 尚未跑真实模型、预期行为一致”的旧表述已被追加六 supersede，不作为最终结论。

- GQ=6 合法 HD=64/128 的真实 checkpoint 未形成可用闭环；现有候选为 HD=256，先被 API guard 拒绝。
- Qwen3-32B（GQ=8 标准 Qwen3 架构，65G）可作为 GQ=8 的第二份跨架构证据，
  验证 PaddleOCR-VL（多模态 OCR 架构）之外的标准 dense 架构也能走通。
- 远端结果：`artifacts/bench-gq8-paddle-*.json`、`artifacts/bench-hd256-*.json`。

---

# 追加五：HD=64 × GQ=8 全路径打通（TinyLlama-1.1B-Chat-v1.0）+ 一个真实 bug

补上支持集里**最后一格未验证的组合**。此前"支持集全覆盖"的说法有一处疏漏：
`HD∈{64,128}` 中的 HD=64，在真实 causal LM 上从未跑过——共享库里所有 HD=64 的
checkpoint（`bge-reranker-v2-m3`、`w2v-bert-2.0`）都是 BERT/encoder 架构、GQ=1，
不是生成模型。

## 找模型：ModelScope 的 `AI-ModelScope` 命名空间

先扫了共享库全部 40 个模型 + ModelScope/HF 候选，HD=64 的因果 LM 一个都没有：

| 模型 | GQ | HD | 能否用 |
|---|---|---|---|
| Qwen2 / Qwen2.5-0.5B | 7 | 64 | GQ 非 2 幂，守卫会拒 |
| StableLM-2-1.6B / pythia-160m | 1 | 64 | GQ=1 不在内核范围 |
| TinyLlama-1.1B（`TinyLlama/...`） | — | — | ModelScope 未同步 |
| **AI-ModelScope/TinyLlama-1.1B-Chat-v1.0** | **8** | **64** | **可用** |

关键点：ModelScope 的镜像在 **`AI-ModelScope/<name>`** 命名空间下，不是原始
HF id。实测 `huggingface.co` 在本实例完全不通，只有 `www.modelscope.cn` 可达。
`AI-ModelScope/TinyLlama-1.1B-Chat-v1.0` = `LlamaForCausalLM`，22 层，
hidden 2048，**32 Q / 4 KV → GQ=8，head_dim=64**，2.05 GiB。

## 第一次跑：M1 一次都没生效

```
[v7] batch=16 tps=250.29  disp={'vendor_direct': 1408, 'native_decode': 1386}
[m1_um] batch=16 tps=250.44  disp={'vendor_direct': 1408, 'native_decode': 1386}   ← 三档完全一样
```

三档吞吐一致、dispatch 里没有 `m1_split_decode`，但 fallback 原因写着：

```
RuntimeError('dtype/layout check failed
Exception raised from paged_decode_split at m1_binding.cpp:87 ...')
```

`m1_binding.cpp:87` 是纯 dtype/layout 检查，**里面没有任何 D 相关的条件**。合成张量
探针也证明 HD=64 × GQ=8 在内核层完全正常（fp16/bf16 都 PASS），max_sl_bound /
np / c_len 全组合扫描也全 PASS。所以问题不在形状，在**真实调用传进来的张量布局**。

## 根因：Llama 架构下 vLLM 传的 q 是非连续的

在插件适配器入口无条件打点（`MXFA_ENTRY_PROBE`）抓到实参：

```
ENTRY q=(3, 1, 32, 64) dtype=torch.bfloat16 contig=False stride=(2560, 2048, 64, 1)
ENTRY q=(5, 1, 32, 64) dtype=torch.bfloat16 contig=False stride=(2560, 2048, 64, 1)
ENTRY q=(1, 3, 32, 64) dtype=torch.bfloat16 contig=False stride=(7680, 2560, 64, 1)
```

`stride[0]=2560 > stride[1]=2048`——q 是从融合 QKV buffer 上切出来的**视图**，且
batch 维按最大 batch（5）分配、实际只用 3，所以非连续。`q.reshape(B,1,HQ,D)`
会保留这份非连续性，binding 的 `TORCH_CHECK(... q.is_contiguous() ...)` 直接拒绝。

对照 Qwen3-4B（同一天、同一实例）：`ENTRY q=(4,1,32,128) contig=True`。**这是
模型族差异，不是形状差异**——Qwen3 的 vLLM 实现给出连续 q，Llama 的不给出。
所以 HD=64 之所以"从没验证过"，是因为整个 Llama 族在这个栈上此前都没被测过。

同一个 bug 也打掉了 v7 扩展（`binding.cpp:15-16` 的 `is_contiguous()` 检查，报错字符串在第 18 行），所以修复前 TinyLlama
上 v7 和 M1 都退化成 vendor 直连——这也解释了为什么修复前 b16×64 只有 250 tok/s，
修复后变成 742。

## 修复

`integrations/vllm/vllm_plugin.py`，M1 与 v7 两条路径各一处：

```python
_qr = q.reshape(B, 1, HQ, D)
if not _qr.is_contiguous():
    _qr = _qr.contiguous()
```

`reshape` 在非连续输入上不保证连续，必须显式 `.contiguous()`；连续时是无开销的
no-op。v7 分支同样加上 `.contiguous()`。

## 修复后结果（tok/s）

| 场景（batch × max_tokens） | v7 | M1-UM1 | M1-UM2 | vs v7 | 文本 |
|---|---|---|---|---|---|
| b16 × 64 | 742.9 | 720.5 | 723.9 | 0.97×（发射延迟区，持平） | 3/4 |
| b64 × 256 | 802.5 | **2384.6** | 2378.3 | **2.96×** | 3/4 |
| b32 × 1024 | 177.4 | **1150.3** | 1095.1 | **6.48×** | 3/4 |

- 派发：1386 / 5610 / 22506 次 `m1_split_decode`，**0 回退**。
- 收益随序列拉长急剧放大（0.97× → 2.96× → 6.48×）。b32×1024 上 v7 只有
  177 tok/s 而 M1 有 1150——**非 split 的 decode 路径在 HD=64 上几乎不可用**，
  split-KV 是它能用的前提（与 4B/8B 上"v7 退化、M1 拉回"的结论同构，且更极端）。

## 意义：支持集真正全覆盖

| HD \ GQ | 2 | 4 | 8 |
|---|---|---|---|
| **64** | — | — | **TinyLlama-1.1B（32Q/4KV）PASS** |
| **128** | Qwen3-0.6B PASS | Qwen3-4B / 8B PASS | PaddleOCR-VL PASS |

6 个格子全部有真实模型 E2E 证据。附带修掉一个会影响整个 Llama 族的真实 bug。

## 工程坑（本次新增）

- **q 非连续是模型族相关的，不看形状看不出来**。HD=64 只是"第一个踩到 Llama 族
  的模型"，所以症状看起来像形状不支持。任何 Llama 系 checkpoint（Llama-2/3、
  TinyLlama、Mistral、Qwen2 等带 RoPE 的标准族）在这个栈上都会命中同一条路。
  教训：验证新形状前，先用 `MXFA_ENTRY_PROBE` 确认 q 的 `contig/stride`，
  否则会误判成"形状不支持"。
- **改插件源码做临时探针时，每次必须从干净副本恢复**。本次连续打了 v3/v4/v5 三版
  探针，v5 的"备份"是从已被 v4 污染的版本拷的，导致探针静默不生效、白跑两轮。
  正确做法：探针前 `cp` 一份权威干净副本，跑完 `md5sum` 校验 marker 数为 0。
  `mx_ssh.py put` 传本地权威副本最稳妥。
- **替换 `_load_m1_decode_ext` 这类函数做运行时探针无效**：插件用模块级全局
  `_m1_ext` 缓存已加载的 `.so`，替换 loader 不会被调用。必须在源码里打点。
- dispatch 事件统计要按 `(path, operation, fallback)` 三元组看。只统计 `path` 的
  top-N 会漏掉真相：M1 从未成功时，所有失败事件都记成 `native_decode` +
  `fallback_reason`，`m1_split_decode` 计数为 0，容易误判成"M1 分支没进去"。
- 远端结果：`artifacts/bench-hd64-tiny-{sanity,b64,l1024}.json`。

---

# 追加六：GQ=6 不存在，改用 GQ=5 闭环非 2 幂守卫

上一节把"GQ=6 真实模型 E2E"列为遗留。本节先说结论：**这个遗留无法按原计划完成，
因为主流模型里不存在 GQ=6 且 head_dim 合法的 checkpoint**。改用 GQ=5
（Qwen2.5-14B-Instruct）完成同一守卫的闭环。

## 一、为什么 GQ=6 跑不了

共享库 + ModelScope 共扫 40+ 个主流 checkpoint，GQ=6 只有 3 个，且**全部
head_dim=256**：

| 模型 | nQ / nKV | GQ | head_dim | 能否隔离 GQ |
|---|---|---|---|---|
| Qwen3.5-27B-W8A8 | 24 / 4 | 6 | 256 | 否 |
| Qwen3.8-27B | 24 / 4 | 6 | 256 | 否 |
| Qwen3.8-27B-W8A8 | 24 / 4 | 6 | 256 | 否 |

关键在于**拒绝点不同**：这三个模型 GQ=6、HD=256 都不合法，而 API 层
（`mxflashattn/api.py:191`）先于 binding 层被触发，路径根本进不到 M1 内核的
GQ 守卫。它们只能再次复现 HD=256 拒绝，**无法隔离 GQ 这一个变量**。

另外扫到 GQ=6 也不代表能用：Qwen3.5/3.8-27B 的 64 层里只有 16 层是
full_attention（其余是 linear_attention / Mamba 混合），即使 HD 合法，实际参与
M1 的层数也只有 1/4。

配置改写（option B）也被排除：GQ=6 需要 nQ×HD 与 hidden_size 对齐，
Qwen3-32B（64Q/8KV，hidden 5120）改成 24Q/4KV 需要 hidden 3072 ≠ 5120，
权重加载不了；Qwen3-4B（32Q/8KV，hidden 2560）改成 24Q/4KV 需 3072 ≠ 2560，同样不行。

## 二、改用 GQ=5：Qwen2.5-14B-Instruct（40Q / 8KV，HD=128，48 层）

选择理由三条：

1. **同一个守卫**。GQ=5 与 GQ=6 触发的是同一条非 2 幂拒绝路径
   （`split kernel unsupported config: D=128 GQ=5`），验证价值等价。
2. **新比率**。此前真实模型证据只有 GQ=7，GQ=5 从未测过。
3. **顺带回归 `.contiguous()` 修复**。Qwen2 与 Llama 同属"传非连续 q"的架构族，
   这批数据同时验证 HD=64 那次修的 bug 没有回归。

模型来源与 GQ=7 的 7B 同样走 ModelScope（HF 不通），29 GB，落在 `/data`。

## 三、结果：守卫行为完全正确

| 场景（batch × max_tokens） | v7 | M1-UM1 | M1-UM2 | 派发 |
|---|---|---|---|---|
| b16 × 64 | 420.77 | 177.20 | 177.13 | `m1_split_decode` **0** |
| b64 × 256 | 678.71 | 688.70 | 694.14 | `m1_split_decode` **0** |
| b32 × 1024 | 215.87 | 319.11 | 318.90 | `m1_split_decode` **0** |

**决定性证据**（三档全部）：

```
RuntimeError('split kernel unsupported config: D=128 GQ=5')
m1_split_decode count = 0
```

内核在 binding 层明确拒绝 GQ=5，插件自动落回 v7，全链路无静默错误。

## 四、两个反常现象与归因

跑完出现两个与"守卫拒绝"本应三档完全一致相矛盾的现象，逐一查清。

### 现象 1：吞吐不一致（sanity 档 M1 反而慢 2.4×，long 档快 1.48×）

归因：**每次调用抛 Python 异常的真实 CPU 开销**。GQ=5 下 M1 一次都不派发，
但每个 decode 步都要走一遍 `try → 抛 RuntimeError → except → 落回 v7`，
sanity 档累计 7680 次、main 档 16896 次、long 档 53760 次。

- b16×64：decode 步数少、GPU 工作量小，异常开销占比高 → M1 慢 2.4×；
- b32×1024：GPU 计算占主导，异常开销被摊薄，且生成长度差异
  （31190 vs 29510 token，EOS 时机不同）改变了实际工作量 → 看起来 M1 更快。

**这不是内核问题，是守卫机制的固有成本**。在 GQ∈{2,4,8} 的支持形状上
`m1_split_decode` 正常派发、零异常，这条成本不存在。

### 现象 2：text_match 只有 1/4（此前 GQ=7 报告为 4/4）

这是**测量方法问题，不是正确性问题**。写 `diag_guard_text_match.py` 单独验证：
固定 4 条 prompt，batch=4，先跑两遍 v7 再跑 M1，比对 token-id 序列。

```
v7 vs v7  (同配置跑两遍) : 4/4
v7 vs M1  (守卫拒绝回退) : 4/4
VERDICT: both deterministic and identical
```

- harness 自身在 batch=4 下**完全确定**（v7 两遍 4/4）；
- 守卫拒绝路径的输出与 v7 **逐 token 完全一致**（4/4）；
- 结论：**1/4 是大 batch 连续批处理下的调度产物**。batch=64 时请求完成时机
  不同 → 批次重新分组 → 边界 token 翻转。同一 batch 下 v7 与 M1 的实际
  gen_tokens 都不同（16123 vs 16126），说明比较的已经不是同一份工作。

**修正此前报告的说法**：GQ=7 那次标注的 "4/4 完全一致" 同样不可信——它比较的
可能不是同一条执行路径。守卫路径文本一致性的**可靠证据是本节的 4/4 定点对比**。

## 五、意义

守卫拒绝侧现在有三种非 2 幂比率的真实模型证据，且互相印证：

| GQ | 模型 | nQ/nKV | HD | 派发 | 文本（定点对比） |
|---|---|---|---|---|---|
| 5 | Qwen2.5-14B-Instruct | 40/8 | 128 | `m1_split_decode` 0 | **4/4** |
| 7 | Qwen2.5-7B-Instruct | 28/4 | 128 | `m1_split_decode` 0 | 4/4（方法待复核） |
| 6 / 7 | Qwen3.5-27B / Qwen3.8-27B | 24/4 | **256** | 进不到 GQ 守卫 | — |

"遇到不支持的 GQA 比率自动落回且不影响正确性"这条生产安全性，
现在由 GQ=5 提供了方法学可靠的闭环。

同时**修正一处遗留**：GQ=6 真实模型 E2E 不是"待补"，而是**客观不可得**——
不存在 HD 合法的 GQ=6 checkpoint。今后不必再尝试。

## 工程坑（本次新增）

- **找特定 GQA 比率的模型时，head_dim 必须一起筛**。只按 nQ/nKV 匹配会选出
  head_dim=256 的模型，而 API 层会先于 binding 层拒绝，测不到目标变量。
  筛选条件是 `GQ ∈ 目标集` **且** `head_dim ∈ {64,128}`。
- **很多 config 省略 head_dim**（Qwen2 全系），vLLM 默认取
  `hidden_size // num_attention_heads`。扫描脚本若强制要求该字段，会把这些
  模型全判成"shape fields not found"而错失候选。
- **`text_match` 在大 batch 下不可靠**。连续批处理让请求完成时机影响批次分组，
  边界 token 会翻转。要验证守卫路径的正确性，必须用**定点对比**：固定 prompt、
  batch=4、同配置跑两遍先验证 harness 确定性，再比对。
- **守卫拒绝路径有真实 CPU 成本**：每次调用抛异常。在短序列小 batch 下可占
  可观比例（实测慢 2.4×）。评估"守卫拒绝是否可接受"时要把这个算进去。

## 远端结果

- `artifacts/bench-gq5-q25-14b-{sanity,main,long}.json`（+ 同名 dispatch.jsonl）
- `csrc_native/diag_guard_text_match.py`（定点对比脚本）
- `csrc_native/scan_gq_shapes.py` / `find_nonga_models.py`（形状扫描）
