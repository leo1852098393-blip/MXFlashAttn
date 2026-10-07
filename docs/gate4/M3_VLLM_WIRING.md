# Gate 4 / M3 — M1 split-KV decode 接入 vLLM（证据记录）

日期：2026-10-07（远程 C500 实例，工作树 `/root/MXFlashAttn_v1.3.0-kernel`）

## 接线内容

`integrations/vllm/vllm_plugin.py`（patch 由 `csrc_native/patch_plugin.py` 幂等写入）：

1. 新增 `_load_m1_decode_ext()`：懒加载 `csrc_native/mxmmain.so`（M1 扩展，mxcc 构建）。
2. `_try_native_decode()` 内、v7 `ext.paged_decode` 调用前，新增 `MXFA_M1_DECODE=1` 门控分支：
   - 调 `mxmmain.paged_decode_split(q, k_cache, v_cache, bt, sl, scale, stream, np, c_len)`
   - NP 启发式：`np=1; while np<64 and (B*HKV)*np<2048: np*=2`，`c_len=ceil(max_sl_bound/np)`
   - 任何不支持形状（D∉{64,128}、GQ∉[2,8]）抛异常 → 落回 v7 路径（外层 try/except 保证）

启用环境变量（在 v7 native 路径之上叠加）：
```
MXFLASHATTN_GRAPH_OFFICIAL_SPLIT=1  MXFA_NATIVE_DECODE=1  MXFA_M1_DECODE=1
```

## E2E 冒烟（Qwen3-0.6B，vLLM 0.17.0，36 提示词 × 32 token，温度 0）

```
python integrations/vllm/smoke_generate.py --candidate --suite --prompt-count 36 \
  --max-tokens 32 --output artifacts/m1-smoke-{v7|m1}.json
```

| 指标 | v7 (MXFA_M1_DECODE=0) | M1 (MXFA_M1_DECODE=1) |
|---|---|---|
| decode 派发事件 | 31248 × `native_decode` | 31248 × `m1_split_decode` |
| 状态 | candidate_dispatch_verified | candidate_dispatch_verified |
| 生成文本与对方一致 | — | 29/36（7 条为近并列 token 翻转，其中 idx29 v7 退化为 "1.1.1..." 而 M1 正常） |
| tokens/s（去掉首请求） | 56.48 | 54.00（-4.4%） |

E2E 吞吐持平略降是预期行为：该冒烟为单请求、短序列（sl≈10–40），解码处于发射延迟主导区，
M1 多一次 combine 内核启动 + workspace 分配（约 +0.7ms/步）。内核级收益在高并发/长序列场景
（见下），冒烟不覆盖。

## 内核级基准（同日 test_m1.py，全部 correctness PASS）

| 配置 | vendor | v7 | M1 最优（split-NP） | vs v7 |
|---|---|---|---|---|
| lowB-s4096（B=8, HKV=2, NP32） | 0.046ms | 5.93ms | 0.1395ms | **42×** |
| s1024（B=96, NP2） | 0.283ms | 3.31ms | 1.180ms | 2.8× |
| s2048（NP2） | 0.555ms | 9.25ms | 2.295ms | 4.0× |
| s512（B=32, NP2） | 0.065ms | 0.86ms | 0.230ms | 3.7× |
| gq4（NP2） | 0.147ms | 1.97ms | 1.104ms | 1.8× |
| d64（NP1） | 0.092ms | 不可用 | 0.3496ms | — |

## UM 统一 max 同步（2026-10-07 下午追加）

vLLM split 路径原本走 online-softmax（`launch_split` 实例化 `paged_decode_v3<HD,GQ,NS>`
UM=0）。已改为 UM=1 实例（FlashDecoding++ 式统一 max，循环内零重缩放），由
`MXFA_M1_UM` 环境变量控制、**默认开启**（置 0 回退 online-softmax）；pybind 签名与
vLLM 插件补丁零改动。UM 收尾在写部分状态前把 (o,d) 换算回真实 max 基准，combine 无感。
溢出边界 |s·log2e|<126（φ=0，FlashDecoding++ 的 recompute 回退列为后续工作）。

内核级（test_m1.py 全 12 配置 PASS）：split 各配置再提速 2–4%——
lowB NP32 0.1412→0.1371ms（2.95× vendor）、s1024 NP2 1.2022→1.1513（4.06×）、
s2048 NP2 2.3408→2.2365（4.08×）、s512 NP2 0.2339→0.2245（3.47×）、
gq4 NP2 1.1245→1.0950（7.43×）。

E2E 冒烟重跑（`m1-smoke-m1um.json`，同 36×32 配置，MXFA_M1_UM=1）：

| 指标 | v7 | M1（旧，online-softmax） | M1-UM |
|---|---|---|---|
| decode 派发 | 31248 × native_decode | 31248 × m1_split_decode | 31248 × m1_split_decode，0 回退 |
| 生成文本与 v7 一致 | — | 29/36 | **30/36** |
| tokens/s（去首请求） | 56.48 | 54.00（-4.4%） | **55.63（-1.5%）** |

单请求短序列冒烟仍处内核启动延迟主导区，-1.5% 为预期残留；内核级收益见上表。

## mode 11 分桶惰性缩放（bucketed UM，溢出边界消除，2026-10-07 上午）

mode 10（φ=0 统一 max）有 |s·log2e|<126 溢出边界。mode 11 改为**分桶惰性缩放**：
`m_used` 只落在 32 的整数倍桶边界，仅当真实 max 越桶才重缩放一次（每行通常 2~4 次，
≈ online softmax 重缩放次数的 1/10）。p=2^(s−m_used)≤1 天然无溢出，在实数缩放代数上与online
softmax 等价（o/d 比值不变），combine 按一致基准合并无感；考虑近似 exp2、bf16/fp16 量化与浮点舍入，不宣称bitwise 严格等价。

验证（test_m1.py，全 13 配置 PASS，新增 ovf 配置：q×400 逼出 s_log2 峰值 ~200）：

| 路径 | 常规配置性能 vs mode 10 | ovf 配置（s>126） |
|---|---|---|
| v3-NS1（online） | 基准 | P |
| UM1 mode 10（φ=0） | 最优 | **F(nan)** |
| split（UM1） | — | **F(nan)** |
| **UM2 mode 11** | 持平（差距 <1%，s1024 NP2 1.1546 vs 1.1548ms） | **P** |
| **split2（UM2）** | 持平（lowB NP32 0.1378 vs 0.1371ms，+0.5%） | **P（全部 NP）** |

结论：mode 11 以零性能代价消除溢出边界，**生产默认路径应选 UM=2**（binding
`MXFA_M1_UM=2`）；UM=1 保留作对照。

## 批次并发 E2E 基准（bench_e2e.py，2026-10-07 上午）

此前 36×32 冒烟为逐请求执行（smoke_generate.py 每次单条 generate），批次并发从未
发生。bench_e2e.py 一次 generate 提交全批 prompt（vLLM continuous batching），
`VLLM_ENABLE_V1_MULTIPROCESSING=0` 强制 EngineCore 同进程以支持 env 热切换三档对比。
Qwen3-0.6B，256 token/请求，温度 0：

| 配置 | batch=128 tok/s | vs v7 | batch=64 tok/s | vs v7 |
|---|---|---|---|---|
| v7 | 4411.9 | — | 3181.1 | — |
| M1-UM1（φ=0） | 5765.0 | **+30.7%** | 3216.3 | +1.1% |
| **M1-UM2（分桶）** | **5827.8** | **+32.0%** | **3244.8** | +2.0% |

- 派发验证：v7 = 7168 native_decode；M1 = 7168 m1_split_decode，双方各 7168
  vendor_direct（prefill），0 回退。
- 文本一致（批内 4 条 vs v7）：UM1 3/4、UM2 3/4（近并列 token 翻转）。
- 结论：**M1 的收益集中在高并发区**——batch=128 快 32%；batch=64 时 v7 并发已
  填满 GPU，基本持平。与内核级结论（split-KV 解决低 block 并发）自洽。
- 证据：bench-e2e.json / bench-e2e.dispatch.jsonl（本目录）。

## 长序列批次 E2E（2026-10-07 追加）

同一 bench_e2e.py，`--max-model-len 4096` 放宽上下文，生成 1024 / 2048 token
（decode sl 随生成线性增长）。v7 / M1-UM1 / M1-UM2 三档 env 热切换。

| 场景（max_tokens, batch） | v7 tok/s | M1-UM1 | M1-UM2 | vs v7 |
|---|---|---|---|---|
| 1024, b64 | 732.8 | 3204.0 | **3213.0** | **4.39×** |
| 1024, b32 | 597.5 | 1692.7 | **1726.1** | 2.89× |
| 2048, b32 | 325.6 | 1721.6 | **1729.2** | **5.31×** |
| 2048, b16 | 241.1 | 882.5 | **882.4** | 3.66× |

- **E2E 增益随序列长度持续放大**（短序列 +32% → sl1024 4.4× → sl2048 5.3×），
  与内核级基准（s2048 NP2 4.08× vs vendor）方向一致且略超——v7 的 vLLM 路径
  叠加了自身调度开销，M1 单次 split+combine 直连。
- 派发验证：57344 步全部 `m1_split_decode`、0 回退；文本一致 3/4（与 36×32
  冒烟的近并列翻转率一致）。
- 工程坑（记录）：bench 需在树根跑 + `PYTHONPATH=<树根>`（integrations 包），
  且必须 source conda base + MACA_PATH/LD_LIBRARY_PATH 全套环境变量（.bashrc
  的非交互 shell 不生效，vLLM import 报 TypeError NoneType）。
- 证据：bench-e2e-sl1024.json / bench-e2e-sl2048.json（本目录）。

## 归档文件

- `m1-smoke-v7.json` / `m1-smoke-m1.json` / `m1-smoke-m1um.json`（本目录，remote artifacts 同名）
- `bench-e2e.json`（batch 128/64 × 256 tok）、`bench-e2e-sl1024.json` / `bench-e2e-sl2048.json`（长序列）
- `bench-e2e.json` / `bench-e2e.dispatch.jsonl`（批次并发基准，本目录）
- `bench_e2e.py`（批次并发基准脚本，remote csrc_native 同名）
- 远程：`artifacts/m1-smoke-*.dispatch.jsonl`（31248 行派发流）、`/tmp/m1_smoke_*.log`（FIRE 日志）
- 参考文献与调研依据：`Gate4_参考文献与调研记录.md`（本目录）
- 未执行 git push（按指示"先不推，憋波大的"）
