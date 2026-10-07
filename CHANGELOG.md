# CHANGELOG

## v1.3.0 (2026-10-07) - M1 paged-KV decode kernel

### 新增
- `csrc_native/m1_kernel.cpp` + `m1_binding.cpp`：split-KV + combine decode 内核
  （grid.z 分 chunk 部分状态 + combine 归并），统一 max（mode 10）与分桶惰性缩放
  （mode 11，默认，零代价消除溢出边界），cp_async 计数等待流水。
- `csrc_native/bench_e2e.py`：vLLM continuous batching 批次并发 E2E 基准。
- `csrc_native/test_m1.py`：稳健性矩阵（s1~s8192 / GQA 2,4,8 / bf16 / ovf 溢出攻击），
  支持按 tag 子进程隔离执行。
- `docs/gate4/`：收官报告、M3 接线证据、参考文献与调研记录、全部 smoke/bench JSON。
- `docs/gate4/E2E_大模型与守卫验证.md`：大模型端到端与守卫拒绝侧验证报告
  （Qwen3-4B / Qwen3-8B 吞吐矩阵、Qwen2.5-7B GQ=7 守卫闭环、跨模型一致性分析、
  PaddleOCR-VL GQ=8 全路径、Qwen3.5-9B HD=256 拒绝、TinyLlama HD=64 全路径）。

### 修复
- binding GQ 守卫收紧为当前已验证 contract `{2,4,8}`；GQ=3/5/6/7 在当前布局下实测会产生静默错误，其他未验证比率（如 GQ=16）不作支持承诺，不支持比率返回 -1
  并由 vLLM 插件自动落回 v7，杜绝非 2 幂 GQA 的静默错误结果。
- **整个 Llama 族在 vLLM 下的 decode 路径失效**：vLLM 传入的 q 是从融合 QKV buffer
  切出的非连续视图（batch 维按最大 batch 分配），`q.reshape()` 不保证连续，binding 的
  `q.is_contiguous()` 守卫同时拒绝 M1 与 v7 两条路径，使自研 decode 整体退化为
  vendor 直连。插件在 M1 与 v7 分支各补一次显式 `.contiguous()`（连续时为 no-op）。
- v7 路径确认上下文上限 4096（>4096 启动失败），测试侧加跳过守卫。

### 性能
- 内核级：M1 延迟为 vendor `flash_attn_with_kvcache` 的 2.95x–7.43x（当前仍慢于 vendor）；相对 v7 自研路径最高提升 42x。
- E2E（Qwen3-0.6B，continuous batching）：256tok/batch128 +32%；1024tok/batch64 4.39x；
  2048tok/batch32 5.31x；decode 派发 0 回退。
- E2E（Qwen3-4B，GQ=4/HD=128/36 层，continuous batching）：256tok/batch128 3.20x；
  256tok/batch64 2.13x；1024tok/batch32 3.67x；1024tok/batch16 2.76x；短序列
  64tok/batch16 持平。派发 0 回退，文本一致 3/4。
- E2E（Qwen3-8B，同形状 hidden 4096）：256tok/batch128 2.94x；256tok/batch64 2.20x；
  1024tok/batch32 3.49x；1024tok/batch16 2.86x。四个主场景与 4B 逐一复现、偏差 <9%
  —— split-KV 收益取决于 batch×序列形态而非模型规模。派发 0 回退，文本一致 3/4。
- 大模型上 v7 自研 decode 比 vendor 直连慢 3.2x（1383 vs 4453 tok/s @4B b128x256），
  M1 将其拉回 vendor 水平并略优（4428，-0.5%）——split-KV 是自研 decode 可用的前提。
- GQ 守卫拒绝侧真实模型闭环（Qwen2.5-7B-Instruct，28Q/4KV = GQ=7）：三档 dispatch
  全部 `native_decode`、`m1_split_decode` 出现 0 次，吞吐差异 <1%，生成文本与 v7
  4/4 完全一致 —— 非 2 幂 GQA 被干净拒绝且不影响正确性。
- E2E（Qwen2.5-7B-Instruct GQ=7，continuous batching）：b16×64 持平、b32×256 持平
  （守卫拒绝，三档均落回 v7），文本 4/4。
- **GQ=8 全路径打通（PaddleOCR-VL，16Q/2KV，continuous batching）**：b16×64 1.20x；
  b64×256 1.53x；b32×1024 2.47x。派发 1134/4590/18414 次 `m1_split_decode`、0 回退，
  文本 4/4。
- HD=256 拒绝路径（Qwen3.5-9B，GQ=4 合法但 head_dim=256 越界）：公开 API 层
  `mxflashattn/api.py:191` 显式 ValueError 拒绝，非静默错误。两层守卫（API 头维度 +
  binding GQ）合计覆盖当前已定义的 unsupported contract，不对未纳入 contract 的其他组合作支持承诺。
- **GQ=5 守卫拒绝侧真实模型闭环（Qwen2.5-14B-Instruct，40Q/8KV = GQ=5，HD=128，48 层）**：内核 binding 层明确拒绝（`split kernel unsupported config: D=128 GQ=5`），三档 `m1_split_decode` 均为 0。另确认 **GQ=6 客观不可得**：40+ 个主流 checkpoint 中的 GQ=6 全部 head_dim=256，会被 API 层先拒。定点对比实测 v7 vs v7 = 4/4、v7 vs M1 = 4/4，守卫路径文本逐 token 一致。
- **HD=64 × GQ=8 全路径打通（TinyLlama-1.1B-Chat-v1.0，32Q/4KV，continuous batching）**：
  b16×64 0.97x（发射延迟区持平）；b64×256 **2.96x**；b32×1024 **6.48x**。派发
  1386/5610/22506 次 `m1_split_decode`、0 回退，文本 3/4。**至此
  `{GQ∈{2,4,8} × HD∈{64,128}}` 全部 6 个格子均有真实模型 E2E 证据**。修复 q 非连续
  缺陷后，HD=64 上非 split 的 v7 仅 177 tok/s（b32×1024）而 M1 达 1150 —— split-KV
  是该形状 decode 可用的前提，比 4B/8B 上的退化更极端。

### 依赖
- vLLM 接线：`integrations/vllm/vllm_plugin.py` 增加 `MXFA_M1_DECODE=1` 门控分支
  （NP 启发式 + 异常自动落回 v7），`MXFA_M1_UM` 控制 softmax 模式（默认 2=分桶）。


## v1.1.0-vendor-wiring

- Added explicit `vendor_direct` dispatch for the MetaX flash-attn wheel.
- Added `mxmac_aten_correctness` as the named ATen extension fallback.
- Vendor call and ATen fallback failures retain their reasons in `DispatchInfo` and vLLM dispatch JSONL.
- Kept v1.0 evidence as provenance; the 36-case model candidate remains slower than vendor and must be rerun on C500 after this wiring change.
