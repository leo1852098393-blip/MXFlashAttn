# MXFlashAttn v1.4 API 说明

## 当前稳定接口

以下接口来自 v1.3.0 基线，v1.4.0 不改变它们的默认行为：

- `flash_attn_func(q, k, v, ...)`
- `flash_attn_varlen_func(q, k, v, cu_seqlens_q, cu_seqlens_k, ...)`
- `flash_attn_with_kvcache(q, k_cache, v_cache, ...)`
- `get_last_dispatch_info()`

公开张量布局是 `[batch, sequence, heads, head_dim]`。当前 M1 真实验证边界是
`GQ in {2, 4, 8}` 和 `head_dim in {64, 128}`；不满足边界的 vLLM 路径必须回退，
公共 API 仍会明确拒绝不支持形状。

## v1.4 研究接口

### `prefill_chunked_reference`

这是正确性优先的 Prefill 分块原型：

```python
from mxflashattn import prefill_chunked_reference

output = prefill_chunked_reference(
    q,
    k,
    v,
    chunk_size=256,
    causal=True,
)
```

它使用在线 softmax 累积跨 KV chunk 的结果。当前实现用于本地参考测试，
不接入 runtime dispatcher，也不代表 C500 生产性能 kernel。

### `AdaptiveDispatchPolicy`

这是无设备访问的候选调度策略：

```python
from mxflashattn import AdaptiveDispatchPolicy, DispatchPolicyConfig, WorkloadShape

policy = AdaptiveDispatchPolicy(
    DispatchPolicyConfig(enabled=True, calibrated=True)
)
decision = policy.select(
    WorkloadShape(
        phase="prefill",
        batch_size=16,
        query_len=1024,
        kv_len=1024,
        head_dim=128,
        gqa_ratio=4,
    )
)
```

默认 `enabled=False`，即使显式启用，在 `calibrated=False` 时也会回退到
`vendor`。阈值只有在 C500 微基准标定后才能进入运行时集成。

### `summarize_error`

用于生成长上下文误差摘要，包括最大绝对误差、均值、P95、P99、最大相对误差
以及是否出现 NaN/Inf。

## 版本兼容

- vLLM 0.17.0：当前已验证。
- vLLM 0.30.x：只做版本识别和 API probe，不宣称兼容。
- 其他版本：未验证。
