# v1.4 技术说明

## Prefill 原型

Prefill 原型把 KV 序列按 `chunk_size` 分段。每个 chunk 计算 attention
logits，并维护三个状态：

- 当前已处理区域的最大 logit；
- 在线 softmax 的归一化和；
- 已归一化前的 value 加权和。

当新的 chunk 出现更大 logit 时，旧状态按指数因子缩放，再合并当前 chunk。
这样不需要把完整 attention 矩阵一次性保存在内存中。

当前实现使用 PyTorch 和 FP32 累加，目标是验证数学过程和数值行为，不是
C500 device kernel。真实性能结论必须在 C500 上与 vendor/v7 同配置测量。

## 调度策略

`AdaptiveDispatchPolicy` 只接收形状元数据，不访问设备，也不执行算子。这样
可以先单测决策边界，再把 C500 实测阈值注入未来的 vLLM adapter。

当前候选条件：

- Prefill：batch 至少 4、KV 长度至少 512；
- Decode：batch 至少 16、KV 长度至少 256、GQ 为 2/4/8、head_dim 为 64/128。

这些数字是实验候选，不是已标定的生产阈值。默认策略关闭。

## 长上下文验证

误差报告必须同时包含：

- `finite`；
- `max_abs`、`mean_abs`；
- P95、P99 绝对误差；
- 最大相对误差；
- 固定输入、随机种子、dtype、形状和 Git commit。

任何 NaN/Inf 都应使 `finite=false`，并把对应误差指标记录为 `inf`，不能被
平均值掩盖。

## vLLM 兼容边界

当前运行时集成的真实合同是 vLLM 0.17.0。v1.4 只提供版本分类和非侵入式
probe 入口。vLLM 0.30 的接口变化必须在目标环境中读取实际对象和签名后再实现，
不能根据路线图中的示例代码直接猜测。
