# v1.3 最小 C500 验证集

本文件只定义下一轮校准所需的最小实验，不代表本地已经完成这些实验。
目标是得到可复核的 correctness、dispatch、吞吐、P95、TTFT 和峰值显存记录，
并据此校准本地调度阈值。

## 固定原则

- 每个配置分别运行 vendor、v7 和 M1，固定模型、prompt、dtype、随机种子和 warmup。
- 每次 M1 结果必须检查 `dispatch.jsonl`，确认事件包含 `m1_split_decode`。
- GQ=7、GQ=5、head_dim=256 只跑 guard/correctness，不计入 M1 性能。
- 记录环境版本、模型路径、commit、编译产物哈希和运行命令。

## 最小矩阵

1. Qwen3-4B，GQ=4、head_dim=128：batch `1/16/32/64/128`，短/中/长 KV。
2. Qwen3-8B，GQ=4、head_dim=128：至少短序列和长序列各一档。
3. 一个 GQ=8、head_dim=64 或 128 的真实模型：短/长序列各一档。
4. GQ=7 或 GQ=5：确认 M1 dispatch 为 0，结果回退到 v7 且 correctness 通过。

## 每项输出

- correctness：固定 batch 的 token-id 或数值误差对照。
- 性能：吞吐、单步延迟、P95、TTFT（如 harness 可测）。
- 资源：峰值显存、显存余量、进程和设备状态。
- 路由：`m1_split_decode`、`native_decode`、`vendor_direct`、fallback reason 计数。

## 结果落盘

建议每个配置写入 `artifacts/c500-minimal/<case>/`，并由本地证据索引工具登记。
没有完整 dispatch 和环境信息的数字不得写入申报材料或 README。
