# v1.4.0 C500 实验计划

**目的**：在租用 C500 前固定实验口径，完成 Prefill 原型、TTFT/P95、长上下文
稳定性和候选调度阈值的最小验证。  
**计划文件**：`benchmarks/configs/v14_c500_prefill.json`  
**清单生成器**：`scripts/prepare_c500_v14.py`

## 执行前

在目标工作树执行：

```bash
python scripts/prepare_c500_v14.py \
  --output artifacts/c500-v14/manifest.json
```

先记录清单中的 Git commit，再开始编译和测量。远程实例信息、密码、模型权重
和原始日志只保存在受限证据目录，不进入公开仓库。

## 实验分组

### 1. Prefill 对比

每个模型/场景固定运行：

- MetaX vendor；
- 项目 v7 baseline；
- `m1_prefill_chunked` correctness prototype。

Prefill 只比较相同输入、相同 dtype、相同 causal 设置和相同 warmup；原型结果
在没有 C500 真实测量前不写成性能结论。

### 2. 端到端指标

必须记录：

- Prefill median/P95；
- TTFT median/P95；
- Decode tokens/s；
- 峰值显存；
- 固定 batch 的数值误差和 token 对照；
- `m1_split_decode`、`native_decode`、vendor 和 fallback 事件数。

所有性能数字必须能追溯到命令、Git commit、环境版本和原始 dispatch 日志。

### 3. Guard / fallback

- GQ=7：预期 M1 dispatch 为 0，走 v7 或 vendor fallback。
- head_dim=256：预期 M1 dispatch 为 0，走 vendor fallback。

Guard case 的“没有 M1 dispatch”是正确结果，不是测试失败。

### 4. 长上下文

优先执行 Qwen3-4B 的 4096 和 16384 token 档。64G 显存可能接近上限，
执行前先确认模型、batch 和 KV cache 配置；如显存不足，记录失败原因，不擅自
降低配置后把结果当成同一实验。

## 停止条件

满足以下任一条件就停止扩展实验，先保存证据：

- 出现 NaN/Inf、错误 token 或 dispatch 路由异常；
- 显存余量不足；
- 某个候选策略在同一场景稳定劣于 baseline；
- 关键环境版本与 v1.3 基线不一致。

## 结果判定

本次 C500 实验只回答三个问题：

1. Prefill 原型是否正确；
2. 哪些场景值得进入专用 C500 kernel；
3. 哪些 batch/KV 长度边界可以作为调度候选。

在完整证据归档和回归完成前，不把候选阈值写入默认生产 dispatch，也不更新
v1.3.0 公开基线。
