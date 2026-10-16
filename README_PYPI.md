# MXFlashAttn

面向 **MetaX C500 / MXMACA** 的 FlashAttention-compatible 前向 API 与
vLLM 推理适配组件。

`mxflashattn` 提供稳定的 Python 调用入口、参数校验、后端选择和显式回退
机制。对于已经部署 MetaX 软件栈的 C500 用户，它还提供经过真实模型验证的
M1 Split-KV Decode 集成路径。

> 当前 PyPI 包是 Python API 与适配层。C500 原生 kernel 需要在目标实例上按
> MetaX/MACA 环境编译，详见“C500 使用”。

## 适合什么场景

- 在 Python 项目中使用 FlashAttention 风格的前向接口。
- 为 MXMACA / MetaX C500 推理服务接入 vLLM 0.17.0。
- 在 C500 上对高并发、长 KV cache 的 Decode 阶段使用 M1 Split-KV kernel。
- 在没有原生扩展的环境中，显式启用 PyTorch reference 路径进行接口调试和正确性测试。

## 安装

```bash
python -m pip install mxflashattn==1.4.1
```

基础依赖：

- Python 3.10+
- PyTorch 2.4+
- PyYAML 6.x

vLLM 适配依赖需要单独安装：

```bash
python -m pip install "mxflashattn[vllm]==1.4.1"
```

当前适配目标是 vLLM 0.17.0，并且需要兼容的 MetaX vLLM/MACA 软件栈。

## Python API

公开入口：

```python
from mxflashattn import (
    flash_attn_func,
    flash_attn_varlen_func,
    flash_attn_with_kvcache,
    get_last_dispatch_info,
)
```

一个最小的显式 reference 调试示例：

```python
import os

import torch

from mxflashattn import flash_attn_func, get_last_dispatch_info

os.environ["MXFLASHATTN_ALLOW_FALLBACK"] = "1"

q = torch.randn(1, 32, 8, 64, dtype=torch.float16)
k = torch.randn(1, 128, 2, 64, dtype=q.dtype)
v = torch.randn_like(k)

out = flash_attn_func(q, k, v, causal=True)
print(out.shape)
print(get_last_dispatch_info())
```

reference 路径只用于调试和正确性对照，不是 C500 性能路径。没有显式设置
`MXFLASHATTN_ALLOW_FALLBACK=1` 时，缺少原生后端会直接报错，避免程序静默降速。

## C500 使用

PyPI wheel 不包含针对特定 C500/MACA 环境编译的 `.so` 文件。要启用 M1
Split-KV Decode，需要从源码仓库获取对应版本并在 C500 实例中编译：

```bash
git clone -b dev/v1.4.0 https://github.com/leo1852098393-blip/MXFlashAttn.git
cd MXFlashAttn

export MACA_PATH=/opt/maca
bash csrc_native/build_m1.sh
python csrc_native/patch_plugin.py
```

然后按项目中的 vLLM 0.17.0 运行说明启动服务或基准。运行性能测试时，必须
检查 dispatch 日志中是否出现 `m1_split_decode`，否则不能把结果归因于 M1。

## 已验证能力

在 MetaX C500、MACA 3.5.3.20、MetaX PyTorch
`2.8.0+metax3.5.3.9` 和 vLLM 0.17.0 环境中，项目已验证：

- FP16 / BF16 前向；
- `head_dim` 64 / 128；
- GQA 比率 2 / 4 / 8；
- vLLM 真实模型生成和 M1 dispatch；
- 不支持形状的安全回退；
- 15,000 token prompt、`max_model_len=16384` 的 vendor Prefill + M1 Decode 回归。

在一组 Qwen3-4B C500 长上下文 Decode 配置中，M1-UM2 相对项目 v7 路径
达到约 3.66x；Qwen3-8B 的对应测试约 1.37x。数字只适用于记录中的模型、
batch、KV 长度、软件栈和基线，不代表所有请求都有固定加速。

## 重要边界

- 项目面向 MetaX C500，不是跨硬件 FlashAttention 替代品。
- M1 的生产验证重点是 Decode；Prefill 仍交给 MetaX vendor。
- `prefill_chunked_reference` 是 correctness-first 研究原型，不是生产 Prefill kernel。
- 短序列、低并发和不支持的 GQA/head dimension 可能回退或没有收益。
- 当前没有对 vLLM 0.30.x、FP8、多卡和 SGLang 做支持承诺。
- backward、dropout、局部窗口、soft-capping、rotary embeddings 等未实现能力
  会显式报错，不会静默返回不可靠结果。

## 文档与源码

- GitHub：[leo1852098393-blip/MXFlashAttn](https://github.com/leo1852098393-blip/MXFlashAttn)
- GitLink：[Leo77/MXFlashAttn](https://gitlink.org.cn/Leo77/MXFlashAttn)
- API 说明：[docs/API.md](https://github.com/leo1852098393-blip/MXFlashAttn/blob/dev/v1.4.0/docs/API.md)
- C500 支持矩阵：[docs/support-matrix.md](https://github.com/leo1852098393-blip/MXFlashAttn/blob/dev/v1.4.0/docs/support-matrix.md)

## 许可证

Apache License 2.0。

