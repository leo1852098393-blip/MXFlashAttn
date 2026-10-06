from __future__ import annotations

import math
from collections.abc import Sequence

import torch


SUPPORTED_HEAD_DIMS = (64, 128)
REFERENCE_DTYPES = (torch.float16, torch.bfloat16, torch.float32)
NATIVE_DTYPES = (torch.float16, torch.bfloat16)


def validate_softmax_scale(softmax_scale: float | None) -> None:
    if softmax_scale is None:
        return
    try:
        scale = float(softmax_scale)
    except (TypeError, ValueError) as error:
        raise TypeError("softmax_scale must be a finite number or None") from error
    if not math.isfinite(scale):
        raise ValueError("softmax_scale must be finite")


def validate_qkv(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> None:
    tensors = {"q": q, "k": k, "v": v}
    for name, tensor in tensors.items():
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor")
        if tensor.ndim != 4:
            raise ValueError(f"{name} must have rank 4 with layout [batch, seq, heads, dim]")

    if q.device != k.device or q.device != v.device:
        raise ValueError("q, k, and v must be on the same device")
    if q.dtype != k.dtype or q.dtype != v.dtype:
        raise ValueError("q, k, and v must have the same dtype")
    if q.dtype not in REFERENCE_DTYPES:
        raise TypeError("supported dtypes are float16, bfloat16, and float32 for reference execution")

    batch, q_len, q_heads, head_dim = q.shape
    kv_batch, kv_len, kv_heads, kv_dim = k.shape
    if v.shape != k.shape:
        raise ValueError("k and v must have identical shapes")
    if batch != kv_batch:
        raise ValueError("q, k, and v batch dimensions must match")
    if q_heads <= 0 or kv_heads <= 0:
        raise ValueError("q and kv head counts must be positive")
    if q_heads % kv_heads:
        raise ValueError("the number of q heads must be divisible by the number of kv heads")
    if head_dim != kv_dim or head_dim not in SUPPORTED_HEAD_DIMS:
        raise ValueError(f"head dimension must be one of {SUPPORTED_HEAD_DIMS} and match across q, k, and v")
    if q_len < 0 or kv_len <= 0:
        raise ValueError("q must have a non-negative sequence length and k/v must be non-empty")


def validate_options(
    *,
    dropout_p: float,
    window_size: Sequence[int],
    softcap: float,
    alibi_slopes: torch.Tensor | None,
    return_attn_probs: bool,
    deterministic: bool,
) -> None:
    if dropout_p != 0:
        raise NotImplementedError("dropout is not supported in the forward-only implementation")
    if tuple(window_size) != (-1, -1):
        raise NotImplementedError("local window attention is not supported")
    if softcap != 0:
        raise NotImplementedError("soft-capping is not supported")
    if alibi_slopes is not None:
        raise NotImplementedError("alibi_slopes are not supported")
    if return_attn_probs:
        raise NotImplementedError("return_attn_probs is not supported")
    if deterministic:
        raise NotImplementedError("deterministic mode is not currently supported")


def validate_cumulative_lengths(
    cu_seqlens: torch.Tensor,
    *,
    total_tokens: int,
    max_seqlen: int,
    name: str,
    require_non_empty: bool = False,
) -> list[int]:
    if not isinstance(cu_seqlens, torch.Tensor) or cu_seqlens.ndim != 1:
        raise TypeError(f"{name} must be a one-dimensional torch.Tensor")
    if cu_seqlens.dtype not in (torch.int32, torch.int64):
        raise TypeError(f"{name} must use int32 or int64")
    values = [int(value) for value in cu_seqlens.detach().cpu().tolist()]
    if len(values) < 2 or values[0] != 0 or values[-1] != total_tokens:
        raise ValueError(f"{name} must start at zero and cover all tokens")
    if any(right < left for left, right in zip(values, values[1:])):
        raise ValueError(f"{name} must be monotonically non-decreasing")
    if require_non_empty and any(right == left for left, right in zip(values, values[1:])):
        raise ValueError("key/value sequences must be non-empty")
    if max_seqlen < 0 or max(values[i + 1] - values[i] for i in range(len(values) - 1)) > max_seqlen:
        raise ValueError(f"{name} contains a sequence longer than max_seqlen")
    return values
