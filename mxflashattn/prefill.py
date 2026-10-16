from __future__ import annotations

import math

import torch

from .reference import attention_batched
from .validation import validate_qkv, validate_softmax_scale


def prefill_chunked_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    chunk_size: int = 256,
    softmax_scale: float | None = None,
    causal: bool = False,
) -> torch.Tensor:
    """Correctness-first chunked attention prototype for Prefill.

    The tensors use the public layout ``[batch, sequence, heads, head_dim]``.
    KV is scanned in chunks and combined with an online softmax accumulator.
    This is a reference implementation for algorithm validation, not a
    C500-optimized kernel and not part of the production dispatch path.
    """
    validate_qkv(q, k, v)
    validate_softmax_scale(softmax_scale)
    if not isinstance(chunk_size, int) or chunk_size <= 0:
        raise ValueError("chunk_size must be a positive integer")

    batch, query_len, query_heads, head_dim = q.shape
    key_len = k.shape[1]
    scale = float(softmax_scale) if softmax_scale is not None else 1.0 / math.sqrt(head_dim)
    groups = query_heads // k.shape[2]

    # Use float accumulators to make the prototype's numerical behavior
    # comparable across float16, bfloat16, and float32 inputs.
    qf = q.float().transpose(1, 2)
    kf = k.float().transpose(1, 2).repeat_interleave(groups, dim=1)
    vf = v.float().transpose(1, 2).repeat_interleave(groups, dim=1)
    output = torch.zeros((batch, query_heads, query_len, head_dim), device=q.device, dtype=torch.float32)
    normalizer = torch.zeros((batch, query_heads, query_len, 1), device=q.device, dtype=torch.float32)
    running_max = torch.full_like(normalizer, float("-inf"))

    query_positions = torch.arange(query_len, device=q.device) + (key_len - query_len)
    for start in range(0, key_len, chunk_size):
        end = min(start + chunk_size, key_len)
        scores = torch.matmul(qf, kf[:, :, start:end].transpose(-2, -1)) * scale

        if causal:
            key_positions = torch.arange(start, end, device=q.device)
            blocked = key_positions.unsqueeze(0) > query_positions.unsqueeze(1)
            scores = scores.masked_fill(blocked.unsqueeze(0).unsqueeze(0), float("-inf"))

        chunk_max = scores.amax(dim=-1, keepdim=True)
        old_valid = torch.isfinite(running_max)
        chunk_valid = torch.isfinite(chunk_max)
        new_max = torch.maximum(running_max, chunk_max)

        old_factor = torch.where(
            old_valid,
            torch.exp(running_max - new_max),
            torch.zeros_like(running_max),
        )
        exp_scores = torch.where(
            torch.isfinite(scores),
            torch.exp(scores - new_max),
            torch.zeros_like(scores),
        )
        normalizer = normalizer * old_factor + exp_scores.sum(dim=-1, keepdim=True)
        output = output * old_factor
        output = output + torch.matmul(exp_scores, vf[:, :, start:end])
        running_max = torch.where(chunk_valid | old_valid, new_max, running_max)

    output = torch.where(
        normalizer > 0,
        output / normalizer.clamp_min(torch.finfo(output.dtype).tiny),
        torch.zeros_like(output),
    )
    return output.transpose(1, 2).to(dtype=q.dtype)


def prefill_chunked_reference_fallback(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    chunk_size: int = 256,
    softmax_scale: float | None = None,
    causal: bool = False,
) -> torch.Tensor:
    """Convenience wrapper used by local experiments.

    The fallback provides a single switch for comparing the prototype with the
    existing readable reference implementation. It is intentionally not used
    by the runtime dispatcher.
    """
    if chunk_size >= k.shape[1]:
        return attention_batched(q, k, v, softmax_scale=softmax_scale, causal=causal)
    return prefill_chunked_reference(
        q,
        k,
        v,
        chunk_size=chunk_size,
        softmax_scale=softmax_scale,
        causal=causal,
    )
