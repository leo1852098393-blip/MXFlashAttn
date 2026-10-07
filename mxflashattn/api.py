from __future__ import annotations

import os

import torch

from . import reference
from .dispatch import dispatch
from .validation import validate_cumulative_lengths, validate_options, validate_qkv, validate_softmax_scale


def _resolved_scale(head_dim: int, softmax_scale: float | None) -> float:
    return float(softmax_scale) if softmax_scale is not None else float(head_dim**-0.5)


def _options(
    dropout_p: float,
    window_size: tuple[int, int],
    softcap: float,
    alibi_slopes: torch.Tensor | None,
    deterministic: bool,
    return_attn_probs: bool,
) -> None:
    validate_options(
        dropout_p=dropout_p,
        window_size=window_size,
        softcap=softcap,
        alibi_slopes=alibi_slopes,
        deterministic=deterministic,
        return_attn_probs=return_attn_probs,
    )


def flash_attn_func(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    dropout_p: float = 0.0,
    softmax_scale: float | None = None,
    causal: bool = False,
    window_size: tuple[int, int] = (-1, -1),
    softcap: float = 0.0,
    alibi_slopes: torch.Tensor | None = None,
    deterministic: bool = False,
    return_attn_probs: bool = False,
) -> torch.Tensor:
    validate_qkv(q, k, v)
    validate_softmax_scale(softmax_scale)
    _options(dropout_p, window_size, softcap, alibi_slopes, deterministic, return_attn_probs)
    return dispatch(
        "flash_attn_func",
        q,
        (q, k, v, causal, _resolved_scale(q.shape[-1], softmax_scale)),
        lambda: reference.attention_batched(q, k, v, softmax_scale=softmax_scale, causal=causal),
        native_args=(q, k, v),
        native_kwargs={
            "dropout_p": dropout_p,
            "softmax_scale": softmax_scale,
            "causal": causal,
            "window_size": window_size,
            "softcap": softcap,
            "alibi_slopes": alibi_slopes,
            "deterministic": deterministic,
            "return_attn_probs": return_attn_probs,
        },
    )


def flash_attn_varlen_func(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    dropout_p: float = 0.0,
    softmax_scale: float | None = None,
    causal: bool = False,
    window_size: tuple[int, int] = (-1, -1),
    softcap: float = 0.0,
    alibi_slopes: torch.Tensor | None = None,
    deterministic: bool = False,
    return_attn_probs: bool = False,
) -> torch.Tensor:
    validate_softmax_scale(softmax_scale)
    for name, tensor in (("q", q), ("k", k), ("v", v)):
        if not isinstance(tensor, torch.Tensor) or tensor.ndim != 3:
            raise ValueError(f"{name} must have rank 3 with layout [total_tokens, heads, dim]")
    if k.shape != v.shape:
        raise ValueError("k and v must have identical shapes")
    if q.device != k.device or q.device != v.device:
        raise ValueError("q, k, and v must be on the same device")
    if q.dtype != k.dtype or q.dtype != v.dtype:
        raise ValueError("q, k, and v must have the same dtype")
    if q.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise TypeError("supported dtypes are float16, bfloat16, and float32 for reference execution")
    if q.shape[1] <= 0 or k.shape[1] <= 0 or q.shape[1] % k.shape[1]:
        raise ValueError("q heads must be a positive multiple of kv heads")
    if q.shape[2] not in (64, 128) or k.shape[2] != q.shape[2]:
        raise ValueError("head dimension must be 64 or 128 and match across q, k, and v")
    if not isinstance(cu_seqlens_q, torch.Tensor) or not isinstance(cu_seqlens_k, torch.Tensor):
        raise TypeError("cu_seqlens_q and cu_seqlens_k must be torch.Tensor values")
    if cu_seqlens_q.device != q.device or cu_seqlens_k.device != q.device:
        raise ValueError("cu_seqlens_q and cu_seqlens_k must be on the same device as q")
    _options(dropout_p, window_size, softcap, alibi_slopes, deterministic, return_attn_probs)
    q_offsets = validate_cumulative_lengths(
        cu_seqlens_q, total_tokens=q.shape[0], max_seqlen=max_seqlen_q, name="cu_seqlens_q"
    )
    k_offsets = validate_cumulative_lengths(
        cu_seqlens_k,
        total_tokens=k.shape[0],
        max_seqlen=max_seqlen_k,
        name="cu_seqlens_k",
        require_non_empty=True,
    )
    if len(q_offsets) != len(k_offsets):
        raise ValueError("cu_seqlens_q and cu_seqlens_k must describe the same batch size")

    return dispatch(
        "flash_attn_varlen_func",
        q,
        (
            q,
            k,
            v,
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q,
            max_seqlen_k,
            causal,
            _resolved_scale(q.shape[-1], softmax_scale),
        ),
        lambda: reference.attention_varlen(
            q, k, v, q_offsets, k_offsets, softmax_scale=softmax_scale, causal=causal
        ),
        native_args=(q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k),
        native_kwargs={
            "dropout_p": dropout_p,
            "softmax_scale": softmax_scale,
            "causal": causal,
            "window_size": window_size,
            "softcap": softcap,
            "alibi_slopes": alibi_slopes,
            "deterministic": deterministic,
            "return_attn_probs": return_attn_probs,
        },
    )


def flash_attn_with_kvcache(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    k: torch.Tensor | None = None,
    v: torch.Tensor | None = None,
    cache_seqlens: torch.Tensor | None = None,
    block_table: torch.Tensor | None = None,
    softmax_scale: float | None = None,
    causal: bool = False,
    window_size: tuple[int, int] = (-1, -1),
    softcap: float = 0.0,
    alibi_slopes: torch.Tensor | None = None,
    rotary_cos: torch.Tensor | None = None,
    rotary_sin: torch.Tensor | None = None,
    return_softmax_lse: bool = False,
) -> torch.Tensor:
    validate_softmax_scale(softmax_scale)
    if rotary_cos is not None or rotary_sin is not None:
        raise NotImplementedError("rotary embeddings are not supported")
    if return_softmax_lse:
        raise NotImplementedError("return_softmax_lse is not supported")
    _options(0.0, window_size, softcap, alibi_slopes, False, False)
    if (k is None) != (v is None):
        raise ValueError("k and v must either both be provided or both be omitted")
    if q.ndim != 4 or k_cache.ndim != 4 or v_cache.shape != k_cache.shape:
        raise ValueError("q and cache tensors must have rank 4 and matching k/v cache shapes")
    if q.device != k_cache.device or q.device != v_cache.device:
        raise ValueError("q and cache tensors must be on the same device")
    if q.dtype != k_cache.dtype or q.dtype != v_cache.dtype:
        raise ValueError("q and cache tensors must have the same dtype")
    if q.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise TypeError("supported dtypes are float16, bfloat16, and float32 for reference execution")
    if q.shape[0] != k_cache.shape[0] and block_table is None:
        raise ValueError("dense cache batch dimension must match q")
    if q.shape[2] <= 0 or k_cache.shape[2] <= 0:
        raise ValueError("q and kv cache head counts must be positive")
    if k_cache.shape[1] <= 0:
        raise ValueError("cache page size must be positive")
    if q.shape[3] not in (64, 128) or k_cache.shape[3] != q.shape[3]:
        raise ValueError("head dimension must be 64 or 128 and match across q and caches")
    if q.shape[2] % k_cache.shape[2]:
        raise ValueError("q heads must be divisible by kv cache heads")
    if (k is not None and (k.shape != v.shape or k.ndim != 4)):
        raise ValueError("appended k and v must have identical rank-4 shapes")
    if k is not None and (k.shape[0] != q.shape[0] or k.shape[2:] != k_cache.shape[2:]):
        raise ValueError("appended k/v must match batch, kv heads, and head dimension")
    if k is not None and (k.device != q.device or k.dtype != q.dtype or v.device != q.device or v.dtype != q.dtype):
        raise ValueError("appended k/v must match the query device and dtype")
    if cache_seqlens is None:
        raise ValueError("cache_seqlens is required for dense and paged caches")
    if not isinstance(cache_seqlens, torch.Tensor):
        raise TypeError("cache_seqlens must be a torch.Tensor")
    if cache_seqlens.ndim != 1 or cache_seqlens.numel() != q.shape[0]:
        raise ValueError("cache_seqlens must be one-dimensional with one value per batch item")
    if cache_seqlens.dtype not in (torch.int32, torch.int64):
        raise TypeError("cache_seqlens must use int32 or int64")
    if cache_seqlens.device != q.device:
        raise ValueError("cache_seqlens must be on the same device as q")
    if block_table is not None:
        if not isinstance(block_table, torch.Tensor):
            raise TypeError("block_table must be a torch.Tensor")
        if block_table.ndim != 2 or block_table.shape[0] != q.shape[0]:
            raise ValueError("block_table must have one row per batch item")
        if block_table.dtype not in (torch.int32, torch.int64):
            raise TypeError("block_table must use int32 or int64")
        if block_table.device != q.device:
            raise ValueError("block_table must be on the same device as q")
    else:
        if k_cache.shape[0] != q.shape[0]:
            raise ValueError("dense cache batch dimension must match q")

    # Deep validation reads cache_seqlens / block_table back to the host, which
    # costs a device synchronisation on every call. Inside vLLM this runs once
    # per layer per decode step, so it dominates the end-to-end cost. Cheap
    # shape/device/dtype checks above always run; the host-syncing checks are
    # gated and enabled by the test suite via MXFLASHATTN_STRICT_VALIDATION=1.
    if os.getenv("MXFLASHATTN_STRICT_VALIDATION", "0") == "1":
        lengths = [int(length) for length in cache_seqlens.detach().cpu().tolist()]
        if any(length < 0 for length in lengths):
            raise ValueError("cache sequence lengths must be non-negative")
        if any(length + (0 if k is None else k.shape[1]) == 0 for length in lengths):
            raise ValueError("attention requires at least one cached or appended key/value token")
        if block_table is None:
            capacity = k_cache.shape[1]
            if any(length + (0 if k is None else k.shape[1]) > capacity for length in lengths):
                raise ValueError("cache sequence length exceeds dense cache capacity")
        else:
            rows = block_table.detach().cpu().tolist()
            max_blocks = block_table.shape[1]
            pages = k_cache.shape[0]
            for length, row in zip(lengths, rows):
                used = (length + (0 if k is None else k.shape[1]) + k_cache.shape[1] - 1) // k_cache.shape[1]
                if used > max_blocks:
                    raise ValueError("block_table does not contain enough pages for cache sequence")
                if any(int(page) < 0 or int(page) >= pages for page in row[:used]):
                    raise ValueError("block_table contains an out-of-range page index")

    return dispatch(
        "flash_attn_with_kvcache",
        q,
        (
            q,
            k_cache,
            v_cache,
            k,
            v,
            cache_seqlens,
            block_table,
            _resolved_scale(q.shape[-1], softmax_scale),
            causal,
        ),
        lambda: reference.attention_kvcache(
            q,
            k_cache,
            v_cache,
            cache_seqlens=cache_seqlens,
            block_table=block_table,
            k=k,
            v=v,
            softmax_scale=softmax_scale,
            causal=causal,
        ),
        native_args=(q, k_cache, v_cache),
        native_kwargs={
            "k": k,
            "v": v,
            "cache_seqlens": cache_seqlens,
            "block_table": block_table,
            "softmax_scale": softmax_scale,
            "causal": causal,
            "window_size": window_size,
            "softcap": softcap,
            "alibi_slopes": alibi_slopes,
        },
    )
