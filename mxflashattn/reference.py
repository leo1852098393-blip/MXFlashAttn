from __future__ import annotations

import math

import torch


def attention_single(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    softmax_scale: float | None = None,
    causal: bool = False,
) -> torch.Tensor:
    """Readable PyTorch reference for [seq, heads, dim] inputs."""
    if q.shape[0] == 0:
        return q.clone()
    query_len, query_heads, head_dim = q.shape
    key_len, kv_heads, _ = k.shape
    if key_len == 0:
        raise ValueError("attention requires at least one key/value token")

    scale = softmax_scale if softmax_scale is not None else 1.0 / math.sqrt(head_dim)
    groups = query_heads // kv_heads
    qf = q.float().transpose(0, 1)
    kf = k.float().repeat_interleave(groups, dim=1).transpose(0, 1)
    vf = v.float().repeat_interleave(groups, dim=1).transpose(0, 1)
    scores = torch.matmul(qf, kf.transpose(-2, -1)) * scale

    if causal:
        q_positions = torch.arange(query_len, device=q.device) + (key_len - query_len)
        k_positions = torch.arange(key_len, device=q.device)
        blocked = k_positions.unsqueeze(0) > q_positions.unsqueeze(1)
        scores = scores.masked_fill(blocked.unsqueeze(0), float("-inf"))

    probabilities = torch.softmax(scores, dim=-1)
    probabilities = torch.nan_to_num(probabilities, nan=0.0, posinf=0.0, neginf=0.0)
    output = torch.matmul(probabilities, vf).transpose(0, 1)
    return output.to(dtype=q.dtype)


def attention_batched(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    softmax_scale: float | None = None,
    causal: bool = False,
) -> torch.Tensor:
    outputs = [
        attention_single(q[index], k[index], v[index], softmax_scale=softmax_scale, causal=causal)
        for index in range(q.shape[0])
    ]
    if not outputs:
        return q.clone()
    return torch.stack(outputs, dim=0)


def attention_varlen(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: list[int],
    cu_seqlens_k: list[int],
    *,
    softmax_scale: float | None = None,
    causal: bool = False,
) -> torch.Tensor:
    outputs = []
    for index in range(len(cu_seqlens_q) - 1):
        q_start, q_end = cu_seqlens_q[index : index + 2]
        k_start, k_end = cu_seqlens_k[index : index + 2]
        outputs.append(
            attention_single(
                q[q_start:q_end],
                k[k_start:k_end],
                v[k_start:k_end],
                softmax_scale=softmax_scale,
                causal=causal,
            )
        )
    if not outputs:
        return q[:0].clone()
    return torch.cat(outputs, dim=0)


def attention_kvcache(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    *,
    cache_seqlens: torch.Tensor,
    block_table: torch.Tensor | None,
    k: torch.Tensor | None = None,
    v: torch.Tensor | None = None,
    softmax_scale: float | None = None,
    causal: bool = False,
) -> torch.Tensor:
    lengths = [int(length) for length in cache_seqlens.detach().cpu().tolist()]
    batch = q.shape[0]
    outputs = []

    for batch_index in range(batch):
        old_len = lengths[batch_index]
        append_len = 0 if k is None else k.shape[1]
        total_len = old_len + append_len
        if block_table is None:
            if append_len:
                k_cache[batch_index, old_len:total_len] = k[batch_index]
                v_cache[batch_index, old_len:total_len] = v[batch_index]
            current_k = k_cache[batch_index, :total_len]
            current_v = v_cache[batch_index, :total_len]
        else:
            page_size = k_cache.shape[1]
            row = [int(block) for block in block_table[batch_index].detach().cpu().tolist()]
            for token_index in range(append_len):
                logical = old_len + token_index
                page_index, page_offset = divmod(logical, page_size)
                physical_page = row[page_index]
                k_cache[physical_page, page_offset] = k[batch_index, token_index]
                v_cache[physical_page, page_offset] = v[batch_index, token_index]
            page_count = (total_len + page_size - 1) // page_size
            pages = row[:page_count]
            current_k = torch.cat([k_cache[page] for page in pages], dim=0)[:total_len]
            current_v = torch.cat([v_cache[page] for page in pages], dim=0)[:total_len]
        outputs.append(
            attention_single(
                q[batch_index],
                current_k,
                current_v,
                softmax_scale=softmax_scale,
                causal=causal,
            )
        )

    if k is not None:
        cache_seqlens.add_(k.shape[1])
    return torch.stack(outputs, dim=0) if outputs else q.clone()
