"""Small, explicit bridge for vLLM 0.17 paged decoder metadata.

This adapter intentionally supports one-token decode only. It converts the
packed vLLM tensors and the ``[2, blocks, block_size, kv_heads, dim]`` cache
layout into ``flash_attn_with_kvcache``. Prefill, cascade attention, FP8,
and scheduler-specific metadata are rejected until they have dedicated tests.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import torch

from mxflashattn import flash_attn_with_kvcache

# inspect_layout() reads query_start_loc / seq_lens back to the host, which is a
# device synchronisation. Inside vLLM this runs once per layer per decode step
# (28 layers x N steps x prompts), so it dominates end-to-end latency. The cheap
# shape/device/dtype checks always run; the host-syncing inspection is gated and
# enabled by the test suite via MXFLASHATTN_STRICT_LAYOUT=1.
STRICT_LAYOUT = os.getenv("MXFLASHATTN_STRICT_LAYOUT", "0") == "1"
def _transparent_enabled() -> bool:
    # Resolve at call time so vLLM workers and tests can configure the mode
    # after importing this module.
    return os.getenv("MXFLASHATTN_TRANSPARENT", "0") == "1"


@dataclass(frozen=True, slots=True)
class DecoderLayout:
    num_actual_tokens: int
    batch_size: int
    query_lens: tuple[int, ...]
    max_query_len: int
    seq_lens: tuple[int, ...]
    query_start_loc: tuple[int, ...]
    block_table_shape: tuple[int, int]
    kv_cache_shape: tuple[int, int, int, int, int]
    scheduler_metadata_present: bool


class DecoderAdapterBlocked(RuntimeError):
    """Raised when vLLM metadata is outside the verified adapter contract."""


def inspect_layout(query: torch.Tensor, kv_cache: torch.Tensor, attn_metadata: Any) -> DecoderLayout:
    """Validate and record the vLLM 0.17 decoder tensor layout."""
    required = (
        "num_actual_tokens", "max_query_len", "query_start_loc", "max_seq_len",
        "seq_lens", "block_table", "scheduler_metadata", "causal", "use_cascade",
    )
    missing = [name for name in required if not hasattr(attn_metadata, name)]
    if missing:
        raise DecoderAdapterBlocked(f"metadata_missing_fields:{','.join(missing)}")
    if query.ndim != 3:
        raise DecoderAdapterBlocked(f"query_rank:{query.ndim},expected:3")
    if kv_cache.ndim != 5 or kv_cache.shape[0] != 2:
        raise DecoderAdapterBlocked(
            f"kv_cache_shape:{tuple(kv_cache.shape)},expected:[2,blocks,block_size,kv_heads,head_dim]"
        )
    starts = tuple(int(x) for x in attn_metadata.query_start_loc.detach().cpu().tolist())
    seq_lens = tuple(int(x) for x in attn_metadata.seq_lens.detach().cpu().tolist())
    if len(starts) != len(seq_lens) + 1 or starts[0] != 0 or starts[-1] != int(attn_metadata.num_actual_tokens):
        raise DecoderAdapterBlocked("query_start_loc_does_not_cover_num_actual_tokens")
    query_lens = tuple(right - left for left, right in zip(starts, starts[1:]))
    block_table = attn_metadata.block_table
    if block_table.ndim != 2 or block_table.shape[0] != len(seq_lens):
        raise DecoderAdapterBlocked("block_table_rows_do_not_match_batch")
    return DecoderLayout(
        num_actual_tokens=int(attn_metadata.num_actual_tokens),
        batch_size=len(seq_lens),
        query_lens=query_lens,
        max_query_len=int(attn_metadata.max_query_len),
        seq_lens=seq_lens,
        query_start_loc=starts,
        block_table_shape=(int(block_table.shape[0]), int(block_table.shape[1])),
        kv_cache_shape=tuple(int(x) for x in kv_cache.shape),
        scheduler_metadata_present=attn_metadata.scheduler_metadata is not None,
    )


def fast_layout(query: torch.Tensor, kv_cache: torch.Tensor, attn_metadata: Any) -> DecoderLayout:
    """Cheap layout capture that performs no device-to-host synchronisation."""
    required = (
        "num_actual_tokens", "max_query_len", "query_start_loc", "max_seq_len",
        "seq_lens", "block_table", "scheduler_metadata", "causal", "use_cascade",
    )
    missing = [name for name in required if not hasattr(attn_metadata, name)]
    if missing:
        raise DecoderAdapterBlocked(f"metadata_missing_fields:{','.join(missing)}")
    if query.ndim != 3:
        raise DecoderAdapterBlocked(f"query_rank:{query.ndim},expected:3")
    if kv_cache.ndim != 5 or kv_cache.shape[0] != 2:
        raise DecoderAdapterBlocked(
            f"kv_cache_shape:{tuple(kv_cache.shape)},expected:[2,blocks,block_size,kv_heads,head_dim]"
        )
    block_table = attn_metadata.block_table
    if block_table.ndim != 2:
        raise DecoderAdapterBlocked("block_table_rows_do_not_match_batch")
    batch = int(block_table.shape[0])
    max_query_len = int(attn_metadata.max_query_len)
    return DecoderLayout(
        num_actual_tokens=int(attn_metadata.num_actual_tokens),
        batch_size=batch,
        query_lens=tuple([max_query_len] * batch),
        max_query_len=max_query_len,
        seq_lens=(),
        query_start_loc=(),
        block_table_shape=(batch, int(block_table.shape[1])),
        kv_cache_shape=tuple(int(x) for x in kv_cache.shape),
        scheduler_metadata_present=attn_metadata.scheduler_metadata is not None,
    )


def decode_forward(
    query: torch.Tensor,
    kv_cache: torch.Tensor,
    attn_metadata: Any,
    *,
    softmax_scale: float,
    causal: bool = True,
) -> tuple[torch.Tensor, DecoderLayout]:
    """Run the verified one-token paged decode path through MXFlashAttn.

    vLLM updates the cache before calling attention. Consequently ``k`` and
    ``v`` are omitted here and ``seq_lens`` describes the already populated
    cache. This is stricter than vLLM's general FlashAttention implementation
    so an unsupported prefill path cannot silently delegate.
    """
    if os.getenv("MXFLASHATTN_GRAPH_DECODE", "0") == "1":
        # Graph-safe path is decoder-only. Prefill may have a static padded
        # query shape, but its metadata is not one cache sequence per query
        # row; sending it to the paged KV decoder corrupts outputs.
        # In full CUDA-graph mode vLLM pads q to a capture size while keeping
        # one metadata row per logical request. A singleton seq_lens/block row
        # therefore identifies a padded one-token decode, even when
        # max_query_len contains the capture/prefill value (for example 15).
        has_padded_decode = (
            attn_metadata.seq_lens.ndim == 1
            and attn_metadata.block_table.ndim == 2
            and attn_metadata.seq_lens.numel() == attn_metadata.block_table.shape[0]
            and attn_metadata.seq_lens.numel() < query.shape[0]
        )
        has_split_decode = (
            int(getattr(attn_metadata, "num_decode_tokens", 0) or 0) > 0
            and int(getattr(attn_metadata, "num_decodes", 0) or 0) > 0
            and getattr(attn_metadata, "decode_seq_lens", None) is not None
            and getattr(attn_metadata, "decode_block_table", None) is not None
        )
        if int(getattr(attn_metadata, "max_query_len", 0)) != 1 and not has_padded_decode and not has_split_decode:
            raise DecoderAdapterBlocked(
                f"only_single_token_decode_supported:max_query_len={int(getattr(attn_metadata, 'max_query_len', 0))}"
            )
        return _decode_forward_graph(query, kv_cache, attn_metadata,
                                     softmax_scale=softmax_scale, causal=causal)
    if _transparent_enabled():
        # Transparent mode still only owns the verified one-token decoder
        # contract. Prefill has one seq_lens entry but many query tokens, so
        # forwarding it to the paged KV API produces a batch-size mismatch;
        # let vLLM's native implementation handle that path.
        if int(attn_metadata.max_query_len) != 1:
            raise DecoderAdapterBlocked(
                f"only_single_token_decode_supported:max_query_len={int(attn_metadata.max_query_len)}"
            )
        key_cache, value_cache = kv_cache.unbind(0)
        n = int(attn_metadata.num_actual_tokens)
        q = query[:n].reshape(n, 1, query.shape[1], query.shape[2])
        # vLLM capture/replay metadata may expose seq_lens as a singleton
        # leading-dimension view (for example [1, batch]).  The public
        # flash-attn contract is strictly one-dimensional, so normalize the
        # view without reading it back to the host.
        seq_lens = attn_metadata.seq_lens.reshape(-1)
        cache_seqlens = (
            seq_lens
            if (seq_lens.device == query.device and seq_lens.dtype == torch.int32)
            else seq_lens.to(device=query.device, dtype=torch.int32)
        )
        bt_raw = attn_metadata.block_table
        block_table = bt_raw if (bt_raw.device == query.device and bt_raw.dtype == torch.int32) else bt_raw.to(device=query.device, dtype=torch.int32)
        result = flash_attn_with_kvcache(q, key_cache, value_cache, cache_seqlens=cache_seqlens, block_table=block_table, softmax_scale=softmax_scale, causal=causal)
        return result.reshape(n, -1), None
    layout = inspect_layout(query, kv_cache, attn_metadata) if STRICT_LAYOUT else fast_layout(query, kv_cache, attn_metadata)
    if not bool(attn_metadata.causal) and causal:
        raise DecoderAdapterBlocked("metadata_causal_flag_is_false")
    if bool(attn_metadata.use_cascade):
        raise DecoderAdapterBlocked("cascade_attention_not_supported_by_adapter")
    if layout.max_query_len != 1:
        import os as _os
        if _os.getenv("MXFLASHATTN_GRAPH_PROBE") == "1":
            try:
                with open("/tmp/graph_probe.log", "a") as _lf:
                    _lf.write(
                        f"max_query_len={layout.max_query_len} "
                        f"query_shape={tuple(query.shape)} "
                        f"n_actual={int(attn_metadata.num_actual_tokens)} "
                        f"meta_type={type(attn_metadata).__name__} "
                        f"has_capture_meta={hasattr(attn_metadata,'attention_metadata_for_capture')} "
                        f"q_start_loc_present={hasattr(attn_metadata,'query_start_loc')} "
                        f"seq_lens_shape={tuple(getattr(attn_metadata,'seq_lens',__import__('torch').empty(0)).shape) if hasattr(attn_metadata,'seq_lens') else 'NA'}\n"
                    )
            except Exception:
                pass
        raise DecoderAdapterBlocked(
            f"only_single_token_decode_supported:max_query_len={layout.max_query_len}"
        )
    if kv_cache.dtype not in (torch.float16, torch.bfloat16):
        raise DecoderAdapterBlocked(f"kv_cache_dtype_not_supported:{kv_cache.dtype}")
    if query.shape[1] % kv_cache.shape[3]:
        raise DecoderAdapterBlocked("query_and_kv_head_counts_are_not_gqa_compatible")
    key_cache, value_cache = kv_cache.unbind(0)
    batch = layout.batch_size
    q = query[: layout.num_actual_tokens].reshape(batch, 1, query.shape[1], query.shape[2])
    # vLLM 0.17 already keeps these on the device in int32. Calling .to()
    # unconditionally costs a Python/dispatch round trip per layer per step, and
    # .clone() cost even more. Only convert when the contract is actually broken.
    seq_lens = attn_metadata.seq_lens
    cache_seqlens = (
        seq_lens if seq_lens.device == query.device and seq_lens.dtype == torch.int32
        else seq_lens.to(device=query.device, dtype=torch.int32)
    )
    block_table_raw = attn_metadata.block_table
    block_table = (
        block_table_raw
        if block_table_raw.device == query.device and block_table_raw.dtype == torch.int32
        else block_table_raw.to(device=query.device, dtype=torch.int32)
    )
    result = flash_attn_with_kvcache(
        q, key_cache, value_cache, cache_seqlens=cache_seqlens,
        block_table=block_table, softmax_scale=softmax_scale, causal=causal,
    )
    return result.reshape(layout.num_actual_tokens, -1), layout


def _decode_forward_graph(
    query: torch.Tensor,
    kv_cache: torch.Tensor,
    attn_metadata: Any,
    *,
    softmax_scale: float,
    causal: bool = True,
):
    """Run the static-shape graph decode using MetaX's split decode metadata.

    vLLM/MetaX keeps padded prefill metadata in ``seq_lens``/``block_table``
    and exposes the actual decode subset separately.  Using the former and
    broadcasting it to ``query.shape[0]`` makes a graph-shaped call execute
    with the wrong KV pages.
    """
    key_cache, value_cache = kv_cache.unbind(0)
    decode_n = int(getattr(attn_metadata, "num_decode_tokens", 0) or 0)
    decode_batch = int(getattr(attn_metadata, "num_decodes", 0) or 0)
    decode_seq = getattr(attn_metadata, "decode_seq_lens", None)
    decode_bt = getattr(attn_metadata, "decode_block_table", None)
    if decode_n > 0 and decode_batch > 0 and decode_seq is not None and decode_bt is not None:
        q_tokens = query[:decode_n]
        if decode_n % decode_batch:
            raise DecoderAdapterBlocked("decode_tokens_not_divisible_by_num_decodes")
        q = q_tokens.reshape(decode_batch, decode_n // decode_batch, query.shape[1], query.shape[2]).contiguous()
        cache_seqlens = decode_seq.reshape(-1)
        block_table = decode_bt
        batch = decode_batch
    else:
        raw_seq = attn_metadata.seq_lens.reshape(-1)
        raw_bt = attn_metadata.block_table
        # Capture padding belongs to the graph buffer, not to the logical
        # attention batch. Use one q row per metadata row and leave padded q
        # rows out of the vendor call.
        batch = int(raw_seq.numel()) if raw_seq.numel() < query.shape[0] else query.shape[0]
        q = query[:batch].reshape(batch, 1, query.shape[1], query.shape[2]).contiguous()
        cache_seqlens = raw_seq
        block_table = raw_bt
    if cache_seqlens.device != query.device or cache_seqlens.dtype != torch.int32:
        cache_seqlens = cache_seqlens.to(device=query.device, dtype=torch.int32)
    if block_table.device != query.device or block_table.dtype != torch.int32:
        block_table = block_table.to(device=query.device, dtype=torch.int32)
    if cache_seqlens.numel() != batch or block_table.shape[0] != batch:
        raise DecoderAdapterBlocked("decode_metadata_batch_mismatch")
    cache_seqlens = cache_seqlens.contiguous()
    block_table = block_table.contiguous()
    result = flash_attn_with_kvcache(
        q, key_cache, value_cache, cache_seqlens=cache_seqlens,
        block_table=block_table, softmax_scale=softmax_scale, causal=causal,
    )
    return result.reshape(q.shape[0] * q.shape[1], -1), None
