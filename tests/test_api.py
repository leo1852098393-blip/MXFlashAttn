import pytest
import torch

import mxflashattn.dispatch as dispatch
from mxflashattn import (
    BackendUnavailableError,
    flash_attn_func,
    flash_attn_varlen_func,
    flash_attn_with_kvcache,
    get_last_dispatch_info,
)
from mxflashattn import reference

pytestmark = pytest.mark.filterwarnings("ignore:MXFlashAttn is using the PyTorch reference path:RuntimeWarning")


def tensors(batch=1, q_len=2, kv_len=3, q_heads=2, kv_heads=1, head_dim=64):
    q = torch.randn(batch, q_len, q_heads, head_dim, dtype=torch.float32)
    k = torch.randn(batch, kv_len, kv_heads, head_dim, dtype=torch.float32)
    v = torch.randn_like(k)
    return q, k, v


def test_native_backend_is_required_unless_fallback_is_explicit(monkeypatch):
    q, k, v = tensors()
    monkeypatch.delenv("MXFLASHATTN_ALLOW_FALLBACK", raising=False)
    # This is a CPU fallback contract test; do not depend on a locally
    # installed MetaX wheel making the call succeed.
    monkeypatch.setattr(dispatch, "_flash_attn_function", lambda operation: None)
    monkeypatch.setattr(dispatch, "_native_extension", lambda: None)

    with pytest.raises(BackendUnavailableError, match="MXFLASHATTN_ALLOW_FALLBACK=1"):
        flash_attn_func(q, k, v)


def test_explicit_reference_fallback_runs_and_records_reason(monkeypatch):
    q, k, v = tensors()
    monkeypatch.setenv("MXFLASHATTN_ALLOW_FALLBACK", "1")
    monkeypatch.setattr(dispatch, "_flash_attn_function", lambda operation: None)
    monkeypatch.setattr(dispatch, "_native_extension", lambda: None)

    with pytest.warns(RuntimeWarning, match="native_backend_requires_accelerator_tensor"):
        output = flash_attn_func(q, k, v, causal=True)

    assert output.shape == q.shape
    assert torch.isfinite(output).all()
    info = get_last_dispatch_info()
    assert info.backend == "reference"
    assert info.fallback_reason


def test_causal_attention_handles_query_longer_than_key_without_nan(monkeypatch):
    monkeypatch.setenv("MXFLASHATTN_ALLOW_FALLBACK", "1")
    q = torch.randn(1, 3, 2, 64)
    k = torch.randn(1, 1, 1, 64)
    v = torch.randn_like(k)

    output = flash_attn_func(q, k, v, causal=True)

    assert torch.isfinite(output).all()
    torch.testing.assert_close(output[:, :2], torch.zeros_like(output[:, :2]))
    torch.testing.assert_close(output[:, 2:], v.expand(-1, -1, 2, -1))


def test_varlen_attention_supports_gqa_and_unequal_sequences(monkeypatch):
    monkeypatch.setenv("MXFLASHATTN_ALLOW_FALLBACK", "1")
    q = torch.randn(3, 2, 64)
    k = torch.randn(5, 1, 64)
    v = torch.randn_like(k)
    cu_q = torch.tensor([0, 2, 3], dtype=torch.int32)
    cu_k = torch.tensor([0, 3, 5], dtype=torch.int32)

    output = flash_attn_varlen_func(
        q,
        k,
        v,
        cu_q,
        cu_k,
        max_seqlen_q=2,
        max_seqlen_k=3,
    )

    assert output.shape == q.shape
    assert torch.isfinite(output).all()


def test_varlen_rejects_cumulative_lengths_that_do_not_cover_tokens(monkeypatch):
    monkeypatch.setenv("MXFLASHATTN_ALLOW_FALLBACK", "1")
    q = torch.randn(2, 1, 64)
    k = torch.randn(2, 1, 64)
    cu = torch.tensor([0, 1], dtype=torch.int32)

    with pytest.raises(ValueError, match="cover all tokens"):
        flash_attn_varlen_func(q, k, k, cu, cu, max_seqlen_q=1, max_seqlen_k=1)


def test_varlen_rejects_empty_key_sequences(monkeypatch):
    monkeypatch.setenv("MXFLASHATTN_ALLOW_FALLBACK", "1")
    q = torch.randn(2, 1, 64)
    k = torch.randn(1, 1, 64)
    cu_q = torch.tensor([0, 1, 2], dtype=torch.int32)
    cu_k = torch.tensor([0, 0, 1], dtype=torch.int32)

    with pytest.raises(ValueError, match="key/value sequences must be non-empty"):
        flash_attn_varlen_func(q, k, k, cu_q, cu_k, max_seqlen_q=1, max_seqlen_k=1)


def test_varlen_rejects_cumulative_lengths_on_a_different_device(monkeypatch):
    monkeypatch.setenv("MXFLASHATTN_ALLOW_FALLBACK", "1")
    q = torch.randn(1, 1, 64)
    k = torch.randn(1, 1, 64)
    cu = torch.tensor([0, 1], device="meta", dtype=torch.int32)

    with pytest.raises(ValueError, match="same device as q"):
        flash_attn_varlen_func(q, k, k, cu, cu, max_seqlen_q=1, max_seqlen_k=1)


def test_kvcache_rejects_zero_page_size(monkeypatch):
    monkeypatch.setenv("MXFLASHATTN_ALLOW_FALLBACK", "1")
    q = torch.randn(1, 1, 1, 64)
    k_cache = torch.empty(1, 0, 1, 64)
    v_cache = torch.empty_like(k_cache)
    cache_seqlens = torch.tensor([1], dtype=torch.int32)
    block_table = torch.tensor([[0]], dtype=torch.int32)

    with pytest.raises(ValueError, match="page size must be positive"):
        flash_attn_with_kvcache(
            q,
            k_cache,
            v_cache,
            cache_seqlens=cache_seqlens,
            block_table=block_table,
        )


def test_paged_kv_cache_reads_blocks_in_table_order(monkeypatch):
    monkeypatch.setenv("MXFLASHATTN_ALLOW_FALLBACK", "1")
    q = torch.randn(1, 1, 2, 64)
    k_cache = torch.randn(2, 2, 1, 64)
    v_cache = torch.randn_like(k_cache)
    block_table = torch.tensor([[1, 0]], dtype=torch.int32)
    cache_seqlens = torch.tensor([3], dtype=torch.int32)

    output = flash_attn_with_kvcache(
        q,
        k_cache,
        v_cache,
        cache_seqlens=cache_seqlens,
        block_table=block_table,
    )

    expected_k = torch.cat((k_cache[1], k_cache[0, :1]), dim=0).unsqueeze(0)
    expected_v = torch.cat((v_cache[1], v_cache[0, :1]), dim=0).unsqueeze(0)
    expected = flash_attn_func(q, expected_k, expected_v)
    torch.testing.assert_close(output, expected)


def test_paged_kv_cache_appends_and_advances_sequence_length(monkeypatch):
    monkeypatch.setenv("MXFLASHATTN_ALLOW_FALLBACK", "1")
    q, k_new, v_new = tensors(q_len=1, kv_len=1)
    k_cache = torch.zeros(2, 2, 1, 64)
    v_cache = torch.zeros_like(k_cache)
    k_cache[0, :, :, :] = torch.randn_like(k_cache[0])
    v_cache[0, :, :, :] = torch.randn_like(v_cache[0])
    block_table = torch.tensor([[0, 1]], dtype=torch.int32)
    cache_seqlens = torch.tensor([2], dtype=torch.int32)

    output = flash_attn_with_kvcache(
        q,
        k_cache,
        v_cache,
        k=k_new,
        v=v_new,
        cache_seqlens=cache_seqlens,
        block_table=block_table,
    )

    assert output.shape == q.shape
    assert cache_seqlens.tolist() == [3]
    torch.testing.assert_close(k_cache[1, 0], k_new[0, 0])
    torch.testing.assert_close(v_cache[1, 0], v_new[0, 0])


def test_dense_kv_cache_appends_and_advances_sequence_length(monkeypatch):
    monkeypatch.setenv("MXFLASHATTN_ALLOW_FALLBACK", "1")
    q, k_new, v_new = tensors(q_len=1, kv_len=1)
    k_cache = torch.zeros(1, 4, 1, 64)
    v_cache = torch.zeros_like(k_cache)
    k_cache[:, :2] = torch.randn(1, 2, 1, 64)
    v_cache[:, :2] = torch.randn(1, 2, 1, 64)
    cache_seqlens = torch.tensor([2], dtype=torch.int32)

    flash_attn_with_kvcache(q, k_cache, v_cache, k=k_new, v=v_new, cache_seqlens=cache_seqlens)

    assert cache_seqlens.tolist() == [3]
    torch.testing.assert_close(k_cache[0, 2], k_new[0, 0])
    torch.testing.assert_close(v_cache[0, 2], v_new[0, 0])


def test_unsupported_flash_attention_options_fail_clearly(monkeypatch):
    monkeypatch.setenv("MXFLASHATTN_ALLOW_FALLBACK", "1")
    q, k, v = tensors()

    with pytest.raises(NotImplementedError, match="dropout"):
        flash_attn_func(q, k, v, dropout_p=0.1)


def test_explicit_zero_scale_is_preserved_and_nonfinite_scale_is_rejected(monkeypatch):
    monkeypatch.setenv("MXFLASHATTN_ALLOW_FALLBACK", "1")
    q, k, v = tensors()

    zero_scale = flash_attn_func(q, k, v, softmax_scale=0.0)
    expected = reference.attention_batched(q, k, v, softmax_scale=0.0)
    default_scale = flash_attn_func(q, k, v)

    torch.testing.assert_close(zero_scale, expected)
    assert not torch.allclose(zero_scale, default_scale)
    with pytest.raises(ValueError, match="softmax_scale must be finite"):
        flash_attn_func(q, k, v, softmax_scale=float("nan"))
