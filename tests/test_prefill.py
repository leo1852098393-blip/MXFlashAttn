import pytest
import torch

from mxflashattn import reference
from mxflashattn.prefill import prefill_chunked_reference, prefill_chunked_reference_fallback


def tensors(batch=2, q_len=5, kv_len=7, q_heads=4, kv_heads=2, head_dim=64):
    torch.manual_seed(20261016)
    q = torch.randn(batch, q_len, q_heads, head_dim, dtype=torch.float32)
    k = torch.randn(batch, kv_len, kv_heads, head_dim, dtype=torch.float32)
    v = torch.randn_like(k)
    return q, k, v


@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("chunk_size", [1, 2, 4, 16])
def test_chunked_prefill_matches_reference_for_gqa(causal, chunk_size):
    q, k, v = tensors()
    actual = prefill_chunked_reference(q, k, v, chunk_size=chunk_size, causal=causal)
    expected = reference.attention_batched(q, k, v, causal=causal)
    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-5)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_chunked_prefill_preserves_input_dtype_and_tolerance(dtype):
    q, k, v = tensors(q_len=4, kv_len=4, q_heads=2, kv_heads=1)
    q, k, v = q.to(dtype), k.to(dtype), v.to(dtype)
    actual = prefill_chunked_reference(q, k, v, chunk_size=2, causal=True)
    expected = reference.attention_batched(q, k, v, causal=True)
    assert actual.dtype == dtype
    torch.testing.assert_close(actual, expected, rtol=3e-3, atol=3e-3)


def test_chunked_prefill_handles_query_longer_than_key_without_nan():
    q, k, v = tensors(q_len=5, kv_len=2, q_heads=2, kv_heads=1)
    actual = prefill_chunked_reference(q, k, v, chunk_size=1, causal=True)
    expected = reference.attention_batched(q, k, v, causal=True)
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-5)


def test_prefill_fallback_wrapper_uses_reference_for_one_chunk():
    q, k, v = tensors(q_len=3, kv_len=5, q_heads=2, kv_heads=1)
    actual = prefill_chunked_reference_fallback(q, k, v, chunk_size=8)
    expected = reference.attention_batched(q, k, v)
    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("chunk_size", [0, -1, 1.5])
def test_chunked_prefill_rejects_invalid_chunk_size(chunk_size):
    q, k, v = tensors(q_len=2, kv_len=2, q_heads=2, kv_heads=1)
    with pytest.raises(ValueError, match="chunk_size"):
        prefill_chunked_reference(q, k, v, chunk_size=chunk_size)
