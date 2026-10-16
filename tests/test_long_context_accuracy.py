import pytest
import torch

from mxflashattn import reference
from mxflashattn.accuracy import summarize_error
from mxflashattn.prefill import prefill_chunked_reference


@pytest.mark.parametrize("kv_len", [128, 512])
def test_long_context_prefill_error_summary_is_stable(kv_len):
    torch.manual_seed(20261016 + kv_len)
    q = torch.randn(1, kv_len, 2, 64)
    k = torch.randn(1, kv_len, 1, 64)
    v = torch.randn_like(k)

    actual = prefill_chunked_reference(q, k, v, chunk_size=64, causal=True)
    expected = reference.attention_batched(q, k, v, causal=True)
    summary = summarize_error(actual, expected)

    assert summary.finite
    assert summary.max_abs < 1e-4
    assert summary.p99_abs < 1e-5
