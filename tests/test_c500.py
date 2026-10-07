from __future__ import annotations

import pytest
import torch

from mxflashattn import (
    flash_attn_func,
    flash_attn_varlen_func,
    flash_attn_with_kvcache,
    get_last_dispatch_info,
)
from mxflashattn import reference
from mxflashattn.dispatch import native_unavailable_reason


def c500_device() -> torch.device:
    for device_type in ("maca", "cuda"):
        module = getattr(torch, device_type, None)
        if module is None or not module.is_available():
            continue
        if "C500" not in module.get_device_name(0):
            continue
        device = torch.device(device_type)
        probe = torch.empty(1, dtype=torch.float16, device=device)
        if native_unavailable_reason(probe, "flash_attn_func") is None:
            return device
        pytest.skip("no MetaX FlashAttention wheel or MXFlashAttn extension is available")
    pytest.skip("this test requires a MetaX C500 accelerator")


def assert_native_dispatch() -> None:
    info = get_last_dispatch_info()
    assert info is not None
    assert info.backend in {"vendor_direct", "mxmac_aten_correctness", "mxmac_flash_attn", "mxmac_aten_extension"}


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_c500_batched_gqa_causal_matches_reference(dtype: torch.dtype) -> None:
    device = c500_device()
    torch.manual_seed(21)
    q = torch.randn((2, 4, 8, 64), device=device, dtype=dtype)
    k = torch.randn((2, 7, 2, 64), device=device, dtype=dtype)
    v = torch.randn_like(k)

    actual = flash_attn_func(q, k, v, causal=True)
    assert_native_dispatch()
    expected = reference.attention_batched(q, k, v, causal=True)

    torch.testing.assert_close(actual, expected, rtol=0.04, atol=0.04)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_c500_native_backend_preserves_zero_scale(dtype: torch.dtype) -> None:
    device = c500_device()
    torch.manual_seed(24)
    q = torch.randn((1, 2, 4, 64), device=device, dtype=dtype)
    k = torch.randn((1, 5, 2, 64), device=device, dtype=dtype)
    v = torch.randn_like(k)

    actual = flash_attn_func(q, k, v, softmax_scale=0.0)
    assert_native_dispatch()
    expected = reference.attention_batched(q, k, v, softmax_scale=0.0)

    torch.testing.assert_close(actual, expected, rtol=0.04, atol=0.04)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_c500_varlen_gqa_matches_reference(dtype: torch.dtype) -> None:
    device = c500_device()
    torch.manual_seed(22)
    q = torch.randn((5, 4, 64), device=device, dtype=dtype)
    k = torch.randn((8, 2, 64), device=device, dtype=dtype)
    v = torch.randn_like(k)
    cu_q = torch.tensor([0, 2, 5], device=device, dtype=torch.int32)
    cu_k = torch.tensor([0, 3, 8], device=device, dtype=torch.int32)

    actual = flash_attn_varlen_func(q, k, v, cu_q, cu_k, max_seqlen_q=3, max_seqlen_k=5)
    assert_native_dispatch()
    expected = reference.attention_varlen(q, k, v, [0, 2, 5], [0, 3, 8])

    torch.testing.assert_close(actual, expected, rtol=0.04, atol=0.04)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_c500_paged_kv_append_matches_reference(dtype: torch.dtype) -> None:
    device = c500_device()
    torch.manual_seed(23)
    q = torch.randn((1, 1, 4, 64), device=device, dtype=dtype)
    k_cache = torch.randn((3, 4, 2, 64), device=device, dtype=dtype)
    v_cache = torch.randn_like(k_cache)
    new_k = torch.randn((1, 1, 2, 64), device=device, dtype=dtype)
    new_v = torch.randn_like(new_k)
    block_table = torch.tensor([[2, 0]], device=device, dtype=torch.int32)
    cache_seqlens = torch.tensor([5], device=device, dtype=torch.int32)

    actual_k_cache = k_cache.clone()
    actual_v_cache = v_cache.clone()
    actual_seqlens = cache_seqlens.clone()
    actual = flash_attn_with_kvcache(
        q,
        actual_k_cache,
        actual_v_cache,
        k=new_k,
        v=new_v,
        cache_seqlens=actual_seqlens,
        block_table=block_table,
    )
    assert_native_dispatch()

    expected_k_cache = k_cache.clone()
    expected_v_cache = v_cache.clone()
    expected_seqlens = cache_seqlens.clone()
    expected = reference.attention_kvcache(
        q,
        expected_k_cache,
        expected_v_cache,
        cache_seqlens=expected_seqlens,
        block_table=block_table,
        k=new_k,
        v=new_v,
    )

    torch.testing.assert_close(actual, expected, rtol=0.04, atol=0.04)
    torch.testing.assert_close(actual_k_cache, expected_k_cache)
    torch.testing.assert_close(actual_v_cache, expected_v_cache)
    assert actual_seqlens.tolist() == expected_seqlens.tolist() == [6]

