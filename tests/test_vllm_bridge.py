import pytest
import torch

import mxflashattn.dispatch as dispatch
pytest.importorskip("vllm")
import integrations.vllm.vllm_plugin as vllm_plugin
from integrations.vllm.backend import attention_forward
from integrations.vllm.config import VLLM_TARGET_VERSION, validate_vllm_version


def test_vllm_target_version_is_pinned():
    validate_vllm_version(VLLM_TARGET_VERSION)
    with pytest.raises(RuntimeError, match="targets vLLM"):
        validate_vllm_version("0.29.0")


def test_dense_vllm_bridge_delegates_to_public_api(monkeypatch):
    monkeypatch.setenv("MXFLASHATTN_ALLOW_FALLBACK", "1")
    monkeypatch.setattr(dispatch, "_flash_attn_function", lambda operation: None)
    monkeypatch.setattr(dispatch, "_native_extension", lambda: None)
    q = torch.randn(3, 2, 64)
    k = torch.randn(4, 1, 64)
    v = torch.randn_like(k)

    with pytest.warns(RuntimeWarning, match="native_backend_requires_accelerator_tensor"):
        output = attention_forward(q, k, v, causal=True, scale=0.125)

    assert output.shape == q.shape
    assert torch.isfinite(output).all()


@pytest.mark.parametrize(
    "head_dim,q_heads,kv_heads",
    [(256, 16, 4), (128, 24, 8)],
)
def test_official_kvcache_compat_falls_back_to_vendor_for_unsupported_shapes(
    monkeypatch, head_dim, q_heads, kv_heads
):
    calls = []
    expected = object()
    args = (torch.empty(1, 1, q_heads, head_dim),)
    kwargs = {
        "k_cache": torch.empty(1, 16, kv_heads, head_dim),
        "v_cache": torch.empty(1, 16, kv_heads, head_dim),
        "block_table": torch.zeros(1, 1, dtype=torch.int32),
        "cache_seqlens": torch.ones(1, dtype=torch.int32),
    }
    monkeypatch.setenv("MXFA_NATIVE_DECODE", "1")
    monkeypatch.setattr(vllm_plugin, "_try_native_decode", lambda *a, **k: None)
    monkeypatch.setattr(
        vllm_plugin,
        "_official_kvcache",
        lambda *a, **k: calls.append((a, k)) or expected,
    )
    monkeypatch.setattr(
        vllm_plugin,
        "flash_attn_with_kvcache",
        lambda *a, **k: pytest.fail("unsupported shape must not enter the MX API"),
    )

    assert vllm_plugin._mx_kvcache_compat(*args, **kwargs) is expected
    assert calls == [(args, kwargs)]
