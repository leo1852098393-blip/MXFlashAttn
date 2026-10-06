import pytest
import torch

import mxflashattn.dispatch as dispatch
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
