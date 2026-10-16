import pytest

pytest.importorskip("vllm")
import integrations.vllm.vllm_plugin as vllm_plugin


def test_official_split_retries_vendor_after_invalid_argument(monkeypatch):
    calls = []
    events = []
    expected = object()

    def fake_forward(*args, **kwargs):
        calls.append(1)
        if len(calls) == 1:
            raise RuntimeError("CUDA error: invalid argument")
        return expected

    monkeypatch.setattr(vllm_plugin.FlashAttentionImpl, "forward", fake_forward)
    monkeypatch.setattr(
        vllm_plugin,
        "_record_worker_event",
        lambda event: events.append(event),
    )

    impl = object.__new__(vllm_plugin.MXFlashAttnImpl)
    result = vllm_plugin._forward_official_split_with_mx(
        impl, None, None, None, None, None, None, None, None, None
    )

    assert result is expected
    assert len(calls) == 2
    assert events == [{
        "path": "vendor_direct",
        "operation": "official_split_invalid_argument_fallback",
        "fallback": "true",
        "fallback_reason": "official_split_invalid_argument",
    }]
