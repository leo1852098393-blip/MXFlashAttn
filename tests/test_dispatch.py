from types import SimpleNamespace

import mxflashattn.dispatch as dispatch
import pytest
import torch


@pytest.fixture(autouse=True)
def clear_provider_cache():
    dispatch._metax_flash_attn_module.cache_clear()
    yield
    dispatch._metax_flash_attn_module.cache_clear()


def test_flash_attn_provider_requires_metax_build(monkeypatch):
    fake_module = SimpleNamespace(flash_attn_func=lambda *args, **kwargs: None)
    monkeypatch.setenv("MXFLASHATTN_BACKEND", "auto")
    monkeypatch.setattr(dispatch.importlib, "import_module", lambda name: fake_module)
    monkeypatch.setattr(dispatch.importlib.metadata, "version", lambda name: "2.6.3")

    assert dispatch._flash_attn_function("flash_attn_func") is None


def test_flash_attn_provider_accepts_metax_build(monkeypatch):
    function = lambda *args, **kwargs: None
    fake_module = SimpleNamespace(flash_attn_func=function)
    versions = []
    monkeypatch.setenv("MXFLASHATTN_BACKEND", "auto")
    monkeypatch.setattr(dispatch.importlib, "import_module", lambda name: fake_module)
    monkeypatch.setattr(
        dispatch.importlib.metadata,
        "version",
        lambda name: versions.append(name) or "2.6.3+metax3.5.3.9torch2.8",
    )

    assert dispatch._flash_attn_function("flash_attn_func") is function
    assert dispatch._flash_attn_function("flash_attn_func") is function
    assert versions == ["flash-attn"]


def test_aten_override_skips_external_flash_attn(monkeypatch):
    monkeypatch.setenv("MXFLASHATTN_BACKEND", "aten")
    monkeypatch.setattr(
        dispatch.importlib,
        "import_module",
        lambda name: (_ for _ in ()).throw(AssertionError("must not import flash_attn")),
    )

    assert dispatch._flash_attn_function("flash_attn_func") is None


def test_zero_scale_uses_aten_extension_even_when_metax_provider_exists(monkeypatch):
    calls = []
    provider = lambda *args, **kwargs: pytest.fail("MetaX provider must not receive zero scale")
    extension = SimpleNamespace(
        is_available=lambda: True,
        flash_attn_func=lambda *args: calls.append(args) or torch.tensor([1]),
    )
    tensor = SimpleNamespace(device=SimpleNamespace(type="cuda"), dtype=torch.float16)
    monkeypatch.setattr(dispatch, "_flash_attn_function", lambda operation: provider)
    monkeypatch.setattr(dispatch, "_native_extension", lambda: extension)

    result = dispatch.dispatch(
        "flash_attn_func",
        tensor,
        (tensor,),
        lambda: pytest.fail("fallback must not run"),
        native_args=(tensor,),
        native_kwargs={"softmax_scale": 0.0},
    )

    assert result.item() == 1
    assert calls == [(tensor,)]
    assert dispatch.get_last_dispatch_info().backend == "mxmac_aten_correctness"


def test_zero_scale_requires_aten_extension_if_metax_provider_is_present(monkeypatch):
    tensor = SimpleNamespace(device=SimpleNamespace(type="cuda"), dtype=torch.float16)
    monkeypatch.setattr(dispatch, "_flash_attn_function", lambda operation: lambda *args: None)
    monkeypatch.setattr(dispatch, "_native_extension", lambda: None)

    assert dispatch.native_unavailable_reason(tensor, "flash_attn_func", require_aten=True) == (
        "zero_softmax_scale_requires_aten_extension"
    )

