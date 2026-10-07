from __future__ import annotations

import importlib
import importlib.metadata
import os
import threading
import warnings
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Callable

import torch


class BackendUnavailableError(RuntimeError):
    """Raised when native execution is unavailable and fallback was not enabled."""


@dataclass(frozen=True)
class DispatchInfo:
    backend: str
    operation: str
    fallback_reason: str | None = None


_state = threading.local()


def get_last_dispatch_info() -> DispatchInfo | None:
    return getattr(_state, "dispatch_info", None)


def _native_extension() -> Any | None:
    try:
        return importlib.import_module("mxflashattn._C")
    except (ImportError, OSError):
        return None


@lru_cache(maxsize=1)
def _metax_flash_attn_module() -> Any | None:
    try:
        module = importlib.import_module("flash_attn")
    except (ImportError, OSError):
        return None
    try:
        version = importlib.metadata.version("flash-attn")
    except importlib.metadata.PackageNotFoundError:
        version = str(getattr(module, "__version__", ""))
    if "metax" not in version.casefold():
        return None
    return module


def _flash_attn_function(operation: str) -> Callable[..., torch.Tensor] | None:
    if os.getenv("MXFLASHATTN_BACKEND", "auto").strip().casefold() == "aten":
        return None
    module = _metax_flash_attn_module()
    if module is None:
        return None
    function = getattr(module, operation, None)
    return function if callable(function) else None


def native_unavailable_reason(
    tensor: torch.Tensor,
    operation: str,
    *,
    require_aten: bool = False,
) -> str | None:
    if tensor.device.type not in {"cuda", "maca", "musa"}:
        return "native_backend_requires_accelerator_tensor"
    if tensor.dtype not in (torch.float16, torch.bfloat16):
        return "native_backend_supports_only_float16_and_bfloat16"
    if _flash_attn_function(operation) is not None and not require_aten:
        return None
    extension = _native_extension()
    if extension is None:
        if require_aten:
            return "zero_softmax_scale_requires_aten_extension"
        return "native_extension_not_built_or_not_importable"
    if not bool(extension.is_available()):
        return "native_extension_reports_unavailable_runtime"
    if not hasattr(extension, operation):
        return f"native_extension_does_not_implement_{operation}"
    return None


def dispatch(
    operation: str,
    tensor: torch.Tensor,
    extension_args: tuple[Any, ...],
    reference: Callable[[], torch.Tensor],
    *,
    native_args: tuple[Any, ...],
    native_kwargs: dict[str, Any],
) -> torch.Tensor:
    """Dispatch vendor-direct first, then ATen correctness, then opt-in reference.

    The vendor wheel is the performance path. Any vendor import, signature, or
    runtime failure is recorded and routed to the ATen extension when it is
    available; it is never silently relabeled as a native FlashAttention path.
    """
    _state.dispatch_info = None
    require_aten = native_kwargs.get("softmax_scale") == 0.0
    vendor = None if require_aten else _flash_attn_function(operation)
    # Preserve the actionable contract reason before attempting fallbacks. In
    # particular, CPU reference tests and callers should see that native
    # execution requires an accelerator rather than a generic import failure.
    failure_reason: str | None = native_unavailable_reason(
        tensor, operation, require_aten=require_aten
    )

    if vendor is not None:
        try:
            result = vendor(*native_args, **native_kwargs)
            if operation == "flash_attn_with_kvcache" and native_kwargs.get("k") is not None:
                # Documented contract: appending k/v advances the caller's
                # cache_seqlens in place. Only this branch mutates it, so callers
                # that pass k=None (the vLLM decode path) need no defensive copy.
                native_kwargs["cache_seqlens"].add_(native_kwargs["k"].shape[1])
            _state.dispatch_info = DispatchInfo(backend="vendor_direct", operation=operation)
            return result
        except (Exception, RuntimeError) as exc:
            failure_reason = f"vendor_call_failed:{type(exc).__name__}:{exc}"

    extension = _native_extension()
    extension_reason = ""
    if extension is not None:
        try:
            available = bool(extension.is_available())
        except Exception as exc:
            available = False
            extension_reason = f"aten_availability_check_failed:{type(exc).__name__}:{exc}"
        if available and hasattr(extension, operation):
            try:
                result = getattr(extension, operation)(*extension_args)
                _state.dispatch_info = DispatchInfo(
                    backend="mxmac_aten_correctness",
                    operation=operation,
                    fallback_reason=failure_reason,
                )
                return result
            except Exception as exc:
                extension_reason = f"aten_call_failed:{type(exc).__name__}:{exc}"
        elif extension is None:
            extension_reason = "aten_extension_not_importable"
        elif not hasattr(extension, operation):
            extension_reason = f"aten_extension_does_not_implement_{operation}"
        elif not extension_reason:
            extension_reason = "aten_extension_reports_unavailable_runtime"

    reason = ";".join(part for part in (failure_reason, extension_reason) if part)
    if not reason:
        reason = "vendor_unavailable_or_parameter_incompatible"
    if os.getenv("MXFLASHATTN_ALLOW_FALLBACK", "").strip() != "1":
        raise BackendUnavailableError(
            f"MXFlashAttn dispatch failed ({reason}); set MXFLASHATTN_ALLOW_FALLBACK=1 "
            "to explicitly allow the PyTorch reference path"
        )

    _state.dispatch_info = DispatchInfo(
        backend="reference", operation=operation, fallback_reason=reason
    )
    warnings.warn(
        f"MXFlashAttn is using the PyTorch reference path: {reason}",
        RuntimeWarning,
        stacklevel=3,
    )
    return reference()


