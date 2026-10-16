"""FlashAttention-compatible APIs with explicit MXMACA backend dispatch."""

from .api import flash_attn_func, flash_attn_varlen_func, flash_attn_with_kvcache
from .dispatch import (
    BackendUnavailableError,
    DispatchInfo,
    get_last_dispatch_info,
)
from .prefill import prefill_chunked_reference, prefill_chunked_reference_fallback

__all__ = [
    "BackendUnavailableError",
    "DispatchInfo",
    "flash_attn_func",
    "flash_attn_varlen_func",
    "flash_attn_with_kvcache",
    "get_last_dispatch_info",
    "prefill_chunked_reference",
    "prefill_chunked_reference_fallback",
]

__version__ = "1.4.0.dev0"
