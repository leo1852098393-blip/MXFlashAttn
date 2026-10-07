"""FlashAttention-compatible APIs with explicit MXMACA backend dispatch."""

from .api import flash_attn_func, flash_attn_varlen_func, flash_attn_with_kvcache
from .dispatch import (
    BackendUnavailableError,
    DispatchInfo,
    get_last_dispatch_info,
)

__all__ = [
    "BackendUnavailableError",
    "DispatchInfo",
    "flash_attn_func",
    "flash_attn_varlen_func",
    "flash_attn_with_kvcache",
    "get_last_dispatch_info",
]

__version__ = "0.1.0"
