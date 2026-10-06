"""vLLM 0.17.0 registration and dense bridge for the local E2E candidate."""

from __future__ import annotations

import json
import os

import torch

from mxflashattn import flash_attn_func


DISPATCH_EVENTS: list[dict[str, str]] = []

# Recording runs once per attention call (28 layers x N steps x prompts). The
# event payloads repeat heavily, so serialisation is cached. Set
# MXFLASHATTN_DISPATCH_SAMPLE=N to write only every Nth event: the log still
# shows which backend served the run, but the write cost leaves the hot path.
_JSON_CACHE: dict[tuple[tuple[str, str], ...], str] = {}
_SAMPLE_EVERY = int(os.getenv("MXFLASHATTN_DISPATCH_SAMPLE", "0") or 0)
_event_count = 0


def get_dispatch_events() -> list[dict[str, str]]:
    return list(DISPATCH_EVENTS)


def clear_dispatch_events(truncate_log: bool = True) -> None:
    """Drop buffered events.

    ``truncate_log=False`` keeps previously written lines, which is what
    multi-prompt suites need: the in-memory window resets per prompt, but the
    on-disk evidence must accumulate across the whole run.
    """
    DISPATCH_EVENTS.clear()
    if not truncate_log:
        return
    path = os.getenv("MXFLASHATTN_DISPATCH_LOG")
    if path:
        try:
            open(path, "w", encoding="utf-8").close()
        except OSError:
            pass


def _record_event(event: dict[str, str]) -> None:
    DISPATCH_EVENTS.append(event)
    path = os.getenv("MXFLASHATTN_DISPATCH_LOG")
    # EngineCore workers may import the plugin before the parent-side smoke
    # runner's environment mutation is visible.  The plugin log is already
    # worker-visible; derive a sibling dispatch path as a durable fallback.
    if not path:
        plugin_log = os.getenv("MXFLASHATTN_PLUGIN_IMPORT_LOG")
        if plugin_log:
            path = plugin_log.removesuffix(".plugin.log") + ".dispatch.jsonl"
    if not path:
        return
    if _SAMPLE_EVERY > 0:
        # Sampling keeps the evidence trail (which backend served which call)
        # while removing the write cost from the measured hot path.
        global _event_count
        _event_count += 1
        if _event_count % _SAMPLE_EVERY != 1:
            return
    try:
        key = tuple(sorted(event.items()))
        rendered = _JSON_CACHE.get(key)
        if rendered is None:
            rendered = json.dumps(event, sort_keys=True) + "\n"
            _JSON_CACHE[key] = rendered
        with open(path, "a", encoding="utf-8") as stream:
            stream.write(rendered)
    except OSError:
        pass


def attention_forward(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    causal: bool,
    scale: float,
) -> torch.Tensor:
    """Adapt dense vLLM token/head tensors to the MXFlashAttn public API."""
    if query.ndim != 3 or key.ndim != 3 or value.shape != key.shape:
        raise ValueError("vLLM Q/K/V must use [tokens, heads, head_dim] layout")
    output = flash_attn_func(
        query.unsqueeze(0), key.unsqueeze(0), value.unsqueeze(0),
        causal=causal, softmax_scale=scale,
    )
    return output.squeeze(0)


def register_vllm_v1_backend() -> dict[str, str]:
    """Register an importable backend path for vLLM 0.17.0 workers."""
    try:
        from vllm.v1.attention.backends.registry import AttentionBackendEnum, register_backend
    except Exception as exc:  # pragma: no cover - depends on installed vLLM
        return {"registered": "false", "reason": f"vllm_v1_api_unavailable: {type(exc).__name__}: {exc}"}
    class_path = "integrations.vllm.vllm_plugin.MXFlashAttnBackend"

    def apply_override() -> None:
        register_backend(AttentionBackendEnum.FLASH_ATTN, class_path=class_path)

    apply_override()
    # The MetaX selector may intentionally return its own backend class even
    # after registry overrides.  Replace only its implementation factory so
    # the vendor metadata builder and constructor remain intact while the
    # decoder implementation is ours.
    try:
        from integrations.vllm.vllm_plugin import MXFlashAttnImpl
        import vllm_metax.v1.attention.backends.flash_attn as metax_flash_attn

        metax_flash_attn.MacaFlashAttentionBackend.get_impl_cls = staticmethod(
            lambda: MXFlashAttnImpl
        )
    except Exception:
        pass
    # The MetaX vLLM plugin re-registers FLASH_ATTN during backend selection.
    # Reapply our importable path after that platform hook runs in each worker.
    try:
        import vllm_metax.platform as metax_platform

        original = metax_platform.register_attention_backends
        if not getattr(original, "_mxflashattn_wrapped", False):
            def wrapped_register_attention_backends() -> None:
                original()
                apply_override()

            wrapped_register_attention_backends._mxflashattn_wrapped = True
            metax_platform.register_attention_backends = wrapped_register_attention_backends

        # MetaX's selector calls its registration hook immediately before
        # resolving the class path.  In some worker start modes that hook is
        # bound before the wrapper above is installed, so enforce the path at
        # the final selector boundary as well.
        platform_base = getattr(metax_platform, "MacaPlatformBase", None)
        descriptor = getattr(platform_base, "__dict__", {}).get("get_attn_backend_cls")
        selector = descriptor.__func__ if isinstance(descriptor, classmethod) else descriptor
        if selector is not None and not getattr(selector, "_mxflashattn_wrapped", False):
            def wrapped_selector(cls, selected_backend, attn_selector_config, num_heads=None):
                if getattr(selected_backend, "name", None) == "FLASH_ATTN":
                    apply_override()
                    return class_path
                return selector(cls, selected_backend, attn_selector_config, num_heads)

            wrapped_selector._mxflashattn_wrapped = True
            platform_base.get_attn_backend_cls = classmethod(wrapped_selector)
    except Exception:
        pass
    return {
        "registered": "true",
        "backend_name": "MXFLASHATTN_V017",
        "backend_path": class_path,
        "decoder_path": "vendor_direct_with_aten_correctness_fallback",
        "encoder_path": "mxflashattn_public_api",
    }
