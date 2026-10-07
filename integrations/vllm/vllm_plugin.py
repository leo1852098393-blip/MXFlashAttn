"""Importable vLLM 0.17.0 backend classes.

The classes live at module scope because vLLM's registry resolves the backend
by import path inside spawned EngineCore workers.
"""

from __future__ import annotations

import os

from vllm.v1.attention.backend import AttentionType
_METAX_IMPL = None
try:
    # MetaX names its backend class MacaFlashAttentionBackend; importing the
    # upstream name FlashAttentionBackend from vllm_metax raises ImportError
    # and previously fell back to the UPSTREAM vLLM classes, silently making
    # the candidate run the upstream attention implementation instead of the
    # vendor one.  Import the real vendor class names here.
    from vllm_metax.v1.attention.backends.flash_attn import (
        FlashAttentionImpl,
        MacaFlashAttentionBackend as FlashAttentionBackend,
    )
    _METAX_IMPL = FlashAttentionImpl
except ImportError as _metax_import_error:  # CPU/mock CI without the vendor distribution.
    from vllm.v1.attention.backends.flash_attn import (
        FlashAttentionBackend,
        FlashAttentionImpl,
    )

# The MetaX attention module can be unavailable during worker import when its
# optional FA2 probe runs before the C500 runtime is initialized.  The vendor
# utility module remains importable and provides the exact cache op and FA
# version contract needed by the upstream implementation.
try:
    import vllm.v1.attention.backends.flash_attn as _upstream_fa
    from vllm_metax.v1.attention.backends import fa_utils as _metax_fa_utils

    _upstream_fa.get_flash_attn_version = (
        lambda requires_alibi=False, head_size=None: _metax_fa_utils.get_flash_attn_version(
            requires_alibi=requires_alibi, head_size=head_size
        )
    )
    if hasattr(_metax_fa_utils, "reshape_and_cache_flash"):
        _upstream_fa.reshape_and_cache_flash = _metax_fa_utils.reshape_and_cache_flash
    for _name in ("flash_attn_varlen_func", "flash_attn_with_kvcache"):
        if hasattr(_metax_fa_utils, _name):
            setattr(_upstream_fa, _name, getattr(_metax_fa_utils, _name))
    if hasattr(_metax_fa_utils, "flash_attn_varlen_func"):
        _vendor_varlen = _metax_fa_utils.flash_attn_varlen_func

        def _compat_varlen(*args, **kwargs):
            out = kwargs.pop("out", None)
            seqused_k = kwargs.pop("seqused_k", None)
            if seqused_k is not None and "cu_seqlens_k" not in kwargs:
                import torch

                lengths = seqused_k.to(dtype=torch.int32)
                if "cu_seqlens_q" in kwargs:
                    kwargs["cu_seqlens_q"] = kwargs["cu_seqlens_q"].to(dtype=torch.int32)
                kwargs["cu_seqlens_k"] = torch.cat(
                    (torch.zeros(1, dtype=lengths.dtype, device=lengths.device), lengths.cumsum(0))
                )
            if "cu_seqlens_k" in kwargs:
                kwargs["cu_seqlens_k"] = kwargs["cu_seqlens_k"].to(dtype=torch.int32)
            if "cu_seqlens_q" in kwargs:
                kwargs["cu_seqlens_q"] = kwargs["cu_seqlens_q"].to(dtype=torch.int32)
            for _unsupported in (
                "fa_version", "q_descale", "k_descale", "v_descale",
                "num_splits", "scheduler_metadata",
            ):
                kwargs.pop(_unsupported, None)
            result = _vendor_varlen(*args, **kwargs)
            if isinstance(result, tuple):
                result = result[0]
            if out is not None:
                out.copy_(result)
                return out
            return result

        _upstream_fa.flash_attn_varlen_func = _compat_varlen
except Exception:
    pass

from integrations.vllm.backend import _record_event
from integrations.vllm.decoder_adapter import DecoderAdapterBlocked, decode_forward
from mxflashattn import flash_attn_func, flash_attn_varlen_func, flash_attn_with_kvcache
from mxflashattn.dispatch import get_last_dispatch_info




_worker_event_count = 0

def _record_worker_event(event: dict[str, str]) -> None:
    """Persist worker-owned evidence even when backend module state is isolated."""
    global _worker_event_count
    _worker_event_count += 1
    sample = int(os.getenv("MXFLASHATTN_DISPATCH_SAMPLE", "0") or 0)
    if sample > 1 and _worker_event_count % sample != 1:
        return
    path = os.getenv("MXFLASHATTN_DISPATCH_LOG")
    if not path:
        plugin_log = os.getenv("MXFLASHATTN_PLUGIN_IMPORT_LOG")
        if plugin_log:
            path = plugin_log.removesuffix(".plugin.log") + ".dispatch.jsonl"
    if not path:
        return
    try:
        import json
        with open(path, "a", encoding="utf-8") as stream:
            stream.write(json.dumps(event, sort_keys=True) + "\n")
    except OSError:
        pass

_plugin_log = os.getenv("MXFLASHATTN_PLUGIN_IMPORT_LOG")
if _plugin_log:
    with open(_plugin_log, "a", encoding="utf-8") as stream:
        stream.write(
            f"plugin_imported pid={os.getpid()} impl_module={FlashAttentionImpl.__module__}\n"
        )
        if _METAX_IMPL is None and "_metax_import_error" in globals():
            stream.write(f"metax_import_error={_metax_import_error!r}\n")


_official_varlen = None

def _mx_varlen_compat(*args, **kwargs):
    _TAG = "varlen_compat"
    if os.getenv("MXFLASHATTN_SPLIT_PROBE", "0") == "1":
        try:
            import torch as _t
            _cap = _t.cuda.is_current_stream_capturing()
            _q = args[0] if args else kwargs.get("q")
            with open("/root/probe_split.log", "a", encoding="utf-8") as _ps:
                _ps.write("%s capturing=%s qshape=%s kwkeys=%s window=%r causal=%r\n" % (
                    _TAG, _cap, tuple(_q.shape) if _q is not None else None,
                    sorted(kwargs.keys()), kwargs.get("window_size"), kwargs.get("causal")))
        except Exception as _e:
            pass
    # MXFlashAttn's public varlen API has no paged block-table parameter.
    # Preserve the official MetaX varlen call for paged prefill; only the
    # kvcache decode call is replaced by MXFlashAttn in this experiment.
    if "block_table" in kwargs:
        return _official_varlen(*args, **kwargs)
    out = kwargs.pop("out", None)
    seqused_k = kwargs.pop("seqused_k", None)
    if seqused_k is not None and "cu_seqlens_k" not in kwargs:
        import torch
        lengths = seqused_k.to(dtype=torch.int32)
        kwargs["cu_seqlens_k"] = torch.cat((torch.zeros(1, dtype=torch.int32, device=lengths.device), lengths.cumsum(0)))
    kwargs.pop("s_aux", None)
    kwargs.pop("return_attn_probs", None)
    kwargs.pop("scheduler_metadata", None)
    kwargs.pop("fa_version", None)
    kwargs.pop("q_descale", None); kwargs.pop("k_descale", None); kwargs.pop("v_descale", None)
    result = flash_attn_varlen_func(*args, **kwargs)
    if isinstance(result, tuple):
        result = result[0]
    if out is not None:
        out.copy_(result)
        return out
    return result


_m1_ext = None

def _load_m1_decode_ext():
    """Lazily import the M1 split-KV decode extension (mxmmain.so, mxcc build)."""
    global _m1_ext
    if _m1_ext is None:
        import importlib.util
        _path = os.path.abspath(os.path.join(
            os.path.dirname(__file__), "..", "..", "csrc_native", "mxmmain.so"))
        _spec = importlib.util.spec_from_file_location("mxmmain", _path)
        _mod = importlib.util.module_from_spec(_spec)
        _spec.loader.exec_module(_mod)
        _m1_ext = _mod
    return _m1_ext

_native_ext = None

def _load_native_decode_ext():
    """Lazily import the csrc_native torch extension (built with mxcc)."""
    global _native_ext
    if _native_ext is None:
        import importlib.util
        _path = os.path.abspath(os.path.join(
            os.path.dirname(__file__), "..", "..", "csrc_native", "mxfadecode.so"))
        _spec = importlib.util.spec_from_file_location("mxfadecode", _path)
        _mod = importlib.util.module_from_spec(_spec)
        _spec.loader.exec_module(_mod)
        _native_ext = _mod
    return _native_ext

def _try_native_decode(args, kwargs):
    """Route single-token paged decode to the native MACA kernel.

    Returns None whenever the call does not match the supported shape so the
    caller falls through to the regular vendor/MXFlashAttn path unchanged.
    """
    if os.getenv("MXFA_NATIVE_DECODE", "0") != "1":
        return None
    import torch

    _dbg = os.getenv("MXFA_NATIVE_DEBUG_LOG")

    def _dbg_write(msg):
        if _dbg:
            try:
                with open(_dbg, "a", encoding="utf-8") as _s:
                    _s.write(msg + "\n")
            except OSError:
                pass

    try:
        q = args[0] if args else kwargs.get("q")
        if q is None or q.dim() not in (3, 4):
            _dbg_write("skip qdim q=%r" % (None if q is None else tuple(q.shape)))
            return None
        bt = kwargs.get("block_table")
        sl = kwargs.get("cache_seqlens")
        if bt is None or sl is None or sl.shape[0] != q.shape[0]:
            _dbg_write("skip bt_sl bt=%r sl=%r q=%r" % (
                None if bt is None else tuple(bt.shape),
                None if sl is None else tuple(sl.shape), tuple(q.shape)))
            return None
        if kwargs.get("alibi_slopes") is not None:
            _dbg_write("skip alibi=%r" % (kwargs.get("alibi_slopes"),))
            return None
        _ws = kwargs.get("window_size")
        if _ws is not None:
            # vLLM passes (-1, -1) (or -1) for "no sliding window"; only a
            # positive window would need masking support we do not provide.
            try:
                _ws_vals = _ws if isinstance(_ws, (tuple, list)) else (_ws,)
                _ws_bad = any(int(v) > 0 for v in _ws_vals)
            except (TypeError, ValueError):
                _ws_bad = True
            if _ws_bad:
                _dbg_write("skip window window=%r" % (_ws,))
                return None
        k_cache = kwargs.get("k_cache")
        if k_cache is None and len(args) > 1:
            k_cache = args[1]
        v_cache = kwargs.get("v_cache")
        if v_cache is None and len(args) > 2:
            v_cache = args[2]
        if k_cache is None or v_cache is None:
            _dbg_write("skip caches kv=%r" % ((k_cache is None, v_cache is None),))
            return None
        if q.dtype not in (torch.float16, torch.bfloat16) or k_cache.dtype != q.dtype:
            _dbg_write("skip dtype q=%r k=%r" % (q.dtype, k_cache.dtype))
            return None
        B, HQ, D = (q.shape[0], q.shape[-2], q.shape[-1])
        if q.shape[-3] != 1 and q.dim() == 4:
            _dbg_write("skip qlen q=%r" % (tuple(q.shape),))
            return None
        ext = _load_native_decode_ext()
        scale = kwargs.get("softmax_scale") or (D ** -0.5)
        max_sl_bound = int(bt.shape[1]) * int(k_cache.shape[1])
        stream = torch.cuda.current_stream().cuda_stream
        if os.getenv("MXFA_M1_DECODE", "0") == "1":
            # M1 split-KV path: kernel handles GQA D in {64,128}, GQ in [2,8];
            # any unsupported shape raises and falls through to the v7 ext below.
            _m1 = _load_m1_decode_ext()
            _scale = kwargs.get("softmax_scale") or (D ** -0.5)
            _msl = int(bt.shape[1]) * int(k_cache.shape[1])
            _base = B * int(k_cache.shape[2])
            _np = 1
            while _np < 64 and _base * _np < 2048:
                _np *= 2
            _c_len = (_msl + _np - 1) // _np
            # vLLM hands us q as a slice of a fused QKV buffer, and some model
            # families (Llama on this stack) keep the batch stride padded, so
            # reshape() alone yields a non-contiguous view.  Both bindings
            # require contiguous q, hence the explicit copy.  No-op when the
            # caller already passed a contiguous tensor.
            _qr = q.reshape(B, 1, HQ, D)
            if not _qr.is_contiguous():
                _qr = _qr.contiguous()
            out = _m1.paged_decode_split(
                _qr, k_cache, v_cache,
                bt.to(torch.int32), sl.to(torch.int32), _scale,
                torch.cuda.current_stream().cuda_stream, _np, _c_len,
            )
            provided = kwargs.get("out")
            if provided is not None:
                provided.copy_(out.reshape_as(provided))
                result = provided
            else:
                result = out.reshape(q.shape)
            _dbg_write("M1_FIRE q=%s np=%d c_len=%d" % (tuple(q.shape), _np, _c_len))
            _record_worker_event({"path": "m1_split_decode",
                                  "operation": "paged_decode_split",
                                  "fallback": "false", "np": _np})
            return result
        out = ext.paged_decode(
            q.reshape(B, 1, HQ, D).contiguous(), k_cache, v_cache,
            bt.to(torch.int32), sl.to(torch.int32), scale, max_sl_bound, stream,
        )
        provided = kwargs.get("out")
        if provided is not None:
            provided.copy_(out.reshape_as(provided))
            result = provided
        else:
            result = out.reshape(q.shape)
        _dbg_write("FIRE q=%s" % (tuple(q.shape),))
        _record_worker_event({"path": "native_decode", "operation": "paged_decode", "fallback": "false"})
        return result
    except Exception as _exc:  # native path must never take the model down
        _dbg_write("EXC %r" % (_exc,))
        _record_worker_event({"path": "native_decode", "fallback": "true", "fallback_reason": repr(_exc)})
        return None

def _mx_kvcache_compat(*args, **kwargs):
    _TAG = "kvcache_compat"
    if os.getenv("MXFLASHATTN_SPLIT_PROBE", "0") == "1":
        try:
            import torch as _t
            _cap = _t.cuda.is_current_stream_capturing()
            _q = args[0] if args else kwargs.get("q")
            with open("/root/probe_split.log", "a", encoding="utf-8") as _ps:
                _ps.write("%s capturing=%s qshape=%s kwkeys=%s window=%r causal=%r\n" % (
                    _TAG, _cap, tuple(_q.shape) if _q is not None else None,
                    sorted(kwargs.keys()), kwargs.get("window_size"), kwargs.get("causal")))
        except Exception as _e:
            pass
    kwargs.pop("s_aux", None)
    kwargs.pop("return_softmax_lse", None)
    kwargs.pop("scheduler_metadata", None)
    kwargs.pop("fa_version", None)
    kwargs.pop("q_descale", None); kwargs.pop("k_descale", None); kwargs.pop("v_descale", None)
    _native = _try_native_decode(args, kwargs)
    if _native is not None:
        return _native
    return flash_attn_with_kvcache(*args, **kwargs)

def _forward_official_split_with_mx(self, layer, query, key, value, kv_cache, metadata, output, output_scale, output_block_scale):
    fn = FlashAttentionImpl.forward
    glb = fn.__globals__
    global _official_varlen
    old_var = glb.get("flash_attn_varlen_func")
    old_kv = glb.get("flash_attn_with_kvcache")
    _official_varlen = old_var
    glb["flash_attn_varlen_func"] = _mx_varlen_compat
    glb["flash_attn_with_kvcache"] = _mx_kvcache_compat
    try:
        result = super(MXFlashAttnImpl, self).forward(
            layer, query, key, value, kv_cache, metadata, output,
            output_scale, output_block_scale
        )
    finally:
        glb["flash_attn_varlen_func"] = old_var
        glb["flash_attn_with_kvcache"] = old_kv
        _official_varlen = None
    _record_worker_event({"path": "vendor_direct", "operation": "official_split_mx" , "fallback": "false"})
    return result

class MXFlashAttnImpl(FlashAttentionImpl):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if _plugin_log:
            with open(_plugin_log, "a", encoding="utf-8") as stream:
                stream.write(f"impl_constructed pid={os.getpid()}\n")

    def forward(
        self, layer, query, key, value, kv_cache, attn_metadata, output=None,
        output_scale=None, output_block_scale=None,
    ):
        if _plugin_log:
            try:
                with open(_plugin_log, "a", encoding="utf-8") as stream:
                    stream.write(
                        f"forward_entered pid={os.getpid()} "
                        f"q_shape={tuple(query.shape)} "
                        f"max_query_len={getattr(attn_metadata, 'max_query_len', None)} "
                        f"num_decodes={getattr(attn_metadata, 'num_decodes', None)} "
                        f"num_decode_tokens={getattr(attn_metadata, 'num_decode_tokens', None)} "
                        f"decode_seq_shape={getattr(getattr(attn_metadata, 'decode_seq_lens', None), 'shape', None)} "
                        f"decode_bt_shape={getattr(getattr(attn_metadata, 'decode_block_table', None), 'shape', None)} "
                        f"num_actual={getattr(attn_metadata, 'num_actual_tokens', None)} "
                        f"q_start_shape={getattr(getattr(attn_metadata, 'query_start_loc', None), 'shape', None)}\n"
                    )
            except OSError:
                pass
        if attn_metadata is None:
            if output is None:
                raise ValueError("vLLM must provide an output buffer for profiling")
            return output.fill_(0)
        # Keep the official MetaX metadata split and output writeback, while
        # replacing only the underlying attention calls with MXFlashAttn.
        if os.getenv("MXFLASHATTN_SPLIT_PURE", "0") == "1":
            return super().forward(
                layer, query, key, value, kv_cache, attn_metadata,
                output, output_scale, output_block_scale
            )
        if os.getenv("MXFLASHATTN_GRAPH_OFFICIAL_SPLIT", "0") == "1":
            return _forward_official_split_with_mx(
                self, layer, query, key, value, kv_cache, attn_metadata,
                output, output_scale, output_block_scale
            )
        if self.attn_type in (AttentionType.ENCODER_ONLY, AttentionType.ENCODER) and attn_metadata is not None:
            if output is None:
                raise ValueError("vLLM must provide an output buffer")
            result = flash_attn_func(
                query[: attn_metadata.num_actual_tokens].unsqueeze(0),
                key[: attn_metadata.num_actual_tokens].unsqueeze(0),
                value[: attn_metadata.num_actual_tokens].unsqueeze(0),
                causal=False, softmax_scale=self.scale,
            ).reshape(-1, self.num_heads * self.head_size)
            _record_event({"path": "mxflashattn_encoder", "operation": "flash_attn_func", "fallback": "false"})
            output[: result.shape[0]].copy_(result)
            return output
        try:
            result, _ = decode_forward(
                query, kv_cache, attn_metadata, softmax_scale=self.scale,
                causal=bool(attn_metadata.causal),
            )
        except DecoderAdapterBlocked as exc:
            if _plugin_log:
                try:
                    with open(_plugin_log, "a", encoding="utf-8") as stream:
                        stream.write(f"adapter_blocked pid={os.getpid()} reason={exc}\n")
                except OSError:
                    pass
            event = {"path": "vendor_prefill_or_blocked", "reason": str(exc)}
            _record_event(event)
            _record_worker_event(event)
            if "single_token_decode" in str(exc):
                return super().forward(layer, query, key, value, kv_cache, attn_metadata, output, output_scale, output_block_scale)
            raise
        dispatch_info = get_last_dispatch_info()
        if dispatch_info is None:
            raise RuntimeError("MXFlashAttn decoder returned without dispatch evidence")
        if dispatch_info.backend == "vendor_direct":
            event = {"path": "vendor_direct", "operation": dispatch_info.operation, "fallback": "false"}
        elif dispatch_info.backend == "mxmac_aten_correctness":
            event = {"path": "mxmac_aten_correctness", "operation": dispatch_info.operation, "fallback": "true", "fallback_reason": dispatch_info.fallback_reason or "unspecified"}
        else:
            event = {"path": "reference", "operation": dispatch_info.operation, "fallback": "true", "fallback_reason": dispatch_info.fallback_reason or "unspecified"}
        _record_event(event)
        _record_worker_event(event)
        if output is None:
            raise ValueError("vLLM must provide an output buffer")
        target = output[: result.shape[0]]
        if target.numel() != result.numel():
            raise RuntimeError(
                f"MXFlashAttn decoder output size mismatch: target={tuple(target.shape)} "
                f"result={tuple(result.shape)}"
            )
        target.copy_(result.reshape_as(target))
        return output


class MXFlashAttnBackend(FlashAttentionBackend):
    @staticmethod
    def get_name() -> str:
        # vLLM Attention layers index the backend enum by this name.  Keep the
        # standard enum key while the registry path points at our class.
        return "FLASH_ATTN"

    @staticmethod
    def get_impl_cls():
        return MXFlashAttnImpl


if _METAX_IMPL is not None:
    # The MetaX implementation supplies the cache-update helper with its
    # vendor-specific symbols.  Keep it even if import order made the selected
    # base class resolve to upstream vLLM.
    MXFlashAttnImpl.do_kv_cache_update = _METAX_IMPL.do_kv_cache_update
