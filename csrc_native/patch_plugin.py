"""Patch vllm_plugin.py: add env-gated M1 split-KV decode branch (MXFA_M1_DECODE=1).

Inserts _load_m1_decode_ext() loader and an M1 branch in _try_native_decode
before the v7 ext.paged_decode call. Idempotent. Falls back to v7 on any error
because the whole M1 branch is inside the existing try/except.
"""
import sys

P = "/root/MXFlashAttn_v1.3.0-kernel/integrations/vllm/vllm_plugin.py"
src = open(P, encoding="utf-8").read()

if "_load_m1_decode_ext" in src:
    print("ALREADY_PATCHED")
    sys.exit(0)

LOADER = '''
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
'''

ANCHOR1 = "_native_ext = None\n"
assert ANCHOR1 in src, "loader anchor missing"
src = src.replace(ANCHOR1, LOADER + "\n" + ANCHOR1, 1)

BRANCH = '''        if os.getenv("MXFA_M1_DECODE", "0") == "1":
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
            out = _m1.paged_decode_split(
                q.reshape(B, 1, HQ, D), k_cache, v_cache,
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
'''

ANCHOR2 = "        out = ext.paged_decode(\n"
assert ANCHOR2 in src, "call anchor missing"
src = src.replace(ANCHOR2, BRANCH, 1)

open(P, "w", encoding="utf-8").write(src)
print("PATCHED_OK")
