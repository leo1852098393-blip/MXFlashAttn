"""Capture vLLM 0.17 decoder layout evidence and exercise the strict adapter."""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path

import torch

from integrations.vllm.decoder_adapter import DecoderAdapterBlocked, decode_forward, inspect_layout
from mxflashattn.dispatch import get_last_dispatch_info


@dataclass
class Metadata:
    num_actual_tokens: int
    max_query_len: int
    query_start_loc: torch.Tensor
    max_seq_len: int
    seq_lens: torch.Tensor
    block_table: torch.Tensor
    slot_mapping: torch.Tensor
    use_cascade: bool = False
    common_prefix_len: int = 0
    cu_prefix_query_lens: torch.Tensor | None = None
    prefix_kv_lens: torch.Tensor | None = None
    suffix_kv_lens: torch.Tensor | None = None
    scheduler_metadata: torch.Tensor | None = None
    causal: bool = True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("artifacts/decoder-layout.json"))
    parser.add_argument("--log", type=Path, default=Path("artifacts/decoder-layout.log"))
    args = parser.parse_args()
    os.environ.setdefault("MXFLASHATTN_ALLOW_FALLBACK", "1")
    torch.manual_seed(7)
    # MetaX PyTorch exposes C500 through the CUDA-compatible device namespace.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16
    batch, q_heads, kv_heads, dim = 2, 4, 2, 64
    block_size, blocks, seq_len = 16, 4, 17
    query = torch.randn(batch, q_heads, dim, device=device, dtype=dtype)
    kv_cache = torch.randn(2, blocks, block_size, kv_heads, dim, device=device, dtype=dtype)
    metadata = Metadata(
        num_actual_tokens=batch, max_query_len=1,
        query_start_loc=torch.tensor([0, 1, 2], device=device, dtype=torch.int32),
        max_seq_len=seq_len,
        seq_lens=torch.tensor([seq_len, seq_len - 1], device=device, dtype=torch.int32),
        block_table=torch.tensor([[0, 1], [2, 3]], device=device, dtype=torch.int32),
        slot_mapping=torch.tensor([0, 1], device=device, dtype=torch.int32),
    )
    started = time.perf_counter()
    record: dict[str, object] = {
        "schema_version": 1, "kind": "vllm_decoder_layout_probe", "target_vllm": "0.17.0",
        "device": str(device), "dtype": str(dtype), "contract": "single_token_decode_paged_kv",
        "source_layout": {
            "query": "[num_tokens,num_heads,head_size]",
            "kv_cache": "[2,num_blocks,block_size,num_kv_heads,head_size]",
            "metadata": ["num_actual_tokens", "max_query_len", "query_start_loc", "max_seq_len",
                          "seq_lens", "block_table", "slot_mapping", "scheduler_metadata", "use_cascade"],
        },
    }
    try:
        layout = inspect_layout(query, kv_cache, metadata)
        output, layout = decode_forward(query, kv_cache, metadata, softmax_scale=dim ** -0.5)
        info = get_last_dispatch_info()
        record.update({"status": "adapter_exercised", "layout": layout.__dict__,
                       "output_shape": list(output.shape),
                       "dispatch": info.__dict__ if info else None,
                       "elapsed_ms": (time.perf_counter() - started) * 1000,
                       "fallback_allowed": True})
    except DecoderAdapterBlocked as exc:
        record.update({"status": "blocked", "blocker": str(exc), "dispatch": None})
    except Exception as exc:
        info = get_last_dispatch_info()
        record.update({"status": "adapter_runtime_error", "error": f"{type(exc).__name__}: {exc}",
                       "dispatch": info.__dict__ if info else None})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, indent=2, default=str), encoding="utf-8")
    args.log.write_text(json.dumps(record, indent=2, default=str) + "\n", encoding="utf-8")
    print(json.dumps(record, indent=2, default=str))


if __name__ == "__main__":
    main()
