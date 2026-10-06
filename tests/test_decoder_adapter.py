from dataclasses import dataclass

import pytest
import torch

from integrations.vllm import decoder_adapter
from integrations.vllm.decoder_adapter import DecoderAdapterBlocked, decode_forward, inspect_layout


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
    scheduler_metadata: torch.Tensor | None = None
    causal: bool = True


def make_case(query_lens=(1, 1)):
    q = torch.randn(sum(query_lens), 4, 64, dtype=torch.float16)
    cache = torch.randn(2, 4, 16, 2, 64, dtype=torch.float16)
    starts = [0]
    for length in query_lens:
        starts.append(starts[-1] + length)
    metadata = Metadata(
        num_actual_tokens=sum(query_lens), max_query_len=max(query_lens),
        query_start_loc=torch.tensor(starts), max_seq_len=17,
        seq_lens=torch.tensor([17, 16]), block_table=torch.tensor([[0, 1], [2, 3]]),
        slot_mapping=torch.tensor([0, 1]),
    )
    return q, cache, metadata


def test_decode_adapter_exercises_paged_api_with_explicit_fallback(monkeypatch):
    monkeypatch.setenv("MXFLASHATTN_ALLOW_FALLBACK", "1")
    q, cache, metadata = make_case()
    output, layout = decode_forward(q, cache, metadata, softmax_scale=64**-0.5)
    assert output.shape == (2, 4 * 64)
    assert layout.query_lens == (1, 1)


def test_prefill_is_blocked_instead_of_delegated():
    q, cache, metadata = make_case((2, 1))
    with pytest.raises(DecoderAdapterBlocked, match="single_token_decode"):
        decode_forward(q, cache, metadata, softmax_scale=64**-0.5)


def test_layout_rejects_wrong_cache_rank():
    q, _, metadata = make_case()
    with pytest.raises(DecoderAdapterBlocked, match="kv_cache_shape"):
        inspect_layout(q, torch.empty(2, 4, 16, 64), metadata)


def test_transparent_path_flattens_capture_seq_lens(monkeypatch):
    monkeypatch.setenv("MXFLASHATTN_TRANSPARENT", "1")
    monkeypatch.setenv("MXFLASHATTN_ALLOW_FALLBACK", "1")
    q, cache, metadata = make_case()
    metadata.seq_lens = metadata.seq_lens.reshape(1, -1)
    seen = {}

    def fake_flash(q, key_cache, value_cache, **kwargs):
        seen["cache_seqlens"] = kwargs["cache_seqlens"]
        seen["block_table"] = kwargs["block_table"]
        return torch.zeros_like(q)

    monkeypatch.setattr(decoder_adapter, "flash_attn_with_kvcache", fake_flash)
    output, layout = decode_forward(q, cache, metadata, softmax_scale=64**-0.5)
    assert output.shape == (2, 4 * 64)
    assert layout is None
    assert seen["cache_seqlens"].shape == (2,)
    assert seen["block_table"].shape == (2, 2)
