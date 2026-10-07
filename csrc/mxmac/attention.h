#pragma once

#include <torch/extension.h>
#include <pybind11/pybind11.h>

#include <cstdint>
#include <vector>

namespace mxflashattn {

at::Tensor flash_attn_func(
    const at::Tensor& q,
    const at::Tensor& k,
    const at::Tensor& v,
    bool causal,
    double softmax_scale);

at::Tensor flash_attn_varlen_func(
    const at::Tensor& q,
    const at::Tensor& k,
    const at::Tensor& v,
    const at::Tensor& cu_seqlens_q,
    const at::Tensor& cu_seqlens_k,
    int64_t max_seqlen_q,
    int64_t max_seqlen_k,
    bool causal,
    double softmax_scale);

at::Tensor flash_attn_with_kvcache(
    const at::Tensor& q,
    const at::Tensor& k_cache,
    const at::Tensor& v_cache,
    pybind11::object k,
    pybind11::object v,
    at::Tensor cache_seqlens,
    pybind11::object block_table,
    double softmax_scale,
    bool causal);

bool is_available();
int64_t workspace_bytes(int64_t batch, int64_t q_len, int64_t kv_len, int64_t head_dim);

}  // namespace mxflashattn
