#include "attention.h"

#include <cmath>
#include <limits>

namespace mxflashattn {
namespace {

at::Tensor attend_one(
    const at::Tensor& q,
    const at::Tensor& k,
    const at::Tensor& v,
    bool causal,
    double softmax_scale) {
  const auto q_len = q.size(0);
  const auto kv_len = k.size(0);
  const auto q_heads = q.size(1);
  const auto kv_heads = k.size(1);
  const auto head_dim = q.size(2);
  TORCH_CHECK(kv_len > 0, "attention requires at least one key/value token");
  TORCH_CHECK(q_heads % kv_heads == 0, "q heads must be divisible by kv heads");

  const auto groups = q_heads / kv_heads;
  auto qh = q.transpose(0, 1).to(at::kFloat);
  auto kh = k.transpose(0, 1).to(at::kFloat);
  auto vh = v.transpose(0, 1).to(at::kFloat);
  if (groups != 1) {
    kh = kh.unsqueeze(1).expand({kv_heads, groups, kv_len, head_dim}).reshape({q_heads, kv_len, head_dim});
    vh = vh.unsqueeze(1).expand({kv_heads, groups, kv_len, head_dim}).reshape({q_heads, kv_len, head_dim});
  }

  TORCH_CHECK(std::isfinite(softmax_scale), "softmax_scale must be finite");
  const double scale = softmax_scale;
  auto scores = at::matmul(qh, kh.transpose(-2, -1)) * scale;
  if (causal) {
    auto q_pos = at::arange(q_len, q.options().dtype(at::kLong)) + (kv_len - q_len);
    auto k_pos = at::arange(kv_len, q.options().dtype(at::kLong));
    auto blocked = k_pos.unsqueeze(0) > q_pos.unsqueeze(1);
    scores = scores.masked_fill(blocked.unsqueeze(0), -std::numeric_limits<float>::infinity());
  }
  auto probabilities = scores.softmax(-1, at::kFloat);
  probabilities = at::where(at::isnan(probabilities), at::zeros_like(probabilities), probabilities);
  return at::matmul(probabilities, vh).transpose(0, 1).to(q.scalar_type());
}

std::vector<int64_t> offsets(const at::Tensor& input) {
  auto values = input.to(at::kLong).to(at::kCPU).contiguous();
  const auto* data = values.data_ptr<int64_t>();
  return std::vector<int64_t>(data, data + values.numel());
}

}  // namespace

at::Tensor flash_attn_func(
    const at::Tensor& q,
    const at::Tensor& k,
    const at::Tensor& v,
    bool causal,
    double softmax_scale) {
  std::vector<at::Tensor> outputs;
  outputs.reserve(q.size(0));
  for (int64_t batch = 0; batch < q.size(0); ++batch) {
    outputs.push_back(attend_one(q.select(0, batch), k.select(0, batch), v.select(0, batch), causal, softmax_scale));
  }
  return outputs.empty() ? q.clone() : at::stack(outputs, 0);
}

at::Tensor flash_attn_varlen_func(
    const at::Tensor& q,
    const at::Tensor& k,
    const at::Tensor& v,
    const at::Tensor& cu_seqlens_q,
    const at::Tensor& cu_seqlens_k,
    int64_t max_seqlen_q,
    int64_t max_seqlen_k,
    bool causal,
    double softmax_scale) {
  const auto q_offsets = offsets(cu_seqlens_q);
  const auto k_offsets = offsets(cu_seqlens_k);
  TORCH_CHECK(q_offsets.size() == k_offsets.size(), "cumulative lengths must have the same batch size");
  std::vector<at::Tensor> outputs;
  outputs.reserve(q_offsets.size() - 1);
  for (size_t batch = 0; batch + 1 < q_offsets.size(); ++batch) {
    const auto q_len = q_offsets[batch + 1] - q_offsets[batch];
    const auto kv_len = k_offsets[batch + 1] - k_offsets[batch];
    TORCH_CHECK(q_len <= max_seqlen_q && kv_len <= max_seqlen_k, "sequence length exceeds supplied maximum");
    TORCH_CHECK(kv_len > 0, "key/value sequences must be non-empty");
    outputs.push_back(attend_one(
        q.slice(0, q_offsets[batch], q_offsets[batch + 1]),
        k.slice(0, k_offsets[batch], k_offsets[batch + 1]),
        v.slice(0, k_offsets[batch], k_offsets[batch + 1]),
        causal,
        softmax_scale));
  }
  return outputs.empty() ? q.slice(0, 0, 0).clone() : at::cat(outputs, 0);
}

bool is_available() {
  return true;
}

}  // namespace mxflashattn

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  namespace py = pybind11;
  module.def("is_available", &mxflashattn::is_available);
  module.def("flash_attn_func", &mxflashattn::flash_attn_func,
             py::arg("q"), py::arg("k"), py::arg("v"), py::arg("causal"), py::arg("softmax_scale"));
  module.def("flash_attn_varlen_func", &mxflashattn::flash_attn_varlen_func,
             py::arg("q"), py::arg("k"), py::arg("v"), py::arg("cu_seqlens_q"), py::arg("cu_seqlens_k"),
             py::arg("max_seqlen_q"), py::arg("max_seqlen_k"), py::arg("causal"), py::arg("softmax_scale"));
  module.def("flash_attn_with_kvcache", &mxflashattn::flash_attn_with_kvcache,
             py::arg("q"), py::arg("k_cache"), py::arg("v_cache"), py::arg("k"), py::arg("v"),
             py::arg("cache_seqlens"), py::arg("block_table"), py::arg("softmax_scale"), py::arg("causal"));
  module.def("workspace_bytes", &mxflashattn::workspace_bytes);
}
