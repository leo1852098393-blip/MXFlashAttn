#include "attention.h"

#include <cmath>
#include <limits>
#include <vector>

namespace mxflashattn {
namespace {

std::vector<int64_t> cpu_values(const at::Tensor& tensor) {
  auto values = tensor.to(at::kLong).to(at::kCPU).contiguous();
  const auto* data = values.data_ptr<int64_t>();
  return std::vector<int64_t>(data, data + values.numel());
}

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

  auto qh = q.transpose(0, 1).to(at::kFloat);
  auto kh = k.transpose(0, 1).to(at::kFloat);
  auto vh = v.transpose(0, 1).to(at::kFloat);
  const auto groups = q_heads / kv_heads;
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
    scores = scores.masked_fill((k_pos.unsqueeze(0) > q_pos.unsqueeze(1)).unsqueeze(0), -std::numeric_limits<float>::infinity());
  }
  auto probabilities = scores.softmax(-1, at::kFloat);
  probabilities = at::where(at::isnan(probabilities), at::zeros_like(probabilities), probabilities);
  return at::matmul(probabilities, vh).transpose(0, 1).to(q.scalar_type());
}

}  // namespace

at::Tensor flash_attn_with_kvcache(
    const at::Tensor& q,
    const at::Tensor& k_cache,
    const at::Tensor& v_cache,
    pybind11::object k_object,
    pybind11::object v_object,
    at::Tensor cache_seqlens,
    pybind11::object block_table_object,
    double softmax_scale,
    bool causal) {
  const bool append = !k_object.is_none();
  TORCH_CHECK(append == !v_object.is_none(), "k and v must be provided together");
  at::Tensor k_new;
  at::Tensor v_new;
  if (append) {
    k_new = k_object.cast<at::Tensor>();
    v_new = v_object.cast<at::Tensor>();
  }
  const bool paged = !block_table_object.is_none();
  at::Tensor block_table;
  if (paged) {
    block_table = block_table_object.cast<at::Tensor>();
  }

  const auto lengths = cpu_values(cache_seqlens);
  const auto table = paged ? cpu_values(block_table) : std::vector<int64_t>();
  const auto table_width = paged ? block_table.size(1) : 0;
  const auto page_size = paged ? k_cache.size(1) : 0;
  std::vector<at::Tensor> outputs;
  outputs.reserve(q.size(0));

  for (int64_t batch = 0; batch < q.size(0); ++batch) {
    const auto old_len = lengths[batch];
    const auto append_len = append ? k_new.size(1) : 0;
    const auto total_len = old_len + append_len;
    TORCH_CHECK(total_len > 0, "attention requires at least one cached or appended key/value token");
    at::Tensor current_k;
    at::Tensor current_v;
    if (!paged) {
      if (append_len > 0) {
        k_cache.select(0, batch).slice(0, old_len, total_len).copy_(k_new.select(0, batch));
        v_cache.select(0, batch).slice(0, old_len, total_len).copy_(v_new.select(0, batch));
      }
      current_k = k_cache.select(0, batch).slice(0, 0, total_len);
      current_v = v_cache.select(0, batch).slice(0, 0, total_len);
    } else {
      const auto* row = table.data() + batch * table_width;
      std::vector<at::Tensor> key_pages;
      std::vector<at::Tensor> value_pages;
      for (int64_t token = 0; token < append_len; ++token) {
        const auto logical = old_len + token;
        const auto page_idx = logical / page_size;
        const auto page_offset = logical % page_size;
        const auto physical = row[page_idx];
        k_cache.select(0, physical).select(0, page_offset).copy_(k_new.select(0, batch).select(0, token));
        v_cache.select(0, physical).select(0, page_offset).copy_(v_new.select(0, batch).select(0, token));
      }
      const auto num_pages = (total_len + page_size - 1) / page_size;
      for (int64_t page = 0; page < num_pages; ++page) {
        key_pages.push_back(k_cache.select(0, row[page]));
        value_pages.push_back(v_cache.select(0, row[page]));
      }
      current_k = at::cat(key_pages, 0).slice(0, 0, total_len);
      current_v = at::cat(value_pages, 0).slice(0, 0, total_len);
    }
    outputs.push_back(attend_one(q.select(0, batch), current_k, current_v, causal, softmax_scale));
  }
  if (append) {
    cache_seqlens.add_(k_new.size(1));
  }
  return outputs.empty() ? q.clone() : at::stack(outputs, 0);
}

}  // namespace mxflashattn
