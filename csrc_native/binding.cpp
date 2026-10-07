// pybind11 binding: mxfadecode.paged_decode(q, k_cache, v_cache, block_table, seq_lens, scale)
#include <torch/extension.h>

extern "C" void mxfa_paged_decode_launch(const void *q, const void *k, const void *v,
                                         const void *bt, const void *sl, void *out,
                                         float scale, int B, int HQ, int HKV, int D,
                                         int PS, int BTW, int max_sl_bound, int bf16, void *stream);

static torch::Tensor paged_decode(at::Tensor q, at::Tensor k, at::Tensor v,
                                  at::Tensor bt, at::Tensor sl, double scale,
                                  int64_t max_sl_bound, int64_t stream) {
  TORCH_CHECK(q.is_cuda() && k.is_cuda() && v.is_cuda(), "tensors must be on device");
  TORCH_CHECK((q.scalar_type() == at::kHalf || q.scalar_type() == at::kBFloat16) &&
                  q.scalar_type() == k.scalar_type() && q.scalar_type() == v.scalar_type() &&
                  q.is_contiguous() && k.is_contiguous() &&
                  v.is_contiguous() && bt.is_contiguous() && sl.is_contiguous() &&
                  sl.scalar_type() == at::kInt && bt.scalar_type() == at::kInt,
              "dtype/layout check failed");
  TORCH_CHECK(q.dim() == 4 && q.size(1) == 1, "q must be [B, 1, HQ, D]");

  int64_t B   = q.size(0);
  int64_t HQ  = q.size(2);
  int64_t D   = q.size(3);
  int64_t HKV = k.size(2);
  int64_t PS  = k.size(1);
  TORCH_CHECK(k.dim() == 4 && k.size(0) == v.size(0) && k.size(3) == D, "kv cache shape");
  TORCH_CHECK(bt.dim() == 2 && bt.size(0) == B, "block_table [B, BTW]");
  TORCH_CHECK(sl.numel() == B, "seq_lens [B]");

  int64_t BTW   = bt.size(1);
  TORCH_CHECK(max_sl_bound >= 1 && max_sl_bound <= 65536, "max_sl_bound out of range");
  // No device-to-host sync anywhere on this path: it must stay legal under
  // CUDA graph capture.  Shared-memory sizing uses the host-side static bound.

  auto out = at::empty({B, 1, HQ, D}, q.options());
  int bf16 = (q.scalar_type() == at::kBFloat16) ? 1 : 0;
  mxfa_paged_decode_launch(q.data_ptr(), k.data_ptr(), v.data_ptr(),
                           bt.data_ptr(), sl.data_ptr(), out.data_ptr(),
                           (float)scale, (int)B, (int)HQ, (int)HKV, (int)D,
                           (int)PS, (int)BTW, (int)max_sl_bound, bf16, (void *)stream);
  return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("paged_decode", &paged_decode,
        "paged-KV decode attention (native MACA kernel)");
}
