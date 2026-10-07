// M1 binding: mxmmain.paged_decode(...) — 与 mxfadecode 接口一致
#include <torch/extension.h>
#include <cstdlib>

extern "C" int mxfa_paged_decode_m1_launch(const void *q, const void *k, const void *v,
                                           const void *bt, const void *sl, void *out,
                                           float scale, int B, int HQ, int HKV, int D,
                                           int PS, int BTW, int max_sl, int bf16,
                                           int mode, void *stream);

extern "C" int mxfa_shfl_probe(const int *in, int *out, void *stream);

extern "C" int mxfa_stream_probe(const void *k, const void *v, const void *bt,
                                 const void *sl, void *out, int B, int HKV, int HD,
                                 int PS, int variant, void *stream);

extern "C" int mxfa_opcost_probe(const float *in, float *out, int iters, int solo,
                                 void *stream);

static torch::Tensor opcost_probe(at::Tensor in, int64_t iters, int64_t solo,
                                  int64_t stream) {
  auto out = at::zeros({5 * 256}, in.options());
  mxfa_opcost_probe((const float *)in.data_ptr(), (float *)out.data_ptr(),
                    (int)iters, (int)solo, (void *)stream);
  return out;
}

static std::vector<torch::Tensor> stream_probe(at::Tensor k, at::Tensor v, at::Tensor bt,
                                               at::Tensor sl, int64_t variant,
                                               int64_t stream) {
  int64_t B = bt.size(0), HKV = k.size(2), HD = k.size(3), PS = k.size(1);
  auto out = at::zeros({B, HKV, 32}, k.options().dtype(at::kFloat));
  mxfa_stream_probe(k.data_ptr(), v.data_ptr(), bt.data_ptr(), sl.data_ptr(),
                    out.data_ptr(), (int)B, (int)HKV, (int)HD, (int)PS, (int)variant,
                    (void *)stream);
  return {out};
}

extern "C" int mxfa_paged_decode_split_launch(const void *q, const void *k, const void *v,
                                              const void *bt, const void *sl, void *out,
                                              void *ows, void *mdws, float scale,
                                              int B, int HQ, int HKV, int D,
                                              int PS, int BTW, int bf16,
                                              int NP, int c_len, int um, void *stream);

static torch::Tensor shfl_probe(at::Tensor in, int64_t stream) {
  auto out = at::empty({64, 4}, in.options());
  mxfa_shfl_probe((const int *)in.data_ptr(), (int *)out.data_ptr(), (void *)stream);
  return out;
}

static torch::Tensor paged_decode(at::Tensor q, at::Tensor k, at::Tensor v,
                                  at::Tensor bt, at::Tensor sl, double scale,
                                  int64_t max_sl_bound, int64_t stream, int64_t mode) {
  TORCH_CHECK(q.is_cuda() && k.is_cuda() && v.is_cuda(), "tensors must be on device");
  TORCH_CHECK((q.scalar_type() == at::kHalf || q.scalar_type() == at::kBFloat16) &&
                  q.scalar_type() == k.scalar_type() && q.scalar_type() == v.scalar_type() &&
                  q.is_contiguous() && k.is_contiguous() && v.is_contiguous() &&
                  bt.is_contiguous() && sl.is_contiguous() &&
                  sl.scalar_type() == at::kInt && bt.scalar_type() == at::kInt,
              "dtype/layout check failed");
  TORCH_CHECK(q.dim() == 4 && q.size(1) == 1, "q must be [B, 1, HQ, D]");

  int64_t B = q.size(0), HQ = q.size(2), D = q.size(3);
  int64_t HKV = k.size(2), PS = k.size(1);
  int64_t BTW = bt.size(1);

  auto out = at::empty({B, 1, HQ, D}, q.options());
  int bf16 = (q.scalar_type() == at::kBFloat16) ? 1 : 0;
  int rc = mxfa_paged_decode_m1_launch(
      q.data_ptr(), k.data_ptr(), v.data_ptr(), bt.data_ptr(), sl.data_ptr(),
      out.data_ptr(), (float)scale, (int)B, (int)HQ, (int)HKV, (int)D, (int)PS,
      (int)BTW, (int)max_sl_bound, bf16, (int)mode, (void *)stream);
  TORCH_CHECK(rc == 0, "m1 kernel unsupported config: D=", D, " GQ=", HQ / HKV);
  return out;
}

static torch::Tensor paged_decode_split(at::Tensor q, at::Tensor k, at::Tensor v,
                                        at::Tensor bt, at::Tensor sl, double scale,
                                        int64_t stream, int64_t np, int64_t c_len) {
  TORCH_CHECK(q.is_cuda() && k.is_cuda() && v.is_cuda(), "tensors must be on device");
  TORCH_CHECK((q.scalar_type() == at::kHalf || q.scalar_type() == at::kBFloat16) &&
                  q.scalar_type() == k.scalar_type() && q.scalar_type() == v.scalar_type() &&
                  q.is_contiguous() && k.is_contiguous() && v.is_contiguous() &&
                  bt.is_contiguous() && sl.is_contiguous() &&
                  sl.scalar_type() == at::kInt && bt.scalar_type() == at::kInt,
              "dtype/layout check failed");
  TORCH_CHECK(q.dim() == 4 && q.size(1) == 1, "q must be [B, 1, HQ, D]");

  int64_t B = q.size(0), HQ = q.size(2), D = q.size(3);
  int64_t HKV = k.size(2), PS = k.size(1);
  int64_t BTW = bt.size(1);

  auto opts_f = q.options().dtype(at::kFloat);
  auto ows = at::empty({B, HQ, np, D}, opts_f);
  auto mdws = at::empty({B, HQ, np, 2}, opts_f);
  auto out = at::empty({B, 1, HQ, D}, q.options());
  int bf16 = (q.scalar_type() == at::kBFloat16) ? 1 : 0;
  /* 统一 max 开关：MXFA_M1_UM 环境变量（每次调用读取）。
   * 0=online softmax；1=φ=0 统一 max（FlashDecoding++ 原式，|s|<126）；2=分桶惰性缩放。
   * 默认 2：与 1 性能持平（差距 <1%）且无溢出边界（mode 11，全配置含 ovf PASS）。 */
  int um = 2;
  const char *e = getenv("MXFA_M1_UM");
  if (e) {
    int v = atoi(e);
    if (v == 0 || v == 1) um = v;
  }
  int rc = mxfa_paged_decode_split_launch(
      q.data_ptr(), k.data_ptr(), v.data_ptr(), bt.data_ptr(), sl.data_ptr(),
      out.data_ptr(), ows.data_ptr(), mdws.data_ptr(), (float)scale,
      (int)B, (int)HQ, (int)HKV, (int)D, (int)PS, (int)BTW, bf16,
      (int)np, (int)c_len, um, (void *)stream);
  TORCH_CHECK(rc == 0, "split kernel unsupported config: D=", D, " GQ=", HQ / HKV);
  return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("paged_decode", &paged_decode, "paged-KV decode (M1 cp_async pipeline)");
  m.def("paged_decode_split", &paged_decode_split,
        "split-KV paged decode: np chunks + combine, c_len = per-chunk position length");
  m.def("shfl_probe", &shfl_probe, "shuffle behavior probe (64-lane warp)");
  m.def("stream_probe", &stream_probe,
        "paged streaming bandwidth probe: variant 0=cp_async+counted, 1=cp_async+wait-all, 2=plain loads");
  m.def("opcost_probe", &opcost_probe,
        "instruction cost probe: out rows 0=empty 1=fma 2=shuffle 3=exp2 4=shuffle+exp2");
}
