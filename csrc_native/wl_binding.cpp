#include <torch/extension.h>

extern "C" int wl_probe_launch(const float *in, float *oB, float *o1, float *o4,
                               float *o8, float *oS, void *st);
extern "C" int wl_cost_launch(const float *in, float *out, int iters, int variant,
                              void *st);

static std::vector<at::Tensor> probe(at::Tensor in, int64_t stream) {
  auto oB = at::empty({64}, in.options());
  auto o1 = at::empty({64}, in.options());
  auto o4 = at::empty({64}, in.options());
  auto o8 = at::empty({64}, in.options());
  auto oS = at::empty({64}, in.options());
  wl_probe_launch((const float *)in.data_ptr(), (float *)oB.data_ptr(),
                  (float *)o1.data_ptr(), (float *)o4.data_ptr(),
                  (float *)o8.data_ptr(), (float *)oS.data_ptr(), (void *)stream);
  return {oB, o1, o4, o8, oS};
}

static at::Tensor cost(at::Tensor in, int64_t iters, int64_t variant, int64_t stream) {
  auto out = at::empty({64}, in.options());
  wl_cost_launch((const float *)in.data_ptr(), (float *)out.data_ptr(),
                 (int)iters, (int)variant, (void *)stream);
  return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("probe", &probe, "mov_shfl semantics probe");
  m.def("cost", &cost, "reduction cost benchmark: 0=bsm-xor 1=scan 2=noop");
}
