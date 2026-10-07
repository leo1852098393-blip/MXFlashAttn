// probe binding: mxprobe
#include <torch/extension.h>

extern "C" void probe_stream(const void *src, void *sink, int n4, int grid, int block, void *stream);
extern "C" void probe_fma_chain(void *sink, int iters, int grid, int block, void *stream);
extern "C" void probe_fma_par(void *sink, int iters, int grid, int block, void *stream);
extern "C" void probe_expf(void *sink, int iters, int grid, int block, void *stream);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("probe_stream", [](int64_t src, int64_t sink, int64_t n4, int64_t grid, int64_t block, int64_t stream) {
    probe_stream((const void *)src, (void *)sink, (int)n4, (int)grid, (int)block, (void *)stream);
  });
  m.def("probe_fma_chain", [](int64_t sink, int64_t iters, int64_t grid, int64_t block, int64_t stream) {
    probe_fma_chain((void *)sink, (int)iters, (int)grid, (int)block, (void *)stream);
  });
  m.def("probe_fma_par", [](int64_t sink, int64_t iters, int64_t grid, int64_t block, int64_t stream) {
    probe_fma_par((void *)sink, (int)iters, (int)grid, (int)block, (void *)stream);
  });
  m.def("probe_expf", [](int64_t sink, int64_t iters, int64_t grid, int64_t block, int64_t stream) {
    probe_expf((void *)sink, (int)iters, (int)grid, (int)block, (void *)stream);
  });
}
