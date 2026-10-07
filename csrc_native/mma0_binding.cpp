// M0 probe binding
#include <torch/extension.h>

extern "C" void mma_probe_run(const void *A, const void *B, void *D, void *WS,
                              void *stream);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, mxmma0) {
  mxmma0.def("probe", [](int64_t A, int64_t B, int64_t D, int64_t WS,
                         int64_t stream) {
    mma_probe_run((const void *)A, (const void *)B, (void *)D, (void *)WS,
                  (void *)stream);
  });
}
