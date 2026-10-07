#!/bin/bash
set -x
export MACA_PATH=/opt/maca
SRC=/root/MXFlashAttn_v1.3.0-kernel/csrc_native
MXCC=/opt/maca/mxgpu_llvm/bin/mxcc
cd "$SRC" || exit 1
"$MXCC" -x maca -offload-arch native -fPIC -O2 -std=c++17 -c mma0_kernel.cpp -o mma0_kernel.o 2>&1 | grep -iE "error" | head -8
"$MXCC" -fPIC -O2 -std=c++17 -c mma0_binding.cpp -o mma0_binding.o \
  -I/opt/conda/lib/python3.10/site-packages/torch/include -I/opt/conda/lib/python3.10/site-packages/torch/include/torch/csrc/api/include -I/opt/conda/include/python3.10 \
  -DTORCH_EXTENSION_NAME=mxmma0 -DTORCH_API_INCLUDE_EXTENSION_H -DUSE_MACA -DNV_ARCH_A100 2>&1 | grep -iE "error" | head -8
"$MXCC" -shared -fPIC mma0_kernel.o mma0_binding.o \
  -L/opt/conda/lib/python3.10/site-packages/torch/lib -lc10 -lc10_cuda -ltorch -ltorch_cpu -ltorch_python -o mxmma0.so 2>&1 | grep -iE "error" | head -8
ls -la mxmma0.so 2>/dev/null
echo M0_BUILD_RC=$?
