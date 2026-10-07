#!/bin/bash
set -x
export MACA_PATH=/opt/maca
SRC=/root/MXFlashAttn_v1.3.0-kernel/csrc_native
TORCH=/opt/conda/lib/python3.10/site-packages/torch
MXCC=/opt/maca/mxgpu_llvm/bin/mxcc
cd "$SRC" || exit 1
"$MXCC" -x maca -offload-arch native -fPIC -O3 -std=c++17 -c probe_kernel.cpp -o probe_kernel.o 2>&1 | grep -iE "error" | head -5
"$MXCC" -fPIC -O3 -std=c++17 -c probe_binding.cpp -o probe_binding.o \
  -I"$TORCH/include" -I"$TORCH/include/torch/csrc/api/include" -I/opt/conda/include/python3.10 \
  -DTORCH_EXTENSION_NAME=mxprobe -DTORCH_API_INCLUDE_EXTENSION_H -DUSE_MACA -DNV_ARCH_A100 2>&1 | grep -iE "error" | head -5
"$MXCC" -shared -fPIC probe_kernel.o probe_binding.o \
  -L"$TORCH/lib" -lc10 -lc10_cuda -ltorch -ltorch_cpu -ltorch_python -o mxprobe.so 2>&1 | grep -iE "error" | head -5
ls -la mxprobe.so 2>/dev/null
echo PROBE_BUILD_RC=$?
