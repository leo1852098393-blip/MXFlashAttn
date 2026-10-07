#!/bin/bash
set -x
export MACA_PATH=/opt/maca
SRC=/root/MXFlashAttn_v1.3.0-kernel/csrc_native
MXCC=/opt/maca/mxgpu_llvm/bin/mxcc
cd "$SRC" || exit 1
rm -f m1_kernel.o m1_binding.o mxmmain.so
"$MXCC" -x maca -offload-arch native -fPIC -O3 -std=c++17 -c m1_kernel.cpp -o m1_kernel.o 2>&1 | tee /tmp/m1_k.log | grep -iE "error|warning" | head -12
RC1=${PIPESTATUS[0]}
[ "$RC1" -ne 0 ] && { echo KERNEL_COMPILE_FAIL RC=$RC1; exit 1; }
"$MXCC" -fPIC -O2 -std=c++17 -c m1_binding.cpp -o m1_binding.o \
  -I/opt/conda/lib/python3.10/site-packages/torch/include -I/opt/conda/lib/python3.10/site-packages/torch/include/torch/csrc/api/include -I/opt/conda/include/python3.10 \
  -DTORCH_EXTENSION_NAME=mxmmain -DTORCH_API_INCLUDE_EXTENSION_H -DUSE_MACA -DNV_ARCH_A100 2>&1 | tee /tmp/m1_b.log | grep -iE "error|warning" | head -12
RC2=${PIPESTATUS[0]}
[ "$RC2" -ne 0 ] && { echo BINDING_COMPILE_FAIL RC=$RC2; exit 1; }
"$MXCC" -shared -fPIC m1_kernel.o m1_binding.o \
  -L/opt/conda/lib/python3.10/site-packages/torch/lib -lc10 -lc10_cuda -ltorch -ltorch_cpu -ltorch_python -o mxmmain.so 2>&1 | grep -iE "error|warning" | head -12
RC3=${PIPESTATUS[0]}
[ "$RC3" -ne 0 ] && { echo LINK_FAIL RC=$RC3; exit 1; }
ls -la mxmmain.so
echo M1_BUILD_OK
