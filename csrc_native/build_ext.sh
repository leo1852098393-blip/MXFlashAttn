#!/bin/bash
# Gate 2: build torch extension (host: g++, device+link: mxcc)
set -x
export MACA_PATH=/opt/maca
SRC=/root/MXFlashAttn_v1.3.0-kernel/csrc_native
TORCH=/opt/conda/lib/python3.10/site-packages/torch
MXCC=/opt/maca/mxgpu_llvm/bin/mxcc
cd "$SRC" || exit 1

"$MXCC" -x maca -offload-arch native -fPIC -O2 -std=c++17 -c dec_kernel.cpp -o dec_kernel.o
RC1=$?
"$MXCC" -fPIC -O2 -std=c++17 -c binding.cpp -o binding.o \
  -I"$TORCH/include" -I"$TORCH/include/torch/csrc/api/include" -I/opt/conda/include/python3.10 \
  -DTORCH_EXTENSION_NAME=mxfadecode -DTORCH_API_INCLUDE_EXTENSION_H -DUSE_MACA -DNV_ARCH_A100
RC2=$?
if [ $RC1 -ne 0 ] || [ $RC2 -ne 0 ]; then echo "COMPILE_FAIL rc=$RC1/$RC2"; exit 1; fi

"$MXCC" -shared -fPIC dec_kernel.o binding.o \
  -L"$TORCH/lib" -lc10 -lc10_cuda -ltorch -ltorch_cpu -ltorch_python -o mxfadecode.so
RC3=$?
if [ $RC3 -ne 0 ]; then
  echo "SIMPLE_LINK_FAIL, retry with maca-link flags"
  "$MXCC" -shared -fPIC -DNV_ARCH_A100 -Xdevice -D__CUDA_ARCH__=800 \
    -forward-unknown-to-compiler -D__XCORE_WN__ -fgpu-rdc --maca-link \
    -lToolsExt_cu -lruntime_cu -lmcToolsExt \
    dec_kernel.o binding.o -L"$TORCH/lib" -lc10 -lc10_cuda -ltorch -ltorch_cpu -ltorch_python \
    -o mxfadecode.so
  RC3=$?
fi
ls -la mxfadecode.so 2>/dev/null
echo "BUILD_RC=$RC3"
