/**
 * 平台体检探针 — MACA/C500 基础吞吐测试
 * 1. stream_read: grid-stride 流式读 float4 → 有效显存带宽
 * 2. fma_chain: 依赖 FMA 长链 → 每时钟指令数 / 有效时钟
 * 3. expf_loop: __expf 密集循环 → expf 吞吐
 * 4. (对照用) 无依赖并行 FMA
 */
#include <mc_runtime.h>
#include <math.h>
#include <string.h>

__global__ void stream_read(const float4 *__restrict__ src, float *__restrict__ sink, int n4) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = gridDim.x * blockDim.x;
  float acc = 0.0f;
  for (; i < n4; i += stride) {
    float4 v = src[i];
    acc += v.x + v.y + v.z + v.w;
  }
  if (acc == 12345.678f) sink[0] = acc;   /* 防 DCE */
}

__global__ void fma_chain(float *__restrict__ sink, int iters) {
  float a = threadIdx.x * 1e-9f + 1.0f;
  float b = 1.0000001f;
  for (int i = 0; i < iters; ++i) a = a * b + b;   /* 串行依赖 */
  if (a == 12345.678f) sink[0] = a;
}

__global__ void fma_par(float *__restrict__ sink, int iters) {
  float a0 = threadIdx.x * 1e-9f + 1.0f, a1 = a0 + 1, a2 = a0 + 2, a3 = a0 + 3;
  float b = 1.0000001f;
  for (int i = 0; i < iters; i += 4) {          /* 4 路无依赖 ILP */
    a0 = a0 * b + b; a1 = a1 * b + b; a2 = a2 * b + b; a3 = a3 * b + b;
  }
  float s = a0 + a1 + a2 + a3;
  if (s == 12345.678f) sink[0] = s;
}

__global__ void expf_loop(float *__restrict__ sink, int iters) {
  float a = threadIdx.x * 1e-9f + 0.5f;
  float s = 0.0f;
  for (int i = 0; i < iters; ++i) { s += __expf(a * 1e-6f); a += 1e-7f; }
  if (s == 12345.678f) sink[0] = s;
}

extern "C" void probe_stream(const void *src, void *sink, int n4, int grid, int block, void *stream) {
  stream_read<<<grid, block, 0, (mcStream_t)stream>>>((const float4 *)src, (float *)sink, n4);
}
extern "C" void probe_fma_chain(void *sink, int iters, int grid, int block, void *stream) {
  fma_chain<<<grid, block, 0, (mcStream_t)stream>>>((float *)sink, iters);
}
extern "C" void probe_fma_par(void *sink, int iters, int grid, int block, void *stream) {
  fma_par<<<grid, block, 0, (mcStream_t)stream>>>((float *)sink, iters);
}
extern "C" void probe_expf(void *sink, int iters, int grid, int block, void *stream) {
  expf_loop<<<grid, block, 0, (mcStream_t)stream>>>((float *)sink, iters);
}
