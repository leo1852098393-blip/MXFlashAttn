/**
 * M0 probe: __builtin_mxc_mma_16x16x16f16 fragment-layout discovery.
 *
 * 依据（远端 SDK 实证）:
 *  - cute/arch/mma_sm80.hpp: MACA_16x16x16_F32F16F16F32, VectorType=__NATIVE_VECTOR__(2,uint32_t),
 *    builtin(b, a, {c0..c3}) -> 4 floats
 *  - cute/atom/mma_traits_sm80.hpp: Shape_MNK=(16,16,16), ThrID=Layout<_64>  (64 线程!)
 *    ALayout==BLayout==CLayout = Layout<Shape<Shape<_16,_4>,_4>, Stride<Stride<_1,_64>,_16>>
 *  预测（待本探针实证）: 线程 t 的第 v 个半字 <-> 行=4*(t/16)+v, 列=t%16（或转置变体）。
 *
 * 本内核 64 线程一块，一次跑 32 种 (aMode x bMode x pkA x pkB x argSwap) 假设，
 * 每种写两种输出排布(o0/o1)，host 端与 4 个 numpy 参考取最接近者。
 */
#include <mc_runtime.h>
#include <stdint.h>

using V2 = __NATIVE_VECTOR__(2, uint32_t);

__device__ __forceinline__ uint32_t pk2(uint16_t lo, uint16_t hi) {
  return (uint32_t)lo | ((uint32_t)hi << 16);
}

__global__ void mma_probe_kernel(const uint16_t *A, const uint16_t *B,
                                 float *D, int *WS) {
  const int t = threadIdx.x;   /* 0..63 */
  const int q = t >> 4;        /* 0..3  */
  const int c = t & 15;        /* 0..15 */
  if (WS && t == 0) WS[0] = (int)warpSize;

  /* A fragment 寄存器: ai=0: slot(t,v)=A[4q+v][c]; ai=1: slot(t,v)=A[c][4q+v] */
  uint32_t ar[2][2][2]; /* [ai][pk][rg] */
  uint32_t br[2][2][2];
#pragma unroll
  for (int ai = 0; ai < 2; ++ai) {
    uint16_t v0, v1, v2, v3;
    if (ai == 0) {
      v0 = A[(4 * q + 0) * 16 + c];
      v1 = A[(4 * q + 1) * 16 + c];
      v2 = A[(4 * q + 2) * 16 + c];
      v3 = A[(4 * q + 3) * 16 + c];
    } else {
      v0 = A[c * 16 + 4 * q + 0];
      v1 = A[c * 16 + 4 * q + 1];
      v2 = A[c * 16 + 4 * q + 2];
      v3 = A[c * 16 + 4 * q + 3];
    }
    ar[ai][0][0] = pk2(v0, v1);
    ar[ai][0][1] = pk2(v2, v3);
    ar[ai][1][0] = pk2(v1, v0);
    ar[ai][1][1] = pk2(v3, v2);
  }
#pragma unroll
  for (int bi = 0; bi < 2; ++bi) {
    uint16_t v0, v1, v2, v3;
    if (bi == 0) {
      v0 = B[(4 * q + 0) * 16 + c];
      v1 = B[(4 * q + 1) * 16 + c];
      v2 = B[(4 * q + 2) * 16 + c];
      v3 = B[(4 * q + 3) * 16 + c];
    } else {
      v0 = B[c * 16 + 4 * q + 0];
      v1 = B[c * 16 + 4 * q + 1];
      v2 = B[c * 16 + 4 * q + 2];
      v3 = B[c * 16 + 4 * q + 3];
    }
    br[bi][0][0] = pk2(v0, v1);
    br[bi][0][1] = pk2(v2, v3);
    br[bi][1][0] = pk2(v1, v0);
    br[bi][1][1] = pk2(v3, v2);
  }

  float acc[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
  for (int ai = 0; ai < 2; ++ai)
  for (int bi = 0; bi < 2; ++bi)
  for (int pa = 0; pa < 2; ++pa)
  for (int pb = 0; pb < 2; ++pb)
  for (int sw = 0; sw < 2; ++sw) {
    V2 av = {ar[ai][pa][0], ar[ai][pa][1]};
    V2 bv = {br[bi][pb][0], br[bi][pb][1]};
    auto r = sw ? __builtin_mxc_mma_16x16x16f16(av, bv,
                 {acc[0], acc[1], acc[2], acc[3]})
                : __builtin_mxc_mma_16x16x16f16(bv, av,
                 {acc[0], acc[1], acc[2], acc[3]});
    float d[4] = {(float)r[0], (float)r[1], (float)r[2], (float)r[3]};
    const int idx = ((((ai * 2 + bi) * 2 + pa) * 2 + pb) * 2 + sw);
    float *o0 = D + idx * 512;        /* outMode0: D[4q+j][c] */
    float *o1 = D + idx * 512 + 256;  /* outMode1: D[c][4q+j] */
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      o0[(4 * q + j) * 16 + c] = d[j];
      o1[c * 16 + 4 * q + j] = d[j];
    }
  }
}

extern "C" void mma_probe_run(const void *A, const void *B, void *D, void *WS,
                              void *stream) {
  mma_probe_kernel<<<1, 64, 0, (mcStream_t)stream>>>(
      (const uint16_t *)A, (const uint16_t *)B, (float *)D, (int *)WS);
}
