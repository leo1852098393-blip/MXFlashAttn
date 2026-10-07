#include <cstdint>

/* 16-lane 寄存器级 shuffle（mov_shfl）语义与成本探针
 * A. __shfl_sync_16 广播方向/组界验证
 * B. __shfl_up_sync_16 方向与边界行为
 * C. scan+broadcast 归约正确性
 * D. 成本对比：4x __shfl_xor_sync(BSM) vs scan+广播(mov_shfl) vs 空转 */

#define WLMASK ((unsigned long long)-1)

__global__ void wl_probe(const float *in, float *outB, float *outU1, float *outU4,
                         float *outU8, float *outScan) {
  const int lane = threadIdx.x;             /* 64 线程 = 1 个 wave */
  const float v = in[lane];
  outB[lane]  = __shfl_sync_16(WLMASK, v, 3);
  outU1[lane] = __shfl_up_sync_16(WLMASK, v, 1);
  outU4[lane] = __shfl_up_sync_16(WLMASK, v, 4);
  outU8[lane] = __shfl_up_sync_16(WLMASK, v, 8);

  /* 16-lane 组内 Hillis-Steele 归约：4 步 scan + 1 次广播 = 5 条 mov_shfl */
  const int gid = lane & 15;
  float x = v;
#pragma unroll
  for (int d = 1; d < 16; d <<= 1) {
    float o = __shfl_up_sync_16(WLMASK, x, d);
    if (gid >= d) x += o;
  }
  outScan[lane] = __shfl_sync_16(WLMASK, x, 15);
}

__global__ void wl_cost(const float *in, float *out, int iters, int variant) {
  const int lane = threadIdx.x;
  const int gid = lane & 15;
  float v = in[lane];
  for (int i = 0; i < iters; ++i) {
    if (variant == 0) {
      /* 4 次 BSM shuffle 蝶形归约（当前正式路径） */
      float s = v;
#pragma unroll
      for (int off = 8; off > 0; off >>= 1) s += __shfl_xor_sync(WLMASK, s, off);
      v = s * 1e-9f + v * 0.5f;
    } else if (variant == 1) {
      /* scan + 广播（寄存器级 mov_shfl） */
      float x = v;
#pragma unroll
      for (int d = 1; d < 16; d <<= 1) {
        float o = __shfl_up_sync_16(WLMASK, x, d);
        if (gid >= d) x += o;
      }
      x = __shfl_sync_16(WLMASK, x, 15);
      v = x * 1e-9f + v * 0.5f;
    } else {
      v = v * 0.5f + 1.f;                   /* 空转对照 */
    }
  }
  out[lane] = v;
}

extern "C" int wl_probe_launch(const float *in, float *oB, float *o1, float *o4,
                               float *o8, float *oS, void *st) {
  wl_probe<<<1, 64, 0, (mcStream_t)st>>>(in, oB, o1, o4, o8, oS);
  return 0;
}

extern "C" int wl_cost_launch(const float *in, float *out, int iters, int variant,
                              void *st) {
  wl_cost<<<1, 64, 0, (mcStream_t)st>>>(in, out, iters, variant);
  return 0;
}
