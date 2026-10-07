#include <cstdint>

/* 编译探测：MACA 是否支持 AMDGCN wave 级 builtin（AMD 原生跨 lane 指令）
 * TEST=1: wave_reduce_add_u32（单指令全 wave 归约，整数）
 * TEST=2: ds_swizzle_b32（单指令 XOR 蝶形交换，LDS 通路）
 * TEST=3: mov_dpp（DPP 数据并行原语） */
__global__ void wave_probe(const unsigned int *in, unsigned int *out) {
  const int t = threadIdx.x;
  unsigned int v = in[t];
#if TEST == 1
  unsigned int r = __builtin_amdgcn_wave_reduce_add_u32(v, false);
  out[t] = r;
#elif TEST == 2
  int r = __builtin_amdgcn_ds_swizzle_b32((int)v, 0x101F);
  out[t] = (unsigned int)r;
#elif TEST == 3
  int r = __builtin_amdgcn_mov_dpp((int)v, 0x110);
  out[t] = (unsigned int)r;
#endif
}
