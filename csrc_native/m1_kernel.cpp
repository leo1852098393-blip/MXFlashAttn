/**
 * m1_kernel.cpp — Gate 4 M1: paged decode（flashinfer 结构 + 寄存器流水优化）
 *
 * mode 0（寄存器流水，推荐）：tz 块的 GQ 个 K/V 行以 uint4 寄存器持有，
 *   head 间用 warp shuffle 交换（要求 BDX*GQ 整除 64，tz 块不跨 warp），
 *   循环内零栅栏、零 smem；iter 开头提前发出下一 iter 的加载以重叠延迟。
 * mode 1（bsm 路径）：vendor 式 smem 暂存 + 栅栏（GQ=3/5/6/7 等 shuffle 不适用的布局）。
 * softmax 全程 log2 域；tile 级 max 一次缩放；bdx 段内 shuffle 归约；跨 tz 合并经 smem。
 */
#include <mc_runtime.h>
#include <math.h>
#include <string.h>

/* ---- fp16/bf16 <-> fp32 ---- */
static __device__ __forceinline__ float dev_bf2f(unsigned short h) {
  unsigned int bits = ((unsigned int)h) << 16;
  float f; memcpy(&f, &bits, 4); return f;
}
static __device__ __forceinline__ unsigned short dev_f2bf(float f) {
  unsigned int x; memcpy(&x, &f, 4);
  unsigned int lsb = (x >> 16) & 1u;
  x += 0x7FFFu + lsb;
  return (unsigned short)(x >> 16);
}
/* f16 -> f32 快路径：normals/inf/nan 精确，denormal 清零（误差 < 6.1e-5） */
static __device__ __forceinline__ float fast_h2f(unsigned short h) {
  unsigned int bits = ((unsigned int)(h & 0x8000u)) << 16;
  unsigned int expv = (h >> 10) & 0x1Fu;
  bits |= ((expv + 112u) << 23) | ((unsigned int)(h & 0x3FFu) << 13);
  float f; memcpy(&f, &bits, 4);
  return expv ? f : 0.f;
}
static __device__ __forceinline__ unsigned short dev_f2h(float f) {
  unsigned int x; memcpy(&x, &f, 4);
  unsigned int sign = (x >> 16) & 0x8000u;
  int e = (int)((x >> 23) & 0xFFu) - 127 + 15;
  unsigned int m = x & 0x7FFFFFu;
  if (((x >> 23) & 0xFFu) == 0xFF)
    return (unsigned short)(sign | 0x7C00u | (m ? 0x200u : 0u));
  if (e >= 31) return (unsigned short)(sign | 0x7C00u);
  if (e <= 0) {
    if (e < -10) return (unsigned short)sign;
    m |= 0x800000u;
    int sh = 14 - e;
    unsigned int man = m >> sh;
    unsigned int rem = m & ((1u << sh) - 1u);
    unsigned int half = 1u << (sh - 1);
    if (rem > half || (rem == half && (man & 1u))) man++;
    return (unsigned short)(sign | man);
  }
  unsigned int man = m >> 13;
  unsigned int rem = m & 0x1FFFu;
  if (rem > 0x1000u || (rem == 0x1000u && (man & 1u))) man++;
  if (man == 0x400u) { man = 0; e++; if (e >= 31) return (unsigned short)(sign | 0x7C00u); }
  return (unsigned short)(sign | ((unsigned int)e << 10) | man);
}
static __device__ __forceinline__ float cvt_fast(unsigned short h, int bf16) {
  return bf16 ? dev_bf2f(h) : fast_h2f(h);
}
static __device__ __forceinline__ unsigned short cvt_out(float f, int bf16) {
  return bf16 ? dev_f2bf(f) : dev_f2h(f);
}

typedef __NATIVE_VECTOR__(4, int) V4;
static __device__ __forceinline__ void bsm128(void *smem_dst, const void *gmem_src, bool pred) {
  __builtin_mxc_ldg_b128_bsm_predicator((V4 *)smem_dst, (V4 *)gmem_src, 0, true, true,
                                        false, true, pred, 1, MACA_ICMP_EQ);
}
static __device__ __forceinline__ float shfl_xor_f(float x, int mask) {
  return __shfl_xor_sync((unsigned long long)-1, x, mask);
}
/* 16-lane 组内寄存器级 shuffle（__builtin_mxc_mov_shfl，~16ns/条），
 * 替代 __shfl_xor_sync（底层 bsm_bpermute，LDS 往返 ~66ns/条） */
static __device__ __forceinline__ float shfl_up_g(float x, int delta) {
  return __shfl_up_sync_16((unsigned long long)-1, x, delta);
}
static __device__ __forceinline__ float shfl_bcast_g(float x, int src) {
  return __shfl_sync_16((unsigned long long)-1, x, src);
}
/* 内联 fast_exp2：MACA CUDA 层的 exp2f 降级为 __nv_exp2f libdevice 外部调用，
 * 调用打断软件流水（实测 exp2f+跨lane 交互慢 3.8×）。6 项多项式 + 指数位拼接，
 * 截断误差 ~1.5e-6，远低于 f16 输出分辨率。 */
static __device__ __forceinline__ float exp2_fast(float x) {
  if (x < -126.f) return 0.f;
  if (x > 126.f) return 3.4e38f;
  const float e = floorf(x);
  const float f = x - e;                  /* [0,1) */
  float p = 1.f + f * (0.69314718056f + f * (0.24022650696f + f *
             (0.05550410866f + f * (0.00961812911f + f * (0.00133335581f +
               f * 0.00015403530f)))));
  union { float f32; int i32; } sc;
  sc.i32 = ((int)e + 127) << 23;
  return p * sc.f32;
}

/* ---- M1 mode 0：寄存器流水内核 ---- */
template <int HD, int GQ>
__global__ void paged_decode_reg(const unsigned short *__restrict__ Q,
                                 const unsigned short *__restrict__ Kp,
                                 const unsigned short *__restrict__ Vp,
                                 const int *__restrict__ bt,
                                 const int *__restrict__ sl_ptr,
                                 unsigned short *__restrict__ out,
                                 float scale, int HQ, int HKV, int PS, int BTW, int bf16) {
  constexpr int VEC = 8;
  constexpr int BDX = HD / VEC;
  constexpr int NTH = (GQ == 8) ? 512 : 256;
  constexpr int BDZ = NTH / (BDX * GQ);
  constexpr int ROWS = BDZ * GQ;
  constexpr float LOG2E = 1.4426950408889634f;
  constexpr unsigned long long FMASK = (unsigned long long)-1;

  const int b = blockIdx.x, kvh = blockIdx.y;
  const int tx = threadIdx.x, ty = threadIdx.y, tz = threadIdx.z;
  const int sl = sl_ptr[b];
  const int row = tz * GQ + ty;                       /* 本线程持有的 K/V 行 */
  const int qo_head = kvh * GQ + ty;
  const int lane = (tx + BDX * ty + BDX * GQ * tz) & 63;

  /* 合并阶段用的 smem（循环外） */
  extern __shared__ float smem_f[];
  float *osm = smem_f;                                /* ROWS*HD floats */
  float *mdm = smem_f + ROWS * HD;                    /* 2*GQ*BDZ floats */

  float q_vec[VEC];
  {
    const unsigned short *qp = Q + ((size_t)b * HQ + qo_head) * HD + tx * VEC;
#pragma unroll
    for (int i = 0; i < VEC; ++i) q_vec[i] = cvt_fast(qp[i], bf16);
  }

  float m = -INFINITY, d = 0.f, o[VEC], s[GQ];
#pragma unroll
  for (int i = 0; i < VEC; ++i) o[i] = 0.f;

  if (sl > 0) {
    const float slog2 = scale * LOG2E;
    const size_t btw_b = (size_t)b * BTW;

    uint4 k_cur = {0u, 0u, 0u, 0u}, v_cur = {0u, 0u, 0u, 0u};
    uint4 k_next, v_next;
    {
      int pos = row;
      bool ok = pos < sl;
      int posc = ok ? pos : sl - 1;
      int pg = posc / PS; if (pg > BTW - 1) pg = BTW - 1;
      int phys = bt[btw_b + pg];
      size_t g = (((size_t)phys * PS + (posc % PS)) * HKV + kvh) * (size_t)HD + tx * 8;
      k_next = ok ? *(const uint4 *)(Kp + g) : k_cur;
      v_next = ok ? *(const uint4 *)(Vp + g) : v_cur;
    }

    const int iters = (sl + ROWS - 1) / ROWS;
#pragma unroll 1
    for (int iter = 0; iter < iters; ++iter) {
      k_cur = k_next;
      v_cur = v_next;
      { /* 提前发出下一 iter 的加载（延迟与计算重叠） */
        int pos = (iter + 1) * ROWS + row;
        bool ok = pos < sl;
        int posc = ok ? pos : sl - 1;
        int pg = posc / PS; if (pg > BTW - 1) pg = BTW - 1;
        int phys = bt[btw_b + pg];
        size_t g = (((size_t)phys * PS + (posc % PS)) * HKV + kvh) * (size_t)HD + tx * 8;
        k_next = ok ? *(const uint4 *)(Kp + g) : k_next;
        v_next = ok ? *(const uint4 *)(Vp + g) : v_next;
      }

      /* compute_qk：head ty 对块内 GQ 个位置打分；K 行经 xor shuffle 交换。
       * 关键：shuffle 必须全 warp 无条件执行（jj==ty 时 src==lane 取自身），
       * 放在 if (jj!=ty) 里会造成 divergence，__shfl_sync 返回未定义值。 */
      float m_prev = m;
#pragma unroll
      for (int jj = 0; jj < GQ; ++jj) {
        int src = lane ^ (BDX * (jj ^ ty));
        uint4 kj;
        kj.x = __shfl_xor_sync(FMASK, k_cur.x, src);
        kj.y = __shfl_xor_sync(FMASK, k_cur.y, src);
        kj.z = __shfl_xor_sync(FMASK, k_cur.z, src);
        kj.w = __shfl_xor_sync(FMASK, k_cur.w, src);
        float sacc = 0.f;
        sacc += q_vec[0] * cvt_fast((unsigned short)(kj.x & 0xFFFFu), bf16);
        sacc += q_vec[1] * cvt_fast((unsigned short)(kj.x >> 16), bf16);
        sacc += q_vec[2] * cvt_fast((unsigned short)(kj.y & 0xFFFFu), bf16);
        sacc += q_vec[3] * cvt_fast((unsigned short)(kj.y >> 16), bf16);
        sacc += q_vec[4] * cvt_fast((unsigned short)(kj.z & 0xFFFFu), bf16);
        sacc += q_vec[5] * cvt_fast((unsigned short)(kj.z >> 16), bf16);
        sacc += q_vec[6] * cvt_fast((unsigned short)(kj.w & 0xFFFFu), bf16);
        sacc += q_vec[7] * cvt_fast((unsigned short)(kj.w >> 16), bf16);
#pragma unroll
        for (int off = BDX / 2; off > 0; off >>= 1) sacc += shfl_xor_f(sacc, off);
        sacc *= slog2;
        if (iter * ROWS + tz * GQ + jj >= sl) sacc = -INFINITY;
        m = fmaxf(m, sacc);
        s[jj] = sacc;
      }
      float os = (m > -INFINITY) ? exp2f(m_prev - m) : 1.f;
      d *= os;
#pragma unroll
      for (int i = 0; i < VEC; ++i) o[i] *= os;
#pragma unroll
      for (int jj = 0; jj < GQ; ++jj) {
        float p = (s[jj] > -INFINITY) ? exp2f(s[jj] - m) : 0.f;
        d += p;
        s[jj] = p;
      }

      /* update：o += Σ p_jj · V_jj（V 行 xor shuffle 交换，无条件执行防 divergence） */
#pragma unroll
      for (int jj = 0; jj < GQ; ++jj) {
        int src = lane ^ (BDX * (jj ^ ty));
        uint4 vj;
        vj.x = __shfl_xor_sync(FMASK, v_cur.x, src);
        vj.y = __shfl_xor_sync(FMASK, v_cur.y, src);
        vj.z = __shfl_xor_sync(FMASK, v_cur.z, src);
        vj.w = __shfl_xor_sync(FMASK, v_cur.w, src);
        o[0] += s[jj] * cvt_fast((unsigned short)(vj.x & 0xFFFFu), bf16);
        o[1] += s[jj] * cvt_fast((unsigned short)(vj.x >> 16), bf16);
        o[2] += s[jj] * cvt_fast((unsigned short)(vj.y & 0xFFFFu), bf16);
        o[3] += s[jj] * cvt_fast((unsigned short)(vj.y >> 16), bf16);
        o[4] += s[jj] * cvt_fast((unsigned short)(vj.z & 0xFFFFu), bf16);
        o[5] += s[jj] * cvt_fast((unsigned short)(vj.z >> 16), bf16);
        o[6] += s[jj] * cvt_fast((unsigned short)(vj.w & 0xFFFFu), bf16);
        o[7] += s[jj] * cvt_fast((unsigned short)(vj.w >> 16), bf16);
      }
    }
  }

  /* 跨 tz 合并 */
  __syncthreads();
#pragma unroll
  for (int i = 0; i < VEC; ++i) osm[((size_t)tz * GQ + ty) * HD + tx * VEC + i] = o[i];
  mdm[(tz * GQ + ty) * 2] = m;
  mdm[(tz * GQ + ty) * 2 + 1] = d;
  __syncthreads();

  float m2 = -INFINITY, d2 = 0.f, o2[VEC];
#pragma unroll
  for (int i = 0; i < VEC; ++i) o2[i] = 0.f;
#pragma unroll
  for (int j = 0; j < BDZ; ++j) {
    int u = j * GQ + ty;
    float mz = mdm[u * 2], dz = mdm[u * 2 + 1];
    float mn = fmaxf(m2, mz);
    float f1 = (m2 > -INFINITY) ? exp2f(m2 - mn) : 0.f;
    float f2 = (mz > -INFINITY) ? exp2f(mz - mn) : 0.f;
#pragma unroll
    for (int i = 0; i < VEC; ++i)
      o2[i] = o2[i] * f1 + osm[(size_t)u * HD + tx * VEC + i] * f2;
    d2 = d2 * f1 + dz * f2;
    m2 = mn;
  }

  float inv = d2 > 0.f ? 1.f / d2 : 0.f;
  if (tz == 0) {
    unsigned short *op = out + ((size_t)b * HQ + qo_head) * HD + tx * VEC;
#pragma unroll
    for (int i = 0; i < VEC; ++i) op[i] = cvt_out(o2[i] * inv, bf16);
  }
}

/* ---- M1 mode 1：bsm smem 路径（GQ 布局不满足 shuffle 约束时使用） ---- */
template <int HD, int GQ, int BDZ>
__device__ __forceinline__ void issue_row_bsm(unsigned short *smem_base,
                                              const unsigned short *__restrict__ src,
                                              const int *__restrict__ bt,
                                              int b, int BTW, int PS, int HKV, int kvh,
                                              int sl, int tx, int ty, int tz, int stg,
                                              int it) {
  constexpr int ROWS = BDZ * GQ;
  const int row = tz * GQ + ty;
  int pos = it * ROWS + row;
  bool ok = pos < sl;
  int posc = ok ? pos : sl - 1;
  int pg = posc / PS; if (pg > BTW - 1) pg = BTW - 1;
  int phys = bt[(size_t)b * BTW + pg];
  size_t g = (((size_t)phys * PS + (posc % PS)) * HKV + kvh) * (size_t)HD + tx * 8;
  unsigned short *dst = smem_base + ((size_t)stg * ROWS + row) * HD + tx * 8;
  bsm128(dst, src + g, ok);
}

template <int HD, int GQ>
__global__ void paged_decode_bsm(const unsigned short *__restrict__ Q,
                                 const unsigned short *__restrict__ Kp,
                                 const unsigned short *__restrict__ Vp,
                                 const int *__restrict__ bt,
                                 const int *__restrict__ sl_ptr,
                                 unsigned short *__restrict__ out,
                                 float scale, int HQ, int HKV, int PS, int BTW, int bf16) {
  constexpr int VEC = 8;
  constexpr int BDX = HD / VEC;
  constexpr int NTH = (GQ == 8) ? 512 : 256;
  constexpr int BDZ = NTH / (BDX * GQ);
  constexpr int NS = 1;
  constexpr int ROWS = BDZ * GQ;
  constexpr float LOG2E = 1.4426950408889634f;

  const int b = blockIdx.x, kvh = blockIdx.y;
  const int tx = threadIdx.x, ty = threadIdx.y, tz = threadIdx.z;
  const int sl = sl_ptr[b];
  const int qo_head = kvh * GQ + ty;

  extern __shared__ unsigned short smem[];
  unsigned short *k_smem = smem;
  unsigned short *v_smem = smem + NS * ROWS * HD;

  float q_vec[VEC];
  {
    const unsigned short *qp = Q + ((size_t)b * HQ + qo_head) * HD + tx * VEC;
#pragma unroll
    for (int i = 0; i < VEC; ++i) q_vec[i] = cvt_fast(qp[i], bf16);
  }

  float m = -INFINITY, d = 0.f, o[VEC], s[GQ];
#pragma unroll
  for (int i = 0; i < VEC; ++i) o[i] = 0.f;

  if (sl > 0) {
    const float slog2 = scale * LOG2E;
#pragma unroll
    for (int it = 0; it < NS; ++it) {
      issue_row_bsm<HD, GQ, BDZ>(k_smem, Kp, bt, b, BTW, PS, HKV, kvh, sl, tx, ty, tz, it, it);
      issue_row_bsm<HD, GQ, BDZ>(v_smem, Vp, bt, b, BTW, PS, HKV, kvh, sl, tx, ty, tz, it, it);
    }
    __builtin_mxc_arrive_gvmcnt(0);
    __builtin_mxc_barrier_inst();
    __syncthreads();

    const int iters = (sl + ROWS - 1) / ROWS;
    int stage = 0;
#pragma unroll 1
    for (int iter = 0; iter < iters; ++iter) {
      float m_prev = m;
#pragma unroll
      for (int jj = 0; jj < GQ; ++jj) {
        const unsigned short *kr =
            k_smem + ((size_t)stage * ROWS + tz * GQ + jj) * HD + tx * VEC;
        float sacc = 0.f;
#pragma unroll
        for (int i = 0; i < VEC; ++i) sacc += q_vec[i] * cvt_fast(kr[i], bf16);
#pragma unroll
        for (int off = BDX / 2; off > 0; off >>= 1) sacc += shfl_xor_f(sacc, off);
        sacc *= slog2;
        if (iter * ROWS + tz * GQ + jj >= sl) sacc = -INFINITY;
        m = fmaxf(m, sacc);
        s[jj] = sacc;
      }
      float os = (m > -INFINITY) ? exp2f(m_prev - m) : 1.f;
      d *= os;
#pragma unroll
      for (int i = 0; i < VEC; ++i) o[i] *= os;
#pragma unroll
      for (int jj = 0; jj < GQ; ++jj) {
        float p = (s[jj] > -INFINITY) ? exp2f(s[jj] - m) : 0.f;
        d += p;
        s[jj] = p;
      }

      __syncthreads();
      issue_row_bsm<HD, GQ, BDZ>(k_smem, Kp, bt, b, BTW, PS, HKV, kvh, sl, tx, ty, tz,
                                 stage, iter + NS);
#pragma unroll
      for (int jj = 0; jj < GQ; ++jj) {
        const unsigned short *vr =
            v_smem + ((size_t)stage * ROWS + tz * GQ + jj) * HD + tx * VEC;
#pragma unroll
        for (int i = 0; i < VEC; ++i) o[i] += s[jj] * cvt_fast(vr[i], bf16);
      }
      __syncthreads();
      issue_row_bsm<HD, GQ, BDZ>(v_smem, Vp, bt, b, BTW, PS, HKV, kvh, sl, tx, ty, tz,
                                 stage, iter + NS);
      __builtin_mxc_arrive_gvmcnt(0);
      __builtin_mxc_barrier_inst();
      __syncthreads();
      stage = (stage + 1) % NS;
    }
  }

  __syncthreads();
  float *osm = (float *)smem;
  float *mdm = (float *)(smem + 2 * NS * ROWS * HD);
#pragma unroll
  for (int i = 0; i < VEC; ++i) osm[((size_t)tz * GQ + ty) * HD + tx * VEC + i] = o[i];
  mdm[(tz * GQ + ty) * 2] = m;
  mdm[(tz * GQ + ty) * 2 + 1] = d;
  __syncthreads();

  float m2 = -INFINITY, d2 = 0.f, o2[VEC];
#pragma unroll
  for (int i = 0; i < VEC; ++i) o2[i] = 0.f;
#pragma unroll
  for (int j = 0; j < BDZ; ++j) {
    int u = j * GQ + ty;
    float mz = mdm[u * 2], dz = mdm[u * 2 + 1];
    float mn = fmaxf(m2, mz);
    float f1 = (m2 > -INFINITY) ? exp2f(m2 - mn) : 0.f;
    float f2 = (mz > -INFINITY) ? exp2f(mz - mn) : 0.f;
#pragma unroll
    for (int i = 0; i < VEC; ++i)
      o2[i] = o2[i] * f1 + osm[(size_t)u * HD + tx * VEC + i] * f2;
    d2 = d2 * f1 + dz * f2;
    m2 = mn;
  }

  float inv = d2 > 0.f ? 1.f / d2 : 0.f;
  if (tz == 0) {
    unsigned short *op = out + ((size_t)b * HQ + qo_head) * HD + tx * VEC;
#pragma unroll
    for (int i = 0; i < VEC; ++i) op[i] = cvt_out(o2[i] * inv, bf16);
  }
}

/* ---- M1 mode 2：直连全局内存版（无 shuffle / 无预取 / 无 smem 暂存，定位基准） ---- */
template <int HD, int GQ>
__global__ void paged_decode_dir(const unsigned short *__restrict__ Q,
                                 const unsigned short *__restrict__ Kp,
                                 const unsigned short *__restrict__ Vp,
                                 const int *__restrict__ bt,
                                 const int *__restrict__ sl_ptr,
                                 unsigned short *__restrict__ out,
                                 float scale, int HQ, int HKV, int PS, int BTW, int bf16) {
  constexpr int VEC = 8;
  constexpr int BDX = HD / VEC;
  constexpr int NTH = (GQ == 8) ? 512 : 256;
  constexpr int BDZ = NTH / (BDX * GQ);
  constexpr int ROWS = BDZ * GQ;
  constexpr float LOG2E = 1.4426950408889634f;

  const int b = blockIdx.x, kvh = blockIdx.y;
  const int tx = threadIdx.x, ty = threadIdx.y, tz = threadIdx.z;
  const int sl = sl_ptr[b];
  const int qo_head = kvh * GQ + ty;
  const size_t btw_b = (size_t)b * BTW;

  extern __shared__ float smem_f[];
  float *osm = smem_f;
  float *mdm = smem_f + ROWS * HD;

  float q_vec[VEC];
  {
    const unsigned short *qp = Q + ((size_t)b * HQ + qo_head) * HD + tx * VEC;
#pragma unroll
    for (int i = 0; i < VEC; ++i) q_vec[i] = cvt_fast(qp[i], bf16);
  }

  float m = -INFINITY, d = 0.f, o[VEC], s[GQ];
#pragma unroll
  for (int i = 0; i < VEC; ++i) o[i] = 0.f;

  if (sl > 0) {
    const float slog2 = scale * LOG2E;
    const int iters = (sl + ROWS - 1) / ROWS;
#pragma unroll 1
    for (int iter = 0; iter < iters; ++iter) {
      float m_prev = m;
#pragma unroll
      for (int jj = 0; jj < GQ; ++jj) {
        int pos = iter * ROWS + tz * GQ + jj;
        bool ok = pos < sl;
        int posc = ok ? pos : sl - 1;
        int pg = posc / PS; if (pg > BTW - 1) pg = BTW - 1;
        int phys = bt[btw_b + pg];
        size_t g = (((size_t)phys * PS + (posc % PS)) * HKV + kvh) * (size_t)HD + tx * 8;
        const uint4 kj = ok ? *(const uint4 *)(Kp + g) : uint4{0u, 0u, 0u, 0u};
        float sacc = 0.f;
        sacc += q_vec[0] * cvt_fast((unsigned short)(kj.x & 0xFFFFu), bf16);
        sacc += q_vec[1] * cvt_fast((unsigned short)(kj.x >> 16), bf16);
        sacc += q_vec[2] * cvt_fast((unsigned short)(kj.y & 0xFFFFu), bf16);
        sacc += q_vec[3] * cvt_fast((unsigned short)(kj.y >> 16), bf16);
        sacc += q_vec[4] * cvt_fast((unsigned short)(kj.z & 0xFFFFu), bf16);
        sacc += q_vec[5] * cvt_fast((unsigned short)(kj.z >> 16), bf16);
        sacc += q_vec[6] * cvt_fast((unsigned short)(kj.w & 0xFFFFu), bf16);
        sacc += q_vec[7] * cvt_fast((unsigned short)(kj.w >> 16), bf16);
#pragma unroll
        for (int off = BDX / 2; off > 0; off >>= 1) sacc += shfl_xor_f(sacc, off);
        sacc *= slog2;
        if (!ok) sacc = -INFINITY;
        m = fmaxf(m, sacc);
        s[jj] = sacc;
      }
      float os = (m > -INFINITY) ? exp2f(m_prev - m) : 1.f;
      d *= os;
#pragma unroll
      for (int i = 0; i < VEC; ++i) o[i] *= os;
#pragma unroll
      for (int jj = 0; jj < GQ; ++jj) {
        float p = (s[jj] > -INFINITY) ? exp2f(s[jj] - m) : 0.f;
        d += p;
        s[jj] = p;
      }
#pragma unroll
      for (int jj = 0; jj < GQ; ++jj) {
        int pos = iter * ROWS + tz * GQ + jj;
        bool ok = pos < sl;
        int posc = ok ? pos : sl - 1;
        int pg = posc / PS; if (pg > BTW - 1) pg = BTW - 1;
        int phys = bt[btw_b + pg];
        size_t g = (((size_t)phys * PS + (posc % PS)) * HKV + kvh) * (size_t)HD + tx * 8;
        const uint4 vj = ok ? *(const uint4 *)(Vp + g) : uint4{0u, 0u, 0u, 0u};
        float p = s[jj];
        o[0] += p * cvt_fast((unsigned short)(vj.x & 0xFFFFu), bf16);
        o[1] += p * cvt_fast((unsigned short)(vj.x >> 16), bf16);
        o[2] += p * cvt_fast((unsigned short)(vj.y & 0xFFFFu), bf16);
        o[3] += p * cvt_fast((unsigned short)(vj.y >> 16), bf16);
        o[4] += p * cvt_fast((unsigned short)(vj.z & 0xFFFFu), bf16);
        o[5] += p * cvt_fast((unsigned short)(vj.z >> 16), bf16);
        o[6] += p * cvt_fast((unsigned short)(vj.w & 0xFFFFu), bf16);
        o[7] += p * cvt_fast((unsigned short)(vj.w >> 16), bf16);
      }
    }
  }

  __syncthreads();
#pragma unroll
  for (int i = 0; i < VEC; ++i) osm[((size_t)tz * GQ + ty) * HD + tx * VEC + i] = o[i];
  mdm[(tz * GQ + ty) * 2] = m;
  mdm[(tz * GQ + ty) * 2 + 1] = d;
  __syncthreads();

  float m2 = -INFINITY, d2 = 0.f, o2[VEC];
#pragma unroll
  for (int i = 0; i < VEC; ++i) o2[i] = 0.f;
#pragma unroll
  for (int j = 0; j < BDZ; ++j) {
    int u = j * GQ + ty;
    float mz = mdm[u * 2], dz = mdm[u * 2 + 1];
    float mn = fmaxf(m2, mz);
    float f1 = (m2 > -INFINITY) ? exp2f(m2 - mn) : 0.f;
    float f2 = (mz > -INFINITY) ? exp2f(mz - mn) : 0.f;
#pragma unroll
    for (int i = 0; i < VEC; ++i)
      o2[i] = o2[i] * f1 + osm[(size_t)u * HD + tx * VEC + i] * f2;
    d2 = d2 * f1 + dz * f2;
    m2 = mn;
  }

  float inv = d2 > 0.f ? 1.f / d2 : 0.f;
  if (tz == 0) {
    unsigned short *op = out + ((size_t)b * HQ + qo_head) * HD + tx * VEC;
#pragma unroll
    for (int i = 0; i < VEC; ++i) op[i] = cvt_out(o2[i] * inv, bf16);
  }
}

/* ---- M1 mode 3/4：v3 = vendor decode.cuh SharedKV 结构复刻 ----
 * 关键差异（对照 vendor 源码得出）：
 *  1. 循环内只用 __builtin_mxc_barrier_inst()，不用 __syncthreads（vendor sync_threads 的做法）；
 *  2. 等待用 arrive_gvmcnt(N) 计数语义（cp_async_bsm_wait<N>），NS=2 时真流水；
 *  3. 下一迭代的 bt 页表项在迭代顶部以普通 load 提前发射，避开 bt→K 地址依赖链；
 *  4. K 块在 QK 后立刻发射（与 Vupdate 重叠），V 块在 Vupdate 后发射。
 * mode 3: NS=1（vendor 默认，7 blocks/SM）；mode 4: NS=2（双缓冲真流水）。
 * MF=1：数学剥离（定位用）——保留全部加载/barrier/GQ 循环/merge，softmax 换假累加。 */
template <int HD, int GQ, int NS, int MF = 0, int UM = 0>
__global__ void paged_decode_v3(const unsigned short *__restrict__ Q,
                                const unsigned short *__restrict__ Kp,
                                const unsigned short *__restrict__ Vp,
                                const int *__restrict__ bt,
                                const int *__restrict__ sl_ptr,
                                unsigned short *__restrict__ out,
                                float scale, int HQ, int HKV, int PS, int BTW, int bf16,
                                float *__restrict__ ows, float *__restrict__ mdws,
                                int NP, int c_len) {
  /* split-KV：ows!=nullptr 时 grid.z=pz，本 block 处理全局位置 [pz*c_len, +c_len)，
   * 部分状态写 ows[B,HQ,NP,HD] / mdws[B,HQ,NP,2]；ows==nullptr 时整段直写 out。 */
  constexpr int VEC = 8;
  constexpr int BDX = HD / VEC;
  constexpr int NTH = (GQ == 8) ? 512 : 256;
  constexpr int BDZ = NTH / (BDX * GQ);
  constexpr int ROWS = BDZ * GQ;
  constexpr float LOG2E = 1.4426950408889634f;
  constexpr int WAITN = 2 * NS - 1;   /* 计数等待：稳态在途 2*NS 组，等最老的 K/V 完成 */

  const int pz = ows ? (int)blockIdx.z : 0;
  const int gsl = sl_ptr[(int)blockIdx.x];
  const int c_start = ows ? pz * c_len : 0;
  const int rem = gsl - c_start;
  const int sl = ows ? (rem > c_len ? c_len : (rem > 0 ? rem : 0)) : gsl;
  const int b = blockIdx.x, kvh = blockIdx.y;
  const int tx = threadIdx.x, ty = threadIdx.y, tz = threadIdx.z;
  const int qo_head = kvh * GQ + ty;
  const int row = tz * GQ + ty;
  const size_t btw_b = (size_t)b * BTW;

  extern __shared__ unsigned short smem[];
  unsigned short *k_smem = smem;                      /* NS*ROWS*HD */
  unsigned short *v_smem = smem + (size_t)NS * ROWS * HD;
  float *mdm = (float *)(smem + 2 * (size_t)NS * ROWS * HD);  /* 2*GQ*BDZ floats */
  int *bt_smem = (int *)(mdm + 2 * GQ * BDZ);         /* BTW ints：页表预存 */
  float *red = (float *)(bt_smem + BTW);              /* ROWS*GQ*BDX floats：部分和归约区 */

  float q_vec[VEC];
  {
    const unsigned short *qp = Q + ((size_t)b * HQ + qo_head) * HD + tx * VEC;
#pragma unroll
    for (int i = 0; i < VEC; ++i) q_vec[i] = cvt_fast(qp[i], bf16);
  }

  /* 协作预载本 chunk 覆盖的页表项（消除循环内 bt 全局加载依赖链） */
  const int p0 = c_start / PS;
  const int nth = BDX * GQ * BDZ;
  const int tid_l = tx + ty * BDX + tz * BDX * GQ;
  int pages_load = 0;
  if (sl > 0) {
    int need = (c_start + sl - 1) / PS - p0 + 1;
    int avail = BTW - p0;
    pages_load = need < avail ? need : avail;
  }
  for (int p = tid_l; p < pages_load; p += nth) bt_smem[p] = bt[btw_b + p0 + p];
  __syncthreads();

  float m = -INFINITY, d = 0.f, o[VEC], s[GQ];
#pragma unroll
  for (int i = 0; i < VEC; ++i) o[i] = 0.f;

  if (sl > 0) {
    const float slog2 = scale * LOG2E;
    /* 页表项提前取：iter+NS 的 phys（smem 命中，无全局依赖；与 QK 计算重叠） */
    auto phys_for = [&](int it) -> int {
      int pos = it * ROWS + row;                    /* chunk 内局部位置 */
      int posc = pos < sl ? pos : sl - 1;
      int posg = c_start + posc;                    /* 全局位置 */
      int pg = posg / PS; if (pg > BTW - 1) pg = BTW - 1;
      return bt_smem[pg - p0];
    };
    auto issue_kv = [&](unsigned short *base, int it, int phys, const unsigned short *src,
                        bool ok) {
      int posc = it * ROWS + row;
      posc = posc < sl ? posc : sl - 1;
      int posg = c_start + posc;
      size_t g = (((size_t)phys * PS + (posg % PS)) * HKV + kvh) * (size_t)HD + tx * 8;
      unsigned short *dst = base + ((size_t)(it % NS) * ROWS + row) * HD + tx * 8;
      bsm128(dst, src + g, ok);
    };

    /* prologue：发射前 NS 个 stage 的 K/V */
#pragma unroll
    for (int stg = 0; stg < NS; ++stg) {
      int phys = phys_for(stg);
      bool ok = stg * ROWS + row < sl;
      issue_kv(k_smem, stg, phys, Kp, ok);
      issue_kv(v_smem, stg, phys, Vp, ok);
    }

    const int iters = (sl + ROWS - 1) / ROWS;
#pragma unroll 2
    for (int iter = 0; iter < iters; ++iter) {
      const int stage = iter % NS;
      bool okn = (iter + NS) * ROWS + row < sl;
      int physn = phys_for(iter + NS);                  /* 下一批页表项，提前发射 */

      __builtin_mxc_arrive_gvmcnt(WAITN);               /* vendor 调度：只等 K(i) 就绪，V(i) 仍在途 */
      __builtin_mxc_barrier_inst();

      /* compute_qk：head ty 对块内 GQ 个位置打分（读 smem K 行，无 shuffle） */
      if (MF) {
        /* 数学剥离：保留 smem 读取链，softmax 换假累加 */
#pragma unroll
        for (int jj = 0; jj < GQ; ++jj) {
          const unsigned short *kr =
              k_smem + ((size_t)stage * ROWS + tz * GQ + jj) * HD + tx * VEC;
          unsigned int a = 0;
#pragma unroll
          for (int i = 0; i < VEC; ++i) a |= ((unsigned int)kr[i]) << i;
          s[jj] = (float)a;
        }
        m = 1.f;
      } else if (MF == 2 || MF == 3) {
        /* 梯度诊断路径（保留旧交错形状） */
        float m_prev = m;
#pragma unroll
        for (int jj = 0; jj < GQ; ++jj) {
          const unsigned short *kr =
              k_smem + ((size_t)stage * ROWS + tz * GQ + jj) * HD + tx * VEC;
          float sacc = 0.f;
#pragma unroll
          for (int i = 0; i < VEC; ++i) sacc += q_vec[i] * cvt_fast(kr[i], bf16);
          if (MF != 2) {
#pragma unroll
            for (int off = BDX / 2; off > 0; off >>= 1) sacc += shfl_xor_f(sacc, off);
          }
          sacc *= slog2;
          if (iter * ROWS + tz * GQ + jj >= sl) sacc = -INFINITY;
          m = fmaxf(m, sacc);
          s[jj] = sacc;
        }
        if (MF != 3) {
          float os = (m > -INFINITY) ? exp2f(m_prev - m) : 1.f;
          d *= os;
#pragma unroll
          for (int i = 0; i < VEC; ++i) o[i] *= os;
#pragma unroll
          for (int jj = 0; jj < GQ; ++jj) {
            float p = (s[jj] > -INFINITY) ? exp2f(s[jj] - m) : 0.f;
            d += p;
            s[jj] = p;
          }
        } else {
          d += 1.f;
        }
      } else {
        /* mov_shfl scan 归约（寄存器级，零 barrier 零 smem）：
         * 4 步 Hillis-Steele scan + 1 次组内广播，5 条 mov_shfl（~16ns/条）
         * 替代 4 条 bsm_bpermute（~66ns/条）或 smem 归约 + barrier。
         * 16-lane 组边界 = K 行线程组（BDX 对齐），天然匹配；
         * 广播源 = ty*BDX + BDX-1（BDX<16 时同组两行各取自己的末 lane）。 */
        float m_prev = m;
#pragma unroll
        for (int jj = 0; jj < GQ; ++jj) {
          const unsigned short *kr =
              k_smem + ((size_t)stage * ROWS + tz * GQ + jj) * HD + tx * VEC;
          float x = 0.f;
#pragma unroll
          for (int i = 0; i < VEC; ++i) x += q_vec[i] * cvt_fast(kr[i], bf16);
#pragma unroll
          for (int dd = 1; dd < BDX; dd <<= 1) {
            float o2 = shfl_up_g(x, dd);
            if (tx >= dd) x += o2;
          }
          float sum = shfl_bcast_g(x, ty * BDX + BDX - 1) * slog2;
          if (iter * ROWS + tz * GQ + jj >= sl) sum = -INFINITY;
          s[jj] = sum;
        }
        if (UM == 1) {
          /* FlashDecoding++ 统一 max（φ=0）：p=2^s 直接累加，循环内零重缩放、
           * 零 m→o 依赖链；φ 在 o/d 比值中数学消掉。循环末换算回标准形态，
           * 跨 tz merge 与输出代码不变。溢出风险：s>126 时 p=inf——
           * 实测数据 |s|<50，gate 测试范围安全；生产用 UM==2（mode 11）。 */
#pragma unroll
          for (int jj = 0; jj < GQ; ++jj) {
            float p = (s[jj] > -INFINITY) ? exp2_fast(s[jj]) : 0.f;
            d += p;
            m = fmaxf(m, s[jj]);
            s[jj] = p;
          }
        } else if (UM == 2) {
          /* 分桶惰性缩放（bucketed UM，mode 11）：m_used 只落在 32 的整数倍桶边界，
           * 仅当真实 max 越桶才重缩放一次（每行通常 2~4 次，≈ online softmax 的
           * 1/10 重缩放次数）。p = 2^(s-m_used) ≤ 1，天然无溢出；m_used 与真实
           * max 只差桶内偏移（<32），o/d 比值在实数缩放代数上与 online softmax 等价；考虑近似 exp2、量化和浮点舍入，不宣称 bitwise 等价。
           * combine/merge 按一致基准合并，无需感知。 */
          float m_new = m;
#pragma unroll
          for (int jj = 0; jj < GQ; ++jj) m_new = fmaxf(m_new, s[jj]);
          if (m_new > m) {
            /*桶上界：m_new/32 先向零截断再加 1 再乘 32（对负数按向零截断处理，
             * 结果恒满足 mb >= m_new）；不是数学 ceil，实现保持不变。 */
            float mb = ((float)(int)(m_new * 0.03125f) + 1.f) * 32.f;
            float os = exp2_fast(m - mb);       /* m=-inf 时 os=0，d/o 本就为零 */
            d *= os;
#pragma unroll
            for (int i = 0; i < VEC; ++i) o[i] *= os;
            m = mb;
          }
#pragma unroll
          for (int jj = 0; jj < GQ; ++jj) {
            float p = (s[jj] > -INFINITY) ? exp2_fast(s[jj] - m) : 0.f;
            d += p;
            s[jj] = p;
          }
        } else {
        float m_new = m;
#pragma unroll
        for (int jj = 0; jj < GQ; ++jj) m_new = fmaxf(m_new, s[jj]);
        float os = (m_new > -INFINITY) ? exp2_fast(m_prev - m_new) : 1.f;
        d *= os;
#pragma unroll
        for (int i = 0; i < VEC; ++i) o[i] *= os;
#pragma unroll
        for (int jj = 0; jj < GQ; ++jj) {
          float p = (s[jj] > -INFINITY) ? exp2_fast(s[jj] - m_new) : 0.f;
          d += p;
          s[jj] = p;
        }
        m = m_new;
        }
      }

      __builtin_mxc_barrier_inst();                     /* k_smem[stage] 读取完毕 */
      issue_kv(k_smem, iter + NS, physn, Kp, okn);      /* 下一批 K（在途：V(i), K(i+1)…） */
      __builtin_mxc_arrive_gvmcnt(WAITN);               /* 等 V(i) 就绪（更新的组仍可在途） */
      __builtin_mxc_barrier_inst();

      /* update：o += Σ p_jj · V_jj */
      if (MF) {
#pragma unroll
        for (int jj = 0; jj < GQ; ++jj) {
          const unsigned short *vr =
              v_smem + ((size_t)stage * ROWS + tz * GQ + jj) * HD + tx * VEC;
#pragma unroll
          for (int i = 0; i < VEC; ++i) o[i] += s[jj] + (float)vr[i];
        }
      } else {
#pragma unroll
      for (int jj = 0; jj < GQ; ++jj) {
        const unsigned short *vr =
            v_smem + ((size_t)stage * ROWS + tz * GQ + jj) * HD + tx * VEC;
#pragma unroll
        for (int i = 0; i < VEC; ++i) o[i] += s[jj] * cvt_fast(vr[i], bf16);
      }
      }

      __builtin_mxc_barrier_inst();                     /* v_smem[stage] 读取完毕 */
      issue_kv(v_smem, iter + NS, physn, Vp, okn);      /* 下一批 V */
    }
    __builtin_mxc_arrive_gvmcnt(0);                     /* 尾部收水 */
    __builtin_mxc_barrier_inst();
  }
  if (UM == 1) {
    /* UM 收尾：把 (o,d) 从 φ=0 基准换算回真实 max 基准：o = o'·2^(-m)。
     * merge/输出代码不变，输出比值 o/d 数学不变。（UM==2 分桶版 m 已是
     * m_used 基准，无需换算。） */
    float osf = (m > -INFINITY) ? exp2_fast(-m) : 0.f;
    d *= osf;
#pragma unroll
    for (int i = 0; i < VEC; ++i) o[i] *= osf;
  }

  /* 跨 tz 合并：osm 覆盖 KV smem 区（不再需要），mdm 独立 */
  __syncthreads();
  float *osm = (float *)smem;
#pragma unroll
  for (int i = 0; i < VEC; ++i) osm[((size_t)tz * GQ + ty) * HD + tx * VEC + i] = o[i];
  mdm[(tz * GQ + ty) * 2] = m;
  mdm[(tz * GQ + ty) * 2 + 1] = d;
  __syncthreads();

  float m2 = -INFINITY, d2 = 0.f, o2[VEC];
#pragma unroll
  for (int i = 0; i < VEC; ++i) o2[i] = 0.f;
#pragma unroll
  for (int j = 0; j < BDZ; ++j) {
    int u = j * GQ + ty;
    float mz = mdm[u * 2], dz = mdm[u * 2 + 1];
    float mn = fmaxf(m2, mz);
    float f1 = (m2 > -INFINITY) ? exp2f(m2 - mn) : 0.f;
    float f2 = (mz > -INFINITY) ? exp2f(mz - mn) : 0.f;
#pragma unroll
    for (int i = 0; i < VEC; ++i)
      o2[i] = o2[i] * f1 + osm[(size_t)u * HD + tx * VEC + i] * f2;
    d2 = d2 * f1 + dz * f2;
    m2 = mn;
  }

  float inv = d2 > 0.f ? 1.f / d2 : 0.f;
  if (tz == 0) {
    if (ows) {
      /* split-KV：写部分状态（未归一化），交由 combine 内核归并 */
      float *op = ows + (((size_t)b * HQ + qo_head) * NP + pz) * HD + tx * VEC;
#pragma unroll
      for (int i = 0; i < VEC; ++i) op[i] = o2[i];
      float *mp = mdws + (((size_t)b * HQ + qo_head) * NP + pz) * 2;
      mp[0] = m2; mp[1] = d2;
    } else {
      unsigned short *op = out + ((size_t)b * HQ + qo_head) * HD + tx * VEC;
#pragma unroll
      for (int i = 0; i < VEC; ++i) op[i] = cvt_out(o2[i] * inv, bf16);
    }
  }
}

/* ---- split-KV combine：每 block 一个 (b,head)，NP 个部分状态在线归并 ---- */
__global__ void split_combine(const float *__restrict__ ows, const float *__restrict__ mdws,
                              unsigned short *__restrict__ out, int NP, int bf16) {
  const int id = blockIdx.x;                 /* b*HQ + head */
  const int tx = threadIdx.x;                /* blockDim = HD/8 */
  const int HD = blockDim.x * 8;
  const float *mw = mdws + (size_t)id * NP * 2;
  const float *ow = ows + (size_t)id * NP * HD;

  float m2 = -INFINITY, d2 = 0.f, o[8];
#pragma unroll
  for (int i = 0; i < 8; ++i) o[i] = 0.f;
  for (int p = 0; p < NP; ++p) {
    float mz = mw[p * 2], dz = mw[p * 2 + 1];
    if (mz <= -INFINITY) continue;
    float mn = fmaxf(m2, mz);
    float f1 = (m2 > -INFINITY) ? exp2f(m2 - mn) : 0.f;
    float f2 = exp2f(mz - mn);
#pragma unroll
    for (int i = 0; i < 8; ++i) o[i] = o[i] * f1 + ow[(size_t)p * HD + tx * 8 + i] * f2;
    d2 = d2 * f1 + dz * f2;
    m2 = mn;
  }
  float inv = d2 > 0.f ? 1.f / d2 : 0.f;
  unsigned short *op = out + (size_t)id * HD + tx * 8;
#pragma unroll
  for (int i = 0; i < 8; ++i) op[i] = bf16 ? dev_f2bf(o[i] * inv) : dev_f2h(o[i] * inv);
}

/* ---- launcher ---- */
template <int HD, int GQ, int LM>
static void launch_cp(const void *q, const void *k, const void *v, const void *bt,
                      const void *sl, void *out, float scale, int B, int HQ, int HKV,
                      int PS, int BTW, int bf16, mcStream_t st) {
  constexpr int BDX = HD / 8;
  constexpr int NTH = (GQ == 8) ? 512 : 256;
  constexpr int BDZ = NTH / (BDX * GQ);
  constexpr int ROWS = BDZ * GQ;
  dim3 grid(B, HKV), thr(BDX, GQ, BDZ);
  if (LM == 0) {
    size_t sm = (size_t)ROWS * HD * sizeof(float) + (size_t)2 * GQ * BDZ * sizeof(float);
    paged_decode_reg<HD, GQ><<<grid, thr, sm, st>>>(
        (const unsigned short *)q, (const unsigned short *)k, (const unsigned short *)v,
        (const int *)bt, (const int *)sl, (unsigned short *)out, scale, HQ, HKV, PS, BTW, bf16);
  } else if (LM == 2) {
    size_t sm = (size_t)ROWS * HD * sizeof(float) + (size_t)2 * GQ * BDZ * sizeof(float);
    paged_decode_dir<HD, GQ><<<grid, thr, sm, st>>>(
        (const unsigned short *)q, (const unsigned short *)k, (const unsigned short *)v,
        (const int *)bt, (const int *)sl, (unsigned short *)out, scale, HQ, HKV, PS, BTW, bf16);
  } else if (LM >= 3 && LM <= 11) {
    constexpr int NS = (LM == 4) ? 2 : 1;
    constexpr int MF = (LM == 7) ? 1 : (LM == 8) ? 2 : (LM == 9) ? 3 : 0;
    constexpr int UM = (LM == 10) ? 1 : (LM == 11) ? 2 : 0;
    size_t sm = (size_t)2 * NS * ROWS * HD * sizeof(unsigned short) +
                (size_t)2 * GQ * BDZ * sizeof(float) + (size_t)BTW * sizeof(int) +
              (size_t)ROWS * GQ * BDX * sizeof(float);
    paged_decode_v3<HD, GQ, NS, MF, UM><<<grid, thr, sm, st>>>(
        (const unsigned short *)q, (const unsigned short *)k, (const unsigned short *)v,
        (const int *)bt, (const int *)sl, (unsigned short *)out, scale, HQ, HKV, PS, BTW,
        bf16, nullptr, nullptr, 0, 0);
  } else {
    size_t sm = (size_t)2 * ROWS * HD * sizeof(unsigned short) +
                (size_t)2 * GQ * BDZ * sizeof(float);
    paged_decode_bsm<HD, GQ><<<grid, thr, sm, st>>>(
        (const unsigned short *)q, (const unsigned short *)k, (const unsigned short *)v,
        (const int *)bt, (const int *)sl, (unsigned short *)out, scale, HQ, HKV, PS, BTW, bf16);
  }
}

/* mode 0 需要 tz 块不跨 warp：64 % (BDX*GQ) == 0 且 BDX*GQ <= 64 */
template <int HD, int GQ>
static int launch_cp_dispatch(const void *q, const void *k, const void *v, const void *bt,
                              const void *sl, void *out, float scale, int B, int HQ,
                              int HKV, int PS, int BTW, int bf16, int mode, mcStream_t st) {
  constexpr int BDX = HD / 8;
  constexpr int BLK = BDX * GQ;
  const bool reg_ok = (BLK <= 64) && (64 % BLK == 0);
  const bool v3_ok = (256 % BLK == 0) || (BLK == 128);   /* NTH%(BDX*GQ)==0 */
  int lm;
  if (mode == 2) lm = 2;
  else if (mode == 0 && reg_ok) lm = 0;
  else if (mode >= 3 && mode <= 11 && v3_ok) lm = mode;
  else lm = 1;
  if (lm == 0)
    launch_cp<HD, GQ, 0>(q, k, v, bt, sl, out, scale, B, HQ, HKV, PS, BTW, bf16, st);
  else if (lm == 2)
    launch_cp<HD, GQ, 2>(q, k, v, bt, sl, out, scale, B, HQ, HKV, PS, BTW, bf16, st);
  else if (lm == 3)
    launch_cp<HD, GQ, 3>(q, k, v, bt, sl, out, scale, B, HQ, HKV, PS, BTW, bf16, st);
  else if (lm == 4)
    launch_cp<HD, GQ, 4>(q, k, v, bt, sl, out, scale, B, HQ, HKV, PS, BTW, bf16, st);
  else if (lm == 7)
    launch_cp<HD, GQ, 7>(q, k, v, bt, sl, out, scale, B, HQ, HKV, PS, BTW, bf16, st);
  else if (lm == 8)
    launch_cp<HD, GQ, 8>(q, k, v, bt, sl, out, scale, B, HQ, HKV, PS, BTW, bf16, st);
  else if (lm == 9)
    launch_cp<HD, GQ, 9>(q, k, v, bt, sl, out, scale, B, HQ, HKV, PS, BTW, bf16, st);
  else if (lm == 10)
    launch_cp<HD, GQ, 10>(q, k, v, bt, sl, out, scale, B, HQ, HKV, PS, BTW, bf16, st);
  else if (lm == 11)
    launch_cp<HD, GQ, 11>(q, k, v, bt, sl, out, scale, B, HQ, HKV, PS, BTW, bf16, st);
  else
    launch_cp<HD, GQ, 1>(q, k, v, bt, sl, out, scale, B, HQ, HKV, PS, BTW, bf16, st);
  return 0;
}

/* ---- split-KV 路径（mode 5）：v3 以 grid.z=NP 写部分状态，再由 combine 归并 ----
 * um=1 → UM=1 实例（φ=0 统一 max）；um=2 → UM=2 实例（分桶惰性缩放，无溢出）；0 → online。
 * UM 收尾已把 (o,d) 换算回真实 max 基准再写部分状态，combine 无需感知。 */
template <int HD, int GQ>
static int launch_split(const void *q, const void *k, const void *v, const void *bt,
                        const void *sl, void *out, void *ows, void *mdws, float scale,
                        int B, int HQ, int HKV, int PS, int BTW, int bf16,
                        int NP, int c_len, int um, mcStream_t st) {
  constexpr int BDX = HD / 8;
  constexpr int NTH = (GQ == 8) ? 512 : 256;
  constexpr int BDZ = NTH / (BDX * GQ);
  constexpr int ROWS = BDZ * GQ;
  constexpr int NS = 1;                    /* split 后每 chunk 更短，NS=1 即可 */
  size_t sm = (size_t)2 * NS * ROWS * HD * sizeof(unsigned short) +
              (size_t)2 * GQ * BDZ * sizeof(float) + (size_t)BTW * sizeof(int) +
              (size_t)ROWS * GQ * BDX * sizeof(float);
  dim3 grid(B, HKV, NP);
  dim3 thr(BDX, GQ, BDZ);
  if (um == 1)
    paged_decode_v3<HD, GQ, NS, 0, 1><<<grid, thr, sm, st>>>(
        (const unsigned short *)q, (const unsigned short *)k, (const unsigned short *)v,
        (const int *)bt, (const int *)sl, nullptr, scale, HQ, HKV, PS, BTW, bf16,
        (float *)ows, (float *)mdws, NP, c_len);
  else if (um == 2)
    paged_decode_v3<HD, GQ, NS, 0, 2><<<grid, thr, sm, st>>>(
        (const unsigned short *)q, (const unsigned short *)k, (const unsigned short *)v,
        (const int *)bt, (const int *)sl, nullptr, scale, HQ, HKV, PS, BTW, bf16,
        (float *)ows, (float *)mdws, NP, c_len);
  else
    paged_decode_v3<HD, GQ, NS><<<grid, thr, sm, st>>>(
        (const unsigned short *)q, (const unsigned short *)k, (const unsigned short *)v,
        (const int *)bt, (const int *)sl, nullptr, scale, HQ, HKV, PS, BTW, bf16,
        (float *)ows, (float *)mdws, NP, c_len);
  split_combine<<<B * HQ, HD / 8, 0, st>>>(
      (const float *)ows, (const float *)mdws, (unsigned short *)out, NP, bf16);
  return 0;
}

template <int HD>
static int launch_split_dispatch(const void *q, const void *k, const void *v, const void *bt,
                                 const void *sl, void *out, void *ows, void *mdws,
                                 float scale, int B, int HQ, int HKV, int PS, int BTW,
                                 int bf16, int NP, int c_len, int um, mcStream_t st) {
  const int GQ = HQ / HKV;
  switch (GQ) {
    case 2: return launch_split<HD, 2>(q, k, v, bt, sl, out, ows, mdws, scale, B, HQ, HKV, PS, BTW, bf16, NP, c_len, um, st);
    case 3: return launch_split<HD, 3>(q, k, v, bt, sl, out, ows, mdws, scale, B, HQ, HKV, PS, BTW, bf16, NP, c_len, um, st);
    case 4: return launch_split<HD, 4>(q, k, v, bt, sl, out, ows, mdws, scale, B, HQ, HKV, PS, BTW, bf16, NP, c_len, um, st);
    case 5: return launch_split<HD, 5>(q, k, v, bt, sl, out, ows, mdws, scale, B, HQ, HKV, PS, BTW, bf16, NP, c_len, um, st);
    case 6: return launch_split<HD, 6>(q, k, v, bt, sl, out, ows, mdws, scale, B, HQ, HKV, PS, BTW, bf16, NP, c_len, um, st);
    case 7: return launch_split<HD, 7>(q, k, v, bt, sl, out, ows, mdws, scale, B, HQ, HKV, PS, BTW, bf16, NP, c_len, um, st);
    case 8: return launch_split<HD, 8>(q, k, v, bt, sl, out, ows, mdws, scale, B, HQ, HKV, PS, BTW, bf16, NP, c_len, um, st);
  }
  return -1;
}

/* ---- 流式带宽探针：与 decode 相同的 paged 遍历 + 发射模式，去掉注意力数学。
 * variant: 0 = cp_async + 计数等待, 1 = cp_async + wait-all, 2 = 纯全局加载 ---- */
__global__ void stream_probe(const unsigned short *__restrict__ Kp,
                             const unsigned short *__restrict__ Vp,
                             const int *__restrict__ bt, const int *__restrict__ sl_ptr,
                             float *__restrict__ out, int HKV, int HD, int PS,
                             int variant) {
  const int b = blockIdx.x, kvh = blockIdx.y;
  const int sl = sl_ptr[b];
  const int tx = threadIdx.x;                       /* blockDim.x = 256 */
  const int VEC = 8, BDX = HD / 8, ROWS = 16;       /* 与 decode 同形：16 行/迭代 */
  const int ty = tx / BDX, txx = tx % BDX;
  const size_t btw_b = (size_t)b * gridDim.y;
  extern __shared__ unsigned short smem[];          /* 2*ROWS*HD */
  unsigned short *k_smem = smem;
  unsigned short *v_smem = smem + (size_t)ROWS * HD;

  auto issue = [&](unsigned short *base, int it, const unsigned short *src) {
    int posc = it * ROWS + ty;
    if (posc > sl - 1) posc = sl - 1;
    int phys = bt[btw_b + posc / PS];
    size_t g = (((size_t)phys * PS + (posc % PS)) * HKV + kvh) * (size_t)HD + txx * 8;
    if (variant == 2) {
      uint4 tmp = *(const uint4 *)(src + g);        /* 纯全局 128b 加载 */
      base[txx * 8] = (unsigned short)(tmp.x ^ tmp.w);
    } else {
      bsm128(base + txx * 8, src + g, true);
    }
  };

  float acc = 0.f;
  if (variant == 2) {
    for (int it = 0; it * ROWS < sl; ++it) {
      int posc = it * ROWS + ty;
      if (posc > sl - 1) posc = sl - 1;
      int phys = bt[btw_b + posc / PS];
#pragma unroll
      for (int jj = 0; jj < ROWS; ++jj) {
        int p2 = it * ROWS + jj;
        if (p2 > sl - 1) p2 = sl - 1;
        int ph2 = bt[btw_b + p2 / PS];
        size_t g = (((size_t)ph2 * PS + (p2 % PS)) * HKV + kvh) * (size_t)HD + txx * 8;
        uint4 tmp = *(const uint4 *)(Kp + g);
        acc += (float)(tmp.x ^ tmp.y ^ tmp.z ^ tmp.w);
      }
    }
    out[(b * HKV + kvh) * 32 + ty] = acc;
    return;
  }

  issue(k_smem, 0, Kp);
  issue(v_smem, 0, Vp);
  const int iters = (sl + ROWS - 1) / ROWS;
  for (int it = 0; it < iters; ++it) {
    if (variant == 0) {
      __builtin_mxc_arrive_gvmcnt(1);
    } else {
      __builtin_mxc_arrive_gvmcnt(0);
    }
    __builtin_mxc_barrier_inst();
    /* 读 K/V smem 求和（依赖，防 DCE） */
    const unsigned short *kr = k_smem + ty * HD + txx * 8;
    const unsigned short *vr = v_smem + ty * HD + txx * 8;
    unsigned int a = 0, c = 0;
#pragma unroll
    for (int i = 0; i < 8; ++i) { a |= ((unsigned int)kr[i]) << i; c |= ((unsigned int)vr[i]) << i; }
    acc += (float)(a ^ c);
    __builtin_mxc_barrier_inst();
    issue(k_smem, it + 1, Kp);
    __builtin_mxc_barrier_inst();
    issue(v_smem, it + 1, Vp);
  }
  __builtin_mxc_arrive_gvmcnt(0);
  __builtin_mxc_barrier_inst();
  out[(b * HKV + kvh) * 32 + ty] = acc;
}
extern "C" int mxfa_stream_probe(const void *k, const void *v, const void *bt,
                                 const void *sl, void *out, int B, int HKV, int HD,
                                 int PS, int variant, void *stream) {
  size_t sm = (variant == 2) ? 0 : (size_t)2 * 16 * HD * sizeof(unsigned short);
  dim3 grid(B, HKV), thr(256);
  stream_probe<<<grid, thr, sm, (mcStream_t)stream>>>(
      (const unsigned short *)k, (const unsigned short *)v, (const int *)bt,
      (const int *)sl, (float *)out, HKV, HD, PS, variant);
  return 0;
}

/* ---- 指令成本微探针：单 block，纯寄存器循环，量化 shuffle/exp2f/FMA 单独与组合成本 ---- */
__global__ void opcost_probe(const float *__restrict__ in, float *__restrict__ out,
                             int iters, int vparam) {
  const int t = threadIdx.x;
  float v = in[t];
  float w = v * 0.5f + 1.f;
  float acc1 = v, acc2 = v;
  const int variant = (vparam >= 0) ? vparam : (int)blockIdx.x;
  for (int i = 0; i < iters; ++i) {
    if (variant == 0) {
      acc1 = acc1 * 1.0000001f + 0.f;          /* 占位，防空循环 */
    } else if (variant == 1) {
#pragma unroll
      for (int k = 0; k < 16; ++k) acc1 = acc1 * v + w;
    } else if (variant == 2) {
#pragma unroll
      for (int k = 0; k < 8; ++k) acc1 += __shfl_xor_sync((unsigned long long)-1, acc1, 1);
    } else if (variant == 3) {
#pragma unroll
      for (int k = 0; k < 8; ++k) acc1 += exp2f(acc2 - w);
    } else {
#pragma unroll
      for (int k = 0; k < 8; ++k) acc1 += __shfl_xor_sync((unsigned long long)-1, acc1, 1);
#pragma unroll
      for (int k = 0; k < 8; ++k) acc1 += exp2f(acc2 - w);
    }
  }
  out[variant * 256 + t] = acc1 + acc2;
}
extern "C" int mxfa_opcost_probe(const float *in, float *out, int iters, int solo,
                                 void *stream) {
  int v = solo;   /* -1 = 5 block 并行各自按 blockIdx；>=0 = 单 block 只跑该变体 */
  dim3 g(v >= 0 ? 1 : 5);
  opcost_probe<<<g, 256, 0, (mcStream_t)stream>>>(in, out, iters, v);
  return 0;
}

/* ---- shuffle 行为探针：indexed shuffle 在 64-lane warp 上 src>=32 是否正确 ---- */
static __device__ __forceinline__ int shfl_idx(int x, int src) {
  return __shfl_sync((unsigned long long)-1, x, src);
}
static __device__ __forceinline__ int shfl_xor_i(int x, int off) {
  return __shfl_xor_sync((unsigned long long)-1, x, off);
}
__global__ void shfl_probe_kernel(const int *__restrict__ in, int *__restrict__ out) {
  const int t = threadIdx.x;            /* 64 线程单 block，in[i]=i*1000 */
  int x = in[t];
  out[t * 4 + 0] = shfl_idx(x, 33);               /* 期望 33000；mod32→1000, clamp→31000 */
  out[t * 4 + 1] = shfl_idx(x, (t + 1) & 63);     /* 期望 ((t+1)&63)*1000 */
  out[t * 4 + 2] = shfl_xor_i(x, 16);             /* 期望 in[t^16] */
  out[t * 4 + 3] = shfl_xor_i(x, 32);             /* 期望 in[t^32] */
}
extern "C" int mxfa_shfl_probe(const int *in, int *out, void *stream) {
  shfl_probe_kernel<<<1, 64, 0, (mcStream_t)stream>>>(in, out);
  return 0;
}

extern "C" int mxfa_paged_decode_m1_launch(const void *q, const void *k, const void *v,
                                           const void *bt, const void *sl, void *out,
                                           float scale, int B, int HQ, int HKV, int D,
                                           int PS, int BTW, int max_sl, int bf16,
                                           int mode, void *stream) {
  const int GQ = HQ / HKV;
  if (HQ % HKV != 0) return -1;
  if (D != 128 && D != 64) return -1;
  /* 当前生产 contract 仅开放已验证的 GQ={2,4,8}。
   * GQ=3/5/6/7 在当前 NTH/BDX 布局下实测会因整数布局截断产生静默错误；
   * 其他比率（例如 GQ=16）未完成线程布局、资源占用和 correctness 验证，当前明确拒绝。 */
  if (GQ != 2 && GQ != 4 && GQ != 8) return -1;
  mcStream_t st = (mcStream_t)stream;
  if (D == 128) {
    switch (GQ) {
      case 2: return launch_cp_dispatch<128, 2>(q, k, v, bt, sl, out, scale, B, HQ, HKV, PS, BTW, bf16, mode, st);
      case 3: return launch_cp_dispatch<128, 3>(q, k, v, bt, sl, out, scale, B, HQ, HKV, PS, BTW, bf16, mode, st);
      case 4: return launch_cp_dispatch<128, 4>(q, k, v, bt, sl, out, scale, B, HQ, HKV, PS, BTW, bf16, mode, st);
      case 5: return launch_cp_dispatch<128, 5>(q, k, v, bt, sl, out, scale, B, HQ, HKV, PS, BTW, bf16, mode, st);
      case 6: return launch_cp_dispatch<128, 6>(q, k, v, bt, sl, out, scale, B, HQ, HKV, PS, BTW, bf16, mode, st);
      case 7: return launch_cp_dispatch<128, 7>(q, k, v, bt, sl, out, scale, B, HQ, HKV, PS, BTW, bf16, mode, st);
      case 8: return launch_cp_dispatch<128, 8>(q, k, v, bt, sl, out, scale, B, HQ, HKV, PS, BTW, bf16, mode, st);
    }
  } else {
    switch (GQ) {
      case 2: return launch_cp_dispatch<64, 2>(q, k, v, bt, sl, out, scale, B, HQ, HKV, PS, BTW, bf16, mode, st);
      case 3: return launch_cp_dispatch<64, 3>(q, k, v, bt, sl, out, scale, B, HQ, HKV, PS, BTW, bf16, mode, st);
      case 4: return launch_cp_dispatch<64, 4>(q, k, v, bt, sl, out, scale, B, HQ, HKV, PS, BTW, bf16, mode, st);
      case 5: return launch_cp_dispatch<64, 5>(q, k, v, bt, sl, out, scale, B, HQ, HKV, PS, BTW, bf16, mode, st);
      case 6: return launch_cp_dispatch<64, 6>(q, k, v, bt, sl, out, scale, B, HQ, HKV, PS, BTW, bf16, mode, st);
      case 7: return launch_cp_dispatch<64, 7>(q, k, v, bt, sl, out, scale, B, HQ, HKV, PS, BTW, bf16, mode, st);
      case 8: return launch_cp_dispatch<64, 8>(q, k, v, bt, sl, out, scale, B, HQ, HKV, PS, BTW, bf16, mode, st);
    }
  }
  return -1;
}

/* split-KV 入口：ows[B,HQ,NP,HD] f32 / mdws[B,HQ,NP,2] f32 由调用方预分配。
 * um: 0=online softmax；1=φ=0 统一 max（溢出边界 |s|<126）；2=分桶惰性缩放（无溢出）。 */
extern "C" int mxfa_paged_decode_split_launch(const void *q, const void *k, const void *v,
                                              const void *bt, const void *sl, void *out,
                                              void *ows, void *mdws, float scale,
                                              int B, int HQ, int HKV, int D,
                                              int PS, int BTW, int bf16,
                                              int NP, int c_len, int um, void *stream) {
  const int GQ = HQ / HKV;
  if (HQ % HKV != 0) return -1;
  if (D != 128 && D != 64) return -1;
  if ((GQ != 2 && GQ != 4 && GQ != 8) || NP < 1) return -1;
  mcStream_t st = (mcStream_t)stream;
  if (D == 128)
    return launch_split_dispatch<128>(q, k, v, bt, sl, out, ows, mdws, scale, B, HQ,
                                      HKV, PS, BTW, bf16, NP, c_len, um, st);
  return launch_split_dispatch<64>(q, k, v, bt, sl, out, ows, mdws, scale, B, HQ,
                                   HKV, PS, BTW, bf16, NP, c_len, um, st);
}
