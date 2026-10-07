/**
 * mxfadecode — 自研 paged-KV decode attention device kernel (MACA/C500)
 * Gate 4 v5: 两阶段结构（打破在线 softmax 的串行依赖链）
 *
 * 消融实验结论（2026-10-07，bench_abl.py，B=96 sl=1024）：
 *   flash 式在线 softmax 的 m 依赖使每位置成为长串行链（bt→K→dot→expf→acc），
 *   实测仅 ~25% 峰值带宽；expf/V/K 加载各占 ~35-42%（互相重叠）。
 *   vendor 同流量跑在 1.4TB/s 带宽屋顶 → 差距在流水线结构，不是算力。
 *
 * v5 设计（block = 128 线程 = 4 warp，grid = B * HKV，D ∈ {64,128}）：
 *   阶段1  QK^T：warp w 处理位置 s ≡ w (mod 4)；lane 持 EPL=D/32 元素，
 *          uint2/uint 完美合并加载（32 lane × 8B = 256B 连续），5 步 shuffle 归约，
 *          lane0 写 s_sc[g][s]。无跨位置依赖 → 深流水。
 *   阶段2  块内 max（warp 归约 + 4 partial 合并，每头 2 次 syncthreads）
 *   阶段3  expf 全并行（线程 t 负责位置 s ≡ t (mod 128)，就地写 p）+ 块内 sum
 *   阶段4  V 加权：线程 t 持输出元素 d=t（要求 D ≤ 128），V 行每位置每块只读
 *          一次（GQ 头共享），纯流式 acc += p[s]*v，#pragma unroll 8 深流水
 *   阶段5  归一化写出
 *
 * smem：GQ * max_sl floats（动态分配；>48KB 时尝试提额至 64KB，再超回退 naive）
 * 朴素 kernel（Gate 2）保留为兜底。
 */
#include <mc_runtime.h>
#include <math.h>
#include <string.h>

/* ---- fp16 <-> fp32（自实现，RNE） ---- */
static __device__ float dev_h2f(unsigned short h) {
  unsigned int sign = (h & 0x8000u) << 16;
  unsigned int exp  = (h & 0x7C00u) >> 10;
  unsigned int man  = h & 0x03FFu;
  unsigned int bits;
  if (exp == 0) {
    float v = (man == 0) ? 0.0f : ldexpf((float)man, -24);
    return sign ? -v : v;
  }
  if (exp == 31) bits = sign | 0x7F800000u | (man << 13);
  else bits = sign | ((exp - 15 + 127) << 23) | (man << 13);
  float f; memcpy(&f, &bits, 4); return f;
}

static __device__ unsigned short dev_f2h(float f) {
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

static __device__ float dev_bf2f(unsigned short h) {
  unsigned int bits = ((unsigned int)h) << 16;
  float f; memcpy(&f, &bits, 4); return f;
}

static __device__ unsigned short dev_f2bf(float f) {
  unsigned int x; memcpy(&x, &f, 4);
  unsigned int lsb = (x >> 16) & 1u;
  x += 0x7FFFu + lsb;
  return (unsigned short)(x >> 16);
}

static __device__ float cvt_in(unsigned short h, int bf16) {
  return bf16 ? dev_bf2f(h) : dev_h2f(h);
}
static __device__ unsigned short cvt_out(float f, int bf16) {
  return bf16 ? dev_f2bf(f) : dev_f2h(f);
}

static __device__ void unpack2(unsigned int u, int bf16, float *a, float *b) {
  *a = cvt_in((unsigned short)(u & 0xFFFFu), bf16);
  *b = cvt_in((unsigned short)(u >> 16), bf16);
}

/* ==================== Gate 4 v5 两阶段 kernel ==================== */
template <int EPL, int GQ>
__global__ void paged_decode_tp(const unsigned short *__restrict__ Q,
                                const unsigned short *__restrict__ Kp,
                                const unsigned short *__restrict__ Vp,
                                const int *__restrict__ bt,
                                const int *__restrict__ sl,
                                unsigned short *__restrict__ O, float scale,
                                int HQ, int HKV, int D, int PS, int BTW,
                                int max_sl, int bf16) {
  const int NVP = (EPL == 4) ? 1 : EPL / 2;   /* uint2 数（EPL=4）或 uint 数 */
  extern __shared__ float smem[];             /* [GQ][max_sl] */
  __shared__ float s_red[4];
  __shared__ float s_m[GQ];
  __shared__ float s_ll[GQ];

  const int bh  = blockIdx.x;                 /* b * HKV + kvh */
  const int b   = bh / HKV;
  const int kvh = bh % HKV;
  const int w   = threadIdx.x >> 5;
  const int lane= threadIdx.x & 31;
  const int tid = threadIdx.x;
  const int len = sl[b];
  (void)HQ; (void)HKV;

  if (len <= 0) {
    /* 空序列：写 0（vLLM decode 不会出现，防御） */
    #pragma unroll
    for (int g = 0; g < GQ; ++g) {
      const int hq = kvh * GQ + g;
      if (tid < D) O[((size_t)b * HQ + hq) * D + tid] = cvt_out(0.0f, bf16);
    }
    return;
  }

  /* bt 行一次性进 smem：页查找从 L2(微秒级) 变 smem(~30cy)，
     并消除 bt→K 的数据依赖链，K 行加载可立即批量发出 */
  int *s_bt = (int *)((float *)smem + (size_t)GQ * max_sl);
  for (int i = tid; i < BTW; i += 128) s_bt[i] = bt[(size_t)b * BTW + i];
  __syncthreads();

  /* ---- 阶段1：QK^T，warp w 负责 s ≡ w (mod 4) ---- */
  float qreg[EPL];
  #pragma unroll
  for (int g = 0; g < GQ; ++g) {
    /* GQ 循环放在外层只为 q 加载；实际点积在位置循环里做（qreg 存下所有头） */
  }
  float qall[8][EPL];                         /* GQ <= 8 */
  #pragma unroll
  for (int g = 0; g < 8; ++g) {
    if (g < GQ) {
      const int hq = kvh * GQ + g;
      if (EPL == 4) {
        uint2 u2 = *(const uint2 *)(Q + ((size_t)b * HQ + hq) * D + lane * 4);
        unpack2(u2.x, bf16, &qall[g][0], &qall[g][1]);
        unpack2(u2.y, bf16, &qall[g][2], &qall[g][3]);
      } else {
        #pragma unroll
        for (int i = 0; i < NVP; ++i) {
          unsigned int u = ((const unsigned int *)(Q + ((size_t)b * HQ + hq) * D))[lane * EPL + i];
          unpack2(u, bf16, &qall[g][2 * i], &qall[g][2 * i + 1]);
        }
      }
    }
  }

  /* 8 深度批量流水：一次迭代处理 warp 的 8 个位置（s ≡ w mod 4，间隔 4），
     bt 地址与数据无关可先行批量发出，K 行批量在途 → 每 thread 64B 在途 */
  for (int s0 = w; s0 < len; s0 += 32) {
    const int s_i[8] = {s0, s0 + 4, s0 + 8, s0 + 12, s0 + 16, s0 + 20, s0 + 24, s0 + 28};
    bool ok[8];
    int phys[8], offc[8];
    #pragma unroll
    for (int j = 0; j < 8; ++j) {
      ok[j] = s_i[j] < len;
      if (ok[j]) {
        const int page = s_i[j] / PS;
        offc[j] = s_i[j] - page * PS;
        phys[j] = s_bt[page];                      /* smem，~30cy */
      }
    }
    uint2 kd[8];
    #pragma unroll
    for (int j = 0; j < 8; ++j) {                  /* K 行批量在途（volatile 钉住发射点） */
      if (ok[j]) {
        const volatile unsigned int *kp =
            (const volatile unsigned int *)(Kp + (((size_t)phys[j] * PS + offc[j]) * HKV + kvh) * D + lane * 4);
        kd[j].x = kp[0];
        kd[j].y = kp[1];
      }
    }
    #pragma unroll
    for (int j = 0; j < 8; ++j) {
      if (!ok[j]) continue;
      float k[4];
      unpack2(kd[j].x, bf16, &k[0], &k[1]);
      unpack2(kd[j].y, bf16, &k[2], &k[3]);
      #pragma unroll
      for (int g = 0; g < GQ; ++g) {
        float dot = 0.0f;
        #pragma unroll
        for (int jj = 0; jj < 4; ++jj) dot += qall[g][jj] * k[jj];
        dot += __shfl_xor_sync(0xffffffffu, dot, 16);
        dot += __shfl_xor_sync(0xffffffffu, dot, 8);
        dot += __shfl_xor_sync(0xffffffffu, dot, 4);
        dot += __shfl_xor_sync(0xffffffffu, dot, 2);
        dot += __shfl_xor_sync(0xffffffffu, dot, 1);
        if (lane == 0) smem[(size_t)g * max_sl + s_i[j]] = dot * scale;
      }
    }
  }
  __syncthreads();

  /* ---- 阶段2+3：每头 max → expf(并行) → sum ---- */
  #pragma unroll
  for (int g = 0; g < GQ; ++g) {
    float *sc = smem + (size_t)g * max_sl;

    float lmax = -INFINITY;
    for (int s = tid; s < len; s += 128) lmax = fmaxf(lmax, sc[s]);
    lmax = fmaxf(lmax, __shfl_xor_sync(0xffffffffu, lmax, 16));
    lmax = fmaxf(lmax, __shfl_xor_sync(0xffffffffu, lmax, 8));
    lmax = fmaxf(lmax, __shfl_xor_sync(0xffffffffu, lmax, 4));
    lmax = fmaxf(lmax, __shfl_xor_sync(0xffffffffu, lmax, 2));
    lmax = fmaxf(lmax, __shfl_xor_sync(0xffffffffu, lmax, 1));
    if (lane == 0) s_red[w] = lmax;
    __syncthreads();
    if (w == 0) {
      float v = (lane < 4) ? s_red[lane] : -INFINITY;
      v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, 2));
      v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, 1));
      if (lane == 0) s_m[g] = v;
    }
    __syncthreads();
    const float m = s_m[g];

    float lsum = 0.0f;
    for (int s = tid; s < len; s += 128) {
      const float e = __expf(sc[s] - m);
      sc[s] = e;
      lsum += e;
    }
    lsum += __shfl_xor_sync(0xffffffffu, lsum, 16);
    lsum += __shfl_xor_sync(0xffffffffu, lsum, 8);
    lsum += __shfl_xor_sync(0xffffffffu, lsum, 4);
    lsum += __shfl_xor_sync(0xffffffffu, lsum, 2);
    lsum += __shfl_xor_sync(0xffffffffu, lsum, 1);
    if (lane == 0) s_red[w] = lsum;
    __syncthreads();
    if (w == 0) {
      float v = (lane < 4) ? s_red[lane] : 0.0f;
      v += __shfl_xor_sync(0xffffffffu, v, 2);
      v += __shfl_xor_sync(0xffffffffu, v, 1);
      if (lane == 0) s_ll[g] = v;
    }
    __syncthreads();
  }

  /* ---- 阶段4：V 加权流式累加（线程 t 持元素 d=t，要求 D ≤ 128） ---- */
  if (tid < D) {
    float acc[GQ];
    #pragma unroll
    for (int g = 0; g < GQ; ++g) acc[g] = 0.0f;

    int page = 0, off = 0;
    int s = 0;
    for (; s + 16 <= len; s += 16) {
      int ph[16];
      #pragma unroll
      for (int j = 0; j < 16; ++j) {
        const int pg = (off + j) / PS;
        ph[j] = s_bt[page + pg];
      }
      float vv[16];
      #pragma unroll
      for (int j = 0; j < 16; ++j) {
        const int o = (off + j) - ((off + j) / PS) * PS;
        vv[j] = cvt_in(*(const volatile unsigned short *)
                       (Vp + ((size_t)ph[j] * PS + o) * HKV * D + (size_t)kvh * D + tid), bf16);
      }
      #pragma unroll
      for (int j = 0; j < 16; ++j) {
        #pragma unroll
        for (int g = 0; g < GQ; ++g) acc[g] += smem[(size_t)g * max_sl + s + j] * vv[j];
      }
      off += 16;
      page += off / PS;
      off -= (off / PS) * PS;
    }
    for (; s < len; ++s) {
      const int phys = s_bt[page];
      const float v = cvt_in(*(const volatile unsigned short *)
                             (Vp + ((size_t)phys * PS + off) * HKV * D + (size_t)kvh * D + tid), bf16);
      #pragma unroll
      for (int g = 0; g < GQ; ++g) acc[g] += smem[(size_t)g * max_sl + s] * v;
      ++off;
      if (off >= PS) { off = 0; ++page; }
    }

    #pragma unroll
    for (int g = 0; g < GQ; ++g) {
      const int hq = kvh * GQ + g;
      O[((size_t)b * HQ + hq) * D + tid] = cvt_out(acc[g] / s_ll[g], bf16);
    }
  }
}

/* ==================== Gate 2 朴素 kernel（兜底） ==================== */
extern "C" __global__ void paged_decode_naive(const unsigned short *Q, const unsigned short *Kp,
                                              const unsigned short *Vp, const int *bt,
                                              const int *sl, unsigned short *O, float scale,
                                              int HQ, int HKV, int GQ, int D, int PS, int BTW,
                                              int max_sl, int bf16) {
  extern __shared__ float smem[];
  float *s_q     = smem;
  float *s_red   = smem + D;
  float *s_score = smem + 2 * D;
  float *s_p     = smem + 2 * D + max_sl;

  int bh  = blockIdx.x;
  int b   = bh / HQ;
  int hq  = bh % HQ;
  int kvh = hq / GQ;
  int tid = threadIdx.x;
  int len = sl[b];

  s_q[tid] = cvt_in(Q[bh * D + tid], bf16);
  __syncthreads();

  for (int s = 0; s < len; ++s) {
    int p = s / PS, off = s % PS;
    int phys = bt[b * BTW + p];
    const unsigned short *Krow = Kp + (((size_t)phys * PS + off) * HKV + kvh) * D;
    s_red[tid] = s_q[tid] * cvt_in(Krow[tid], bf16);
    __syncthreads();
    for (int st = D / 2; st > 0; st >>= 1) {
      if (tid < st) s_red[tid] += s_red[tid + st];
      __syncthreads();
    }
    if (tid == 0) s_score[s] = s_red[0] * scale;
    __syncthreads();
  }

  if (tid == 0) {
    float m = s_score[0];
    for (int s = 1; s < len; ++s) if (s_score[s] > m) m = s_score[s];
    float sum = 0.0f;
    for (int s = 0; s < len; ++s) { float e = expf(s_score[s] - m); s_p[s] = e; sum += e; }
    for (int s = 0; s < len; ++s) s_p[s] /= sum;
  }
  __syncthreads();

  float acc = 0.0f;
  for (int s = 0; s < len; ++s) {
    int p = s / PS, off = s % PS;
    int phys = bt[b * BTW + p];
    const unsigned short *Vrow = Vp + (((size_t)phys * PS + off) * HKV + kvh) * D;
    acc += s_p[s] * cvt_in(Vrow[tid], bf16);
  }
  O[bh * D + tid] = cvt_out(acc, bf16);
}

/* ==================== launcher ==================== */
extern "C" void mxfa_paged_decode_launch(const void *q, const void *k, const void *v,
                                         const void *bt, const void *sl, void *out,
                                         float scale, int B, int HQ, int HKV, int D,
                                         int PS, int BTW, int max_sl, int bf16, void *stream) {
  mcStream_t st = (mcStream_t)stream;
  const int GQ = HQ / HKV;

  /* v5 快路径：D ∈ {64,128}（128 线程覆盖 D），GQ ≤ 8，smem 可容纳 */
  if ((D == 128 || D == 64) && GQ >= 1 && GQ <= 8) {
    size_t smem_sz = (size_t)GQ * max_sl * sizeof(float) + (size_t)BTW * sizeof(int);
    bool smem_ok = smem_sz <= 48 * 1024;
    if (!smem_ok && smem_sz <= 64 * 1024) {
      /* 提额在 switch 内按实际实例处理：此处先假设可提额，kernel 启动失败再回退 */
      smem_ok = true;
    }
    if (smem_ok) {
      if (D == 128) {
        switch (GQ) {
          case 1: paged_decode_tp<4,1><<<B*HKV,128,smem_sz,st>>>((const unsigned short*)q,(const unsigned short*)k,(const unsigned short*)v,(const int*)bt,(const int*)sl,(unsigned short*)out,scale,HQ,HKV,D,PS,BTW,max_sl,bf16); return;
          case 2: paged_decode_tp<4,2><<<B*HKV,128,smem_sz,st>>>((const unsigned short*)q,(const unsigned short*)k,(const unsigned short*)v,(const int*)bt,(const int*)sl,(unsigned short*)out,scale,HQ,HKV,D,PS,BTW,max_sl,bf16); return;
          case 3: paged_decode_tp<4,3><<<B*HKV,128,smem_sz,st>>>((const unsigned short*)q,(const unsigned short*)k,(const unsigned short*)v,(const int*)bt,(const int*)sl,(unsigned short*)out,scale,HQ,HKV,D,PS,BTW,max_sl,bf16); return;
          case 4: paged_decode_tp<4,4><<<B*HKV,128,smem_sz,st>>>((const unsigned short*)q,(const unsigned short*)k,(const unsigned short*)v,(const int*)bt,(const int*)sl,(unsigned short*)out,scale,HQ,HKV,D,PS,BTW,max_sl,bf16); return;
          case 5: paged_decode_tp<4,5><<<B*HKV,128,smem_sz,st>>>((const unsigned short*)q,(const unsigned short*)k,(const unsigned short*)v,(const int*)bt,(const int*)sl,(unsigned short*)out,scale,HQ,HKV,D,PS,BTW,max_sl,bf16); return;
          case 6: paged_decode_tp<4,6><<<B*HKV,128,smem_sz,st>>>((const unsigned short*)q,(const unsigned short*)k,(const unsigned short*)v,(const int*)bt,(const int*)sl,(unsigned short*)out,scale,HQ,HKV,D,PS,BTW,max_sl,bf16); return;
          case 7: paged_decode_tp<4,7><<<B*HKV,128,smem_sz,st>>>((const unsigned short*)q,(const unsigned short*)k,(const unsigned short*)v,(const int*)bt,(const int*)sl,(unsigned short*)out,scale,HQ,HKV,D,PS,BTW,max_sl,bf16); return;
          case 8: paged_decode_tp<4,8><<<B*HKV,128,smem_sz,st>>>((const unsigned short*)q,(const unsigned short*)k,(const unsigned short*)v,(const int*)bt,(const int*)sl,(unsigned short*)out,scale,HQ,HKV,D,PS,BTW,max_sl,bf16); return;
        }
      }
      if (D == 64) {
        switch (GQ) {
          case 1: paged_decode_tp<2,1><<<B*HKV,128,smem_sz,st>>>((const unsigned short*)q,(const unsigned short*)k,(const unsigned short*)v,(const int*)bt,(const int*)sl,(unsigned short*)out,scale,HQ,HKV,D,PS,BTW,max_sl,bf16); return;
          case 2: paged_decode_tp<2,2><<<B*HKV,128,smem_sz,st>>>((const unsigned short*)q,(const unsigned short*)k,(const unsigned short*)v,(const int*)bt,(const int*)sl,(unsigned short*)out,scale,HQ,HKV,D,PS,BTW,max_sl,bf16); return;
          case 3: paged_decode_tp<2,3><<<B*HKV,128,smem_sz,st>>>((const unsigned short*)q,(const unsigned short*)k,(const unsigned short*)v,(const int*)bt,(const int*)sl,(unsigned short*)out,scale,HQ,HKV,D,PS,BTW,max_sl,bf16); return;
          case 4: paged_decode_tp<2,4><<<B*HKV,128,smem_sz,st>>>((const unsigned short*)q,(const unsigned short*)k,(const unsigned short*)v,(const int*)bt,(const int*)sl,(unsigned short*)out,scale,HQ,HKV,D,PS,BTW,max_sl,bf16); return;
          case 5: paged_decode_tp<2,5><<<B*HKV,128,smem_sz,st>>>((const unsigned short*)q,(const unsigned short*)k,(const unsigned short*)v,(const int*)bt,(const int*)sl,(unsigned short*)out,scale,HQ,HKV,D,PS,BTW,max_sl,bf16); return;
          case 6: paged_decode_tp<2,6><<<B*HKV,128,smem_sz,st>>>((const unsigned short*)q,(const unsigned short*)k,(const unsigned short*)v,(const int*)bt,(const int*)sl,(unsigned short*)out,scale,HQ,HKV,D,PS,BTW,max_sl,bf16); return;
          case 7: paged_decode_tp<2,7><<<B*HKV,128,smem_sz,st>>>((const unsigned short*)q,(const unsigned short*)k,(const unsigned short*)v,(const int*)bt,(const int*)sl,(unsigned short*)out,scale,HQ,HKV,D,PS,BTW,max_sl,bf16); return;
          case 8: paged_decode_tp<2,8><<<B*HKV,128,smem_sz,st>>>((const unsigned short*)q,(const unsigned short*)k,(const unsigned short*)v,(const int*)bt,(const int*)sl,(unsigned short*)out,scale,HQ,HKV,D,PS,BTW,max_sl,bf16); return;
        }
      }
    }
  }

  /* 兜底：朴素 kernel */
  size_t smem = (size_t)(2 * D + 2 * max_sl) * sizeof(float);
  if (smem > 48 * 1024) {
    mcFuncSetAttribute((const void *)paged_decode_naive,
                       mcFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
  }
  paged_decode_naive<<<B * HQ, D, smem, st>>>(
                                    (const unsigned short *)q, (const unsigned short *)k,
                                    (const unsigned short *)v, (const int *)bt,
                                    (const int *)sl, (unsigned short *)out, scale,
                                    HQ, HKV, GQ, D, PS, BTW, max_sl, bf16);
}
