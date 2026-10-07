/**
 * Gate 1c: 自写 paged-KV decode kernel  vs  官方 vendor kernel（mcFlashAttn）
 *
 * 数据：dump_vendor.py 生成的 vendor_*.bin（fp16 原始位型）
 *   q  [B,1,HQ,D]  k_cache/v_cache [PAGES,PS,HKV,D]  bt [B,PPT] int32  sl [B] int32
 *   vendor out [B,1,HQ,D] fp16
 * 布局（vendor 实测要求）：k/v 按 (num_blocks, page_block_size, num_heads_k, head_size)
 * 流程：fp16→fp32 上采样 → 跑我们的 kernel → 与 ①double reference ②vendor fp16 输出 比对
 */
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <mc_runtime.h>

#define B      2
#define HKV    2
#define GQ     2
#define HQ     (HKV * GQ)
#define D      128
#define PS     16
#define PPT    16
#define PAGES  (B * PPT)
#define THREADS D

static float half2float(unsigned short h) {
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

__global__ void decode_attn_paged(const float *Q, const float *Kpool,
                                  const float *Vpool, const int *block_table,
                                  const int *seq_lens, float scale, float *O) {
  int bh = blockIdx.x;
  int b  = bh / HQ;
  int hq = bh % HQ;
  int kvh = hq / GQ;
  int tid = threadIdx.x;
  int sl = seq_lens[b];

  __shared__ float s_q[D];
  __shared__ float s_red[D];
  __shared__ float s_score[PPT * PS];
  __shared__ float s_p[PPT * PS];

  s_q[tid] = Q[bh * D + tid];
  __syncthreads();

  for (int s = 0; s < sl; ++s) {
    int p = s / PS, off = s % PS;
    int phys = block_table[b * PPT + p];
    const float *Krow = Kpool + (((size_t)phys * PS + off) * HKV + kvh) * D;
    s_red[tid] = s_q[tid] * Krow[tid];
    __syncthreads();
    for (int stride = D / 2; stride > 0; stride >>= 1) {
      if (tid < stride) s_red[tid] += s_red[tid + stride];
      __syncthreads();
    }
    if (tid == 0) s_score[s] = s_red[0] * scale;
    __syncthreads();
  }

  if (tid == 0) {
    float m = s_score[0];
    for (int s = 1; s < sl; ++s) if (s_score[s] > m) m = s_score[s];
    float sum = 0.0f;
    for (int s = 0; s < sl; ++s) { float e = expf(s_score[s] - m); s_p[s] = e; sum += e; }
    for (int s = 0; s < sl; ++s) s_p[s] /= sum;
  }
  __syncthreads();

  float acc = 0.0f;
  for (int s = 0; s < sl; ++s) {
    int p = s / PS, off = s % PS;
    int phys = block_table[b * PPT + p];
    const float *Vrow = Vpool + (((size_t)phys * PS + off) * HKV + kvh) * D;
    acc += s_p[s] * Vrow[tid];
  }
  O[bh * D + tid] = acc;
}

static unsigned short *readfile(const char *path, size_t *nbytes) {
  FILE *f = fopen(path, "rb");
  if (!f) { fprintf(stderr, "open %s failed\n", path); exit(1); }
  fseek(f, 0, SEEK_END); long n = ftell(f); fseek(f, 0, SEEK_SET);
  unsigned short *buf = (unsigned short *)malloc((size_t)n);
  if (fread(buf, 1, (size_t)n, f) != (size_t)n) { fprintf(stderr, "read %s failed\n", path); exit(1); }
  fclose(f);
  *nbytes = (size_t)n;
  return buf;
}

int main(void) {
  size_t nq, nk, nv, nbt, nsl, nout;
  unsigned short *hq16 = readfile("vendor_q.bin", &nq);
  unsigned short *hk16 = readfile("vendor_k.bin", &nk);
  unsigned short *hv16 = readfile("vendor_v.bin", &nv);
  unsigned short *hb16 = readfile("vendor_bt.bin", &nbt);
  unsigned short *hs16 = readfile("vendor_sl.bin", &nsl);
  unsigned short *ho16 = readfile("vendor_out.bin", &nout);
  int *h_bt = (int *)hb16;
  int *h_sl = (int *)hs16;

  size_t qsz = (size_t)B * HQ * D, psz = (size_t)PAGES * PS * HKV * D;
  float *h_Q = (float *)malloc(qsz * 4);
  float *h_K = (float *)malloc(psz * 4);
  float *h_V = (float *)malloc(psz * 4);
  float *h_O = (float *)malloc(qsz * 4);
  for (size_t i = 0; i < qsz; ++i) h_Q[i] = half2float(hq16[i]);
  for (size_t i = 0; i < psz; ++i) { h_K[i] = half2float(hk16[i]); h_V[i] = half2float(hv16[i]); }
  printf("sl = {%d, %d}, bt[0..3] = %d %d %d %d\n",
         h_sl[0], h_sl[1], h_bt[0], h_bt[1], h_bt[2], h_bt[3]);

  /* ---- double reference（同 fp16 上采样输入） ---- */
  double *ref = (double *)malloc(qsz * 8);
  for (int b = 0; b < B; ++b) {
    int sl = h_sl[b];
    double score[PPT * PS], p[PPT * PS];
    for (int hq = 0; hq < HQ; ++hq) {
      int kvh = hq / GQ;
      for (int s = 0; s < sl; ++s) {
        int pg = s / PS, off = s % PS;
        int phys = h_bt[b * PPT + pg];
        const float *Krow = h_K + (((size_t)phys * PS + off) * HKV + kvh) * D;
        double acc = 0.0;
        for (int d = 0; d < D; ++d) acc += (double)h_Q[(b * HQ + hq) * D + d] * Krow[d];
        score[s] = acc * (double)(1.0f / sqrtf((float)D));
      }
      double m = score[0];
      for (int s = 1; s < sl; ++s) if (score[s] > m) m = score[s];
      double sum = 0.0;
      for (int s = 0; s < sl; ++s) { p[s] = exp(score[s] - m); sum += p[s]; }
      for (int s = 0; s < sl; ++s) p[s] /= sum;
      for (int d = 0; d < D; ++d) {
        double acc = 0.0;
        for (int s = 0; s < sl; ++s) {
          int pg = s / PS, off = s % PS;
          int phys = h_bt[b * PPT + pg];
          const float *Vrow = h_V + (((size_t)phys * PS + off) * HKV + kvh) * D;
          acc += p[s] * Vrow[d];
        }
        ref[(b * HQ + hq) * D + d] = acc;
      }
    }
  }

  /* ---- device kernel ---- */
  float *d_Q, *d_K, *d_V, *d_O; int *d_bt, *d_sl;
  if (mcMalloc((void **)&d_Q, qsz * 4) != mcSuccess ||
      mcMalloc((void **)&d_K, psz * 4) != mcSuccess ||
      mcMalloc((void **)&d_V, psz * 4) != mcSuccess ||
      mcMalloc((void **)&d_O, qsz * 4) != mcSuccess ||
      mcMalloc((void **)&d_bt, nbt) != mcSuccess ||
      mcMalloc((void **)&d_sl, nsl) != mcSuccess) {
    fprintf(stderr, "mcMalloc failed\n"); return 1;
  }
  mcMemcpy(d_Q, h_Q, qsz * 4, mcMemcpyHostToDevice);
  mcMemcpy(d_K, h_K, psz * 4, mcMemcpyHostToDevice);
  mcMemcpy(d_V, h_V, psz * 4, mcMemcpyHostToDevice);
  mcMemcpy(d_bt, h_bt, nbt, mcMemcpyHostToDevice);
  mcMemcpy(d_sl, h_sl, nsl, mcMemcpyHostToDevice);
  decode_attn_paged<<<B * HQ, THREADS>>>(d_Q, d_K, d_V, d_bt, d_sl,
                                         1.0f / sqrtf((float)D), d_O);
  if (mcGetLastError() != mcSuccess) { fprintf(stderr, "launch failed\n"); return 1; }
  mcDeviceSynchronize();
  mcMemcpy(h_O, d_O, qsz * 4, mcMemcpyDeviceToHost);

  /* ---- compare ---- */
  double max_ref = 0.0, max_vendor = 0.0, max_ven_ref = 0.0;
  int wref = -1, wven = -1, wvr = -1;
  double vensum = 0.0;
  for (int i = 0; i < (int)qsz; ++i) {
    double dref = fabs(ref[i] - (double)h_O[i]);
    if (dref > max_ref) { max_ref = dref; wref = i; }
    double dven = fabs(half2float(ho16[i]) - (double)h_O[i]);
    if (dven > max_vendor) { max_vendor = dven; wven = i; }
    double dvr = fabs(half2float(ho16[i]) - ref[i]);
    if (dvr > max_ven_ref) { max_ven_ref = dvr; wvr = i; }
    vensum += fabs(half2float(ho16[i]));
  }
  printf("[1] ours vs double ref   : max_abs = %.3e (idx %d)\n", max_ref, wref);
  printf("[2] ours vs vendor       : max_abs = %.3e (idx %d)\n", max_vendor, wven);
  printf("[2b] vendor vs double ref: max_abs = %.3e (idx %d)\n", max_ven_ref, wvr);
  printf("[3] vendor mean |o|    = %.4f\n", vensum / (double)qsz);
  printf("[4] ours   vs vendor   : max_rel = %.3e\n", max_vendor / (vensum / (double)qsz));
  printf("%s\n", max_vendor < 1e-3 ? "GATE1C PASSED (<1e-3)" : "GATE1C: diff above 1e-3, see numbers");

  mcFree(d_Q); mcFree(d_K); mcFree(d_V); mcFree(d_O); mcFree(d_bt); mcFree(d_sl);
  free(h_Q); free(h_K); free(h_V); free(h_O); free(ref);
  free(hq16); free(hk16); free(hv16); free(hb16); free(hs16); free(ho16);
  return 0;
}
