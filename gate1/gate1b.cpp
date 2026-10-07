/**
 * Gate 1b: paged-KV + GQA + 变长 batch 的 decode attention device kernel
 *
 * 形状：B=2, Hq=4 (GQA group=2, Hkv=2), D=128, page_size=16,
 *       seq_lens = {200, 256}（变长），block_table 随机乱序映射物理页
 * 数学：O[b,hq] = softmax(Q[b,hq] K[b,kvh]^T) V[b,kvh]，kvh = hq / 2
 * 布局：Kpool/Vpool = [P][Hkv][PS][D]，block_table[b][p] = 物理页号
 * 验证：host 端 double reference（含同样的页表间接寻址），阈值 1e-4
 *
 * 编译：mxcc -x maca -offload-arch native gate1b.cpp -o gate1b --maca-path=/opt/maca
 */
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <mc_runtime.h>

#define B      2
#define HKV    2
#define GQ     2
#define HQ     (HKV * GQ)
#define D      128
#define PS     16
#define PPT    16          /* max pages per (batch) */
#define PAGES  (B * PPT)   /* physical pool pages */
#define THREADS D

__global__ void decode_attn_paged(const float *Q, const float *Kpool,
                                  const float *Vpool, const int *block_table,
                                  const int *seq_lens, float *O) {
  int bh = blockIdx.x;          /* b * HQ + hq */
  int b  = bh / HQ;
  int hq = bh % HQ;
  int kvh = hq / GQ;
  int tid = threadIdx.x;        /* tid == d */
  int sl = seq_lens[b];

  __shared__ float s_q[D];
  __shared__ float s_red[D];
  __shared__ float s_score[PPT * PS];
  __shared__ float s_p[PPT * PS];

  s_q[tid] = Q[bh * D + tid];
  __syncthreads();

  /* score[s] = dot(Q, K[b,kvh,page(s),off(s)]) */
  for (int s = 0; s < sl; ++s) {
    int p = s / PS, off = s % PS;
    int phys = block_table[b * PPT + p];
    const float *Krow = Kpool + (((size_t)phys * HKV + kvh) * PS + off) * D;
    s_red[tid] = s_q[tid] * Krow[tid];
    __syncthreads();
    for (int stride = D / 2; stride > 0; stride >>= 1) {
      if (tid < stride) s_red[tid] += s_red[tid + stride];
      __syncthreads();
    }
    if (tid == 0) s_score[s] = s_red[0];
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
    const float *Vrow = Vpool + (((size_t)phys * HKV + kvh) * PS + off) * D;
    acc += s_p[s] * Vrow[tid];
  }
  O[bh * D + tid] = acc;
}

static unsigned long long rng_state = 20261008ULL;
static float frand(void) {
  rng_state = rng_state * 6364136223846793005ULL + 1442695040888963407ULL;
  return ((rng_state >> 40) & 0xFFFFFF) / (float)0xFFFFFF * 2.0f - 1.0f;
}
static int irand(int n) {
  rng_state = rng_state * 6364136223846793005ULL + 1442695040888963407ULL;
  return (int)((rng_state >> 33) % (unsigned long long)n);
}

int main(void) {
  size_t qsz  = (size_t)B * HQ * D * sizeof(float);
  size_t psz  = (size_t)PAGES * HKV * PS * D * sizeof(float);
  size_t btsz = (size_t)B * PPT * sizeof(int);
  float *h_Q = (float *)malloc(qsz);
  float *h_K = (float *)malloc(psz);
  float *h_V = (float *)malloc(psz);
  float *h_O = (float *)malloc(qsz);
  int *h_bt = (int *)malloc(btsz);
  int h_sl[B] = {200, 256};

  for (int i = 0; i < B * HQ * D; ++i) h_Q[i] = frand() * 0.5f;
  for (int i = 0; i < PAGES * HKV * PS * D; ++i) { h_K[i] = frand() * 0.5f; h_V[i] = frand(); }

  /* block_table: 每批从 0..PAGES-1 里不重复抽 PPT 个物理页，乱序 */
  int perm[PAGES];
  for (int i = 0; i < PAGES; ++i) perm[i] = i;
  for (int i = PAGES - 1; i > 0; --i) { int j = irand(i + 1); int t = perm[i]; perm[i] = perm[j]; perm[j] = t; }
  for (int b = 0; b < B; ++b)
    for (int p = 0; p < PPT; ++p)
      h_bt[b * PPT + p] = perm[b * PPT + p];

  /* ---- host double reference ---- */
  double ref[B * HQ * D];
  for (int b = 0; b < B; ++b) {
    int sl = h_sl[b];
    double score[PPT * PS], p[PPT * PS];
    for (int hq = 0; hq < HQ; ++hq) {
      int kvh = hq / GQ;
      for (int s = 0; s < sl; ++s) {
        int pg = s / PS, off = s % PS;
        int phys = h_bt[b * PPT + pg];
        const float *Krow = h_K + (((size_t)phys * HKV + kvh) * PS + off) * D;
        double acc = 0.0;
        for (int d = 0; d < D; ++d) acc += (double)h_Q[(b * HQ + hq) * D + d] * Krow[d];
        score[s] = acc;
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
          const float *Vrow = h_V + (((size_t)phys * HKV + kvh) * PS + off) * D;
          acc += p[s] * Vrow[d];
        }
        ref[(b * HQ + hq) * D + d] = acc;
      }
    }
  }

  /* ---- device ---- */
  float *d_Q, *d_K, *d_V, *d_O; int *d_bt, *d_sl;
  if (mcMalloc((void **)&d_Q, qsz) != mcSuccess ||
      mcMalloc((void **)&d_K, psz) != mcSuccess ||
      mcMalloc((void **)&d_V, psz) != mcSuccess ||
      mcMalloc((void **)&d_O, qsz) != mcSuccess ||
      mcMalloc((void **)&d_bt, btsz) != mcSuccess ||
      mcMalloc((void **)&d_sl, B * sizeof(int)) != mcSuccess) {
    fprintf(stderr, "mcMalloc failed\n"); return 1;
  }
  mcMemcpy(d_Q, h_Q, qsz, mcMemcpyHostToDevice);
  mcMemcpy(d_K, h_K, psz, mcMemcpyHostToDevice);
  mcMemcpy(d_V, h_V, psz, mcMemcpyHostToDevice);
  mcMemcpy(d_bt, h_bt, btsz, mcMemcpyHostToDevice);
  mcMemcpy(d_sl, h_sl, B * sizeof(int), mcMemcpyHostToDevice);

  decode_attn_paged<<<B * HQ, THREADS>>>(d_Q, d_K, d_V, d_bt, d_sl, d_O);
  mcError_t err = mcGetLastError();
  if (err != mcSuccess) { fprintf(stderr, "launch failed: %s\n", mcGetErrorString(err)); return 1; }
  mcDeviceSynchronize();
  mcMemcpy(h_O, d_O, qsz, mcMemcpyDeviceToHost);

  double max_abs = 0.0; int worst = -1;
  for (int i = 0; i < B * HQ * D; ++i) {
    double diff = fabs(ref[i] - (double)h_O[i]);
    if (diff > max_abs) { max_abs = diff; worst = i; }
  }
  printf("blocks=%d threads=%d seq_lens={%d,%d} pages shuffled\n", B * HQ, THREADS, h_sl[0], h_sl[1]);
  printf("max_abs_err = %.3e (at idx=%d, ref=%.6f got=%.6f)\n",
         max_abs, worst, ref[worst], (double)h_O[worst]);
  printf("%s\n", max_abs < 1e-4 ? "GATE1B PASSED" : "GATE1B FAILED");

  mcFree(d_Q); mcFree(d_K); mcFree(d_V); mcFree(d_O); mcFree(d_bt); mcFree(d_sl);
  free(h_Q); free(h_K); free(h_V); free(h_O); free(h_bt);
  return 0;
}
