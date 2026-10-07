/**
 * Gate 1a: 自写 decode attention device kernel（最简形状）
 *
 * 形状：B=1, H=1, S=128 KV 长度, D=128 head_dim, query 长度=1 (decode)
 * 数学：O = softmax(Q K^T) V   （fp32 全程，正确性优先）
 * 验证：host 端用 double 算同一数学的 reference，比对 device 输出
 *
 * 编译：mxcc -x maca -offload-arch native gate1a.cpp -o gate1a --maca-path=/opt/maca
 * 运行：./gate1a
 */
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <mc_runtime.h>

#define D 128   // head dim
#define S 128   // kv seq len
#define THREADS D

// 一个 block 算一个 (batch, head) 的 decode attention
__global__ void decode_attn_kernel(const float *Q, const float *K,
                                   const float *V, float *O) {
  int tid = threadIdx.x;  // tid == d (head dim 维度)
  __shared__ float s_q[D];
  __shared__ float s_red[D];
  __shared__ float s_score[S];
  __shared__ float s_p[S];

  s_q[tid] = Q[tid];
  __syncthreads();

  // 阶段1: score[s] = dot(Q, K[s])  —— 逐 s 的 block 内归约
  for (int s = 0; s < S; ++s) {
    s_red[tid] = s_q[tid] * K[s * D + tid];
    __syncthreads();
    for (int stride = D / 2; stride > 0; stride >>= 1) {
      if (tid < stride) s_red[tid] += s_red[tid + stride];
      __syncthreads();
    }
    if (tid == 0) s_score[s] = s_red[0];
    __syncthreads();
  }

  // 阶段2: softmax(score) —— 单线程，正确性优先
  if (tid == 0) {
    float m = s_score[0];
    for (int s = 1; s < S; ++s)
      if (s_score[s] > m) m = s_score[s];
    float sum = 0.0f;
    for (int s = 0; s < S; ++s) {
      float e = expf(s_score[s] - m);
      s_p[s] = e;
      sum += e;
    }
    for (int s = 0; s < S; ++s) s_p[s] /= sum;
  }
  __syncthreads();

  // 阶段3: O[d] = sum_s p[s] * V[s][d]
  float acc = 0.0f;
  for (int s = 0; s < S; ++s) acc += s_p[s] * V[s * D + tid];
  O[tid] = acc;
}

static unsigned long long rng_state = 20261007ULL;
static float frand(void) {
  rng_state = rng_state * 6364136223846793005ULL + 1442695040888963407ULL;
  return ((rng_state >> 40) & 0xFFFFFF) / (float)0xFFFFFF * 2.0f - 1.0f;
}

int main(void) {
  size_t vsz = D * sizeof(float);
  size_t msz = S * D * sizeof(float);
  float *h_Q = (float *)malloc(vsz);
  float *h_K = (float *)malloc(msz);
  float *h_V = (float *)malloc(msz);
  float *h_O = (float *)malloc(vsz);

  for (int d = 0; d < D; ++d) h_Q[d] = frand() * 0.5f;
  for (int i = 0; i < S * D; ++i) { h_K[i] = frand() * 0.5f; h_V[i] = frand(); }

  // ---- host reference (double) ----
  double score[S];
  for (int s = 0; s < S; ++s) {
    double acc = 0.0;
    for (int d = 0; d < D; ++d) acc += (double)h_Q[d] * h_K[s * D + d];
    score[s] = acc;
  }
  double m = score[0];
  for (int s = 1; s < S; ++s) if (score[s] > m) m = score[s];
  double sum = 0.0;
  double p[S];
  for (int s = 0; s < S; ++s) { p[s] = exp(score[s] - m); sum += p[s]; }
  for (int s = 0; s < S; ++s) p[s] /= sum;
  double ref[D];
  for (int d = 0; d < D; ++d) {
    double acc = 0.0;
    for (int s = 0; s < S; ++s) acc += p[s] * h_V[s * D + d];
    ref[d] = acc;
  }

  // ---- device ----
  float *d_Q, *d_K, *d_V, *d_O;
  if (mcMalloc((void **)&d_Q, vsz) != mcSuccess ||
      mcMalloc((void **)&d_K, msz) != mcSuccess ||
      mcMalloc((void **)&d_V, msz) != mcSuccess ||
      mcMalloc((void **)&d_O, vsz) != mcSuccess) {
    fprintf(stderr, "mcMalloc failed\n");
    return 1;
  }
  mcMemcpy(d_Q, h_Q, vsz, mcMemcpyHostToDevice);
  mcMemcpy(d_K, h_K, msz, mcMemcpyHostToDevice);
  mcMemcpy(d_V, h_V, msz, mcMemcpyHostToDevice);

  decode_attn_kernel<<<1, THREADS>>>(d_Q, d_K, d_V, d_O);
  mcError_t err = mcGetLastError();
  if (err != mcSuccess) {
    fprintf(stderr, "kernel launch failed: %s\n", mcGetErrorString(err));
    return 1;
  }
  mcDeviceSynchronize();
  mcMemcpy(h_O, d_O, vsz, mcMemcpyDeviceToHost);

  // ---- compare ----
  double max_abs = 0.0;
  int worst = -1;
  for (int d = 0; d < D; ++d) {
    double diff = fabs(ref[d] - (double)h_O[d]);
    if (diff > max_abs) { max_abs = diff; worst = d; }
  }
  printf("max_abs_err = %.3e (at d=%d, ref=%.6f got=%.6f)\n",
         max_abs, worst, ref[worst], (double)h_O[worst]);
  printf("%s\n", max_abs < 1e-4 ? "GATE1A PASSED" : "GATE1A FAILED");

  mcFree(d_Q); mcFree(d_K); mcFree(d_V); mcFree(d_O);
  free(h_Q); free(h_K); free(h_V); free(h_O);
  return 0;
}
