#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void attnv_kernel(const float* attn, const float* qkv, float* out, int B, int Hh, int N, int D, int Dh) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = B * N * D;
    if (idx >= total) return;
    int d = idx % D;
    int i = (idx / D) % N;
    int b = idx / (N * D);
    int h = d / Dh;
    int kd = d % Dh;

    float sum = 0.f;
    for (int j = 0; j < N; ++j) {
        float a = attn[((b * Hh + h) * N + i) * N + j];
        float v = qkv[((b * N + j) * (3 * D)) + (2 * D + h * Dh + kd)];
        sum += a * v;
    }
    out[idx] = sum;
}

torch::Tensor SelfAttn_AttnV_MatMul(torch::Tensor attn_probs, torch::Tensor qkv, int64_t heads) {
    int B = qkv.size(0), N = qkv.size(1), D = qkv.size(2) / 3, Hh = (int)heads, Dh = D / Hh;
    auto out = torch::zeros({B, N, D}, qkv.options());
    int total = B * N * D;
    int threads = 256, blocks = (total + threads - 1) / threads;
    attnv_kernel<<<blocks, threads>>>(attn_probs.data_ptr<float>(), qkv.data_ptr<float>(), out.data_ptr<float>(), B, Hh, N, D, Dh);
    return out;
}
