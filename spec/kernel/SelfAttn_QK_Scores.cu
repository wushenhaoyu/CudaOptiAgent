#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

__global__ void qk_scores_kernel(const float* qkv, float* scores, int B, int N, int D, int Hh) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int Dh = D / Hh;
    int total = B * Hh * N * N;
    if (idx >= total) return;

    int j = idx % N;
    int i = (idx / N) % N;
    int h = (idx / (N * N)) % Hh;
    int b = idx / (Hh * N * N);

    float sum = 0.f;
    for (int k = 0; k < Dh; ++k) {
        int qidx = ((b * N + i) * (3 * D)) + (h * Dh + k);
        int kidx = ((b * N + j) * (3 * D)) + (D + h * Dh + k);
        sum += qkv[qidx] * qkv[kidx];
    }
    scores[idx] = sum / sqrtf((float)Dh);
}

torch::Tensor SelfAttn_QK_Scores(torch::Tensor qkv, int64_t heads) {
    int B = qkv.size(0), N = qkv.size(1), D3 = qkv.size(2), D = D3 / 3, Hh = (int)heads;
    auto out = torch::zeros({B, Hh, N, N}, qkv.options());
    int total = B * Hh * N * N;
    int threads = 256, blocks = (total + threads - 1) / threads;
    qk_scores_kernel<<<blocks, threads>>>(qkv.data_ptr<float>(), out.data_ptr<float>(), B, N, D, Hh);
    return out;
}
