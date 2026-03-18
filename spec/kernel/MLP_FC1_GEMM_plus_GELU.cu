#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

__device__ float gelu(float x) {
    return 0.5f * x * (1.f + tanhf(0.7978845608f * (x + 0.044715f * x * x * x)));
}

__global__ void fc1_gelu_kernel(const float* x, const float* w, const float* b, float* y, int B, int N, int D, int M) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = B * N * M;
    if (idx >= total) return;
    int m = idx % M;
    int n = (idx / M) % N;
    int bs = idx / (N * M);
    float sum = b ? b[m] : 0.f;
    const float* xv = x + (bs * N + n) * D;
    const float* ww = w + m * D;
    for (int k = 0; k < D; ++k) sum += xv[k] * ww[k];
    y[idx] = gelu(sum);
}

torch::Tensor MLP_FC1_GEMM_plus_GELU(torch::Tensor x, torch::Tensor w, torch::Tensor b) {
    int B = x.size(0), N = x.size(1), D = x.size(2), M = w.size(0);
    auto out = torch::zeros({B, N, M}, x.options());
    int total = B * N * M;
    int threads = 256, blocks = (total + threads - 1) / threads;
    fc1_gelu_kernel<<<blocks, threads>>>(x.data_ptr<float>(), w.data_ptr<float>(), b.data_ptr<float>(), out.data_ptr<float>(), B, N, D, M);
    return out;
}
