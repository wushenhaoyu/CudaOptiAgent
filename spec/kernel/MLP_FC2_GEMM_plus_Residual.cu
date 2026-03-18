#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void fc2_res_kernel(const float* x, const float* w, const float* b, const float* r, float* y, int B, int N, int M, int D) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = B * N * D;
    if (idx >= total) return;
    int d = idx % D;
    int n = (idx / D) % N;
    int bs = idx / (N * D);

    float sum = b ? b[d] : 0.f;
    const float* xv = x + (bs * N + n) * M;
    const float* ww = w + d * M;
    for (int k = 0; k < M; ++k) sum += xv[k] * ww[k];
    y[idx] = sum + r[idx];
}

torch::Tensor MLP_FC2_GEMM_plus_Residual(torch::Tensor x, torch::Tensor w, torch::Tensor b, torch::Tensor residual) {
    int B = x.size(0), N = x.size(1), M = x.size(2), D = w.size(0);
    auto out = torch::zeros({B, N, D}, x.options());
    int total = B * N * D;
    int threads = 256, blocks = (total + threads - 1) / threads;
    fc2_res_kernel<<<blocks, threads>>>(x.data_ptr<float>(), w.data_ptr<float>(), b.data_ptr<float>(), residual.data_ptr<float>(), out.data_ptr<float>(), B, N, M, D);
    return out;
}
