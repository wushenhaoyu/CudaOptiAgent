#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void outproj_res_kernel(const float* x, const float* w, const float* b, const float* r, float* y, int B, int N, int D) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = B * N * D;
    if (idx >= total) return;
    int d = idx % D;
    int n = (idx / D) % N;
    int bs = idx / (N * D);
    float sum = b ? b[d] : 0.f;
    const float* xv = x + (bs * N + n) * D;
    const float* ww = w + d * D;
    for (int k = 0; k < D; ++k) sum += xv[k] * ww[k];
    y[idx] = sum + r[idx];
}

torch::Tensor OutProj_GEMM_plus_Residual(torch::Tensor x, torch::Tensor w, torch::Tensor b, torch::Tensor residual) {
    int B = x.size(0), N = x.size(1), D = x.size(2);
    auto out = torch::zeros_like(x);
    int total = B * N * D;
    int threads = 256, blocks = (total + threads - 1) / threads;
    outproj_res_kernel<<<blocks, threads>>>(x.data_ptr<float>(), w.data_ptr<float>(), b.data_ptr<float>(), residual.data_ptr<float>(), out.data_ptr<float>(), B, N, D);
    return out;
}
