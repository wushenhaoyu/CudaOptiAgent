#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void head_fc2_kernel(const float* x, const float* w, const float* b, float* y, int B, int M, int C) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = B * C;
    if (idx >= total) return;
    int c = idx % C;
    int bs = idx / C;
    float sum = b ? b[c] : 0.f;
    const float* xv = x + bs * M;
    const float* ww = w + c * M;
    for (int k = 0; k < M; ++k) sum += xv[k] * ww[k];
    y[idx] = sum;
}

torch::Tensor Head_FC2_GEMM(torch::Tensor x, torch::Tensor w, torch::Tensor b) {
    int B = x.size(0), M = x.size(1), C = w.size(0);
    auto out = torch::zeros({B, C}, x.options());
    int total = B * C;
    int threads = 256, blocks = (total + threads - 1) / threads;
    head_fc2_kernel<<<blocks, threads>>>(x.data_ptr<float>(), w.data_ptr<float>(), b.data_ptr<float>(), out.data_ptr<float>(), B, M, C);
    return out;
}
