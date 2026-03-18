#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void tocls_kernel(const float* x, float* y, int B, int N, int D) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = B * D;
    if (idx >= total) return;
    int d = idx % D;
    int b = idx / D;
    y[idx] = x[(b * N + 0) * D + d];
}

torch::Tensor ToCLS_Slice(torch::Tensor x) {
    int B = x.size(0), N = x.size(1), D = x.size(2);
    auto out = torch::zeros({B, D}, x.options());
    int total = B * D;
    int threads = 256, blocks = (total + threads - 1) / threads;
    tocls_kernel<<<blocks, threads>>>(x.data_ptr<float>(), out.data_ptr<float>(), B, N, D);
    return out;
}
