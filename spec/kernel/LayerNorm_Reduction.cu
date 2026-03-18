#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

__global__ void layernorm_kernel(const float* x, const float* g, const float* be, float* y, int rows, int D, float eps) {
    int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= rows) return;
    const float* in = x + r * D;
    float* out = y + r * D;
    float mean = 0.f;
    for (int i = 0; i < D; ++i) mean += in[i];
    mean /= D;
    float var = 0.f;
    for (int i = 0; i < D; ++i) { float t = in[i] - mean; var += t * t; }
    var /= D;
    float inv = rsqrtf(var + eps);
    for (int i = 0; i < D; ++i) out[i] = (in[i] - mean) * inv * g[i] + be[i];
}

torch::Tensor LayerNorm_Reduction(torch::Tensor x, torch::Tensor gamma, torch::Tensor beta, double eps) {
    int B = x.size(0), N = x.size(1), D = x.size(2);
    auto out = torch::zeros_like(x);
    int rows = B * N;
    int threads = 256, blocks = (rows + threads - 1) / threads;
    layernorm_kernel<<<blocks, threads>>>(x.data_ptr<float>(), gamma.data_ptr<float>(), beta.data_ptr<float>(), out.data_ptr<float>(), rows, D, (float)eps);
    return out;
}
