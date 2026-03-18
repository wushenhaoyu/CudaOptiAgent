#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void qkv_proj_kernel(const float* x, const float* w, const float* b, float* out, int B, int N, int D) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int O = 3 * D;
    int total = B * N * O;
    if (idx >= total) return;
    int o = idx % O;
    int n = (idx / O) % N;
    int bs = idx / (N * O);

    float sum = b ? b[o] : 0.f;
    const float* xv = x + (bs * N + n) * D;
    const float* ww = w + o * D;
    for (int k = 0; k < D; ++k) sum += xv[k] * ww[k];
    out[idx] = sum;
}

torch::Tensor SelfAttn_QKV_Proj(torch::Tensor x, torch::Tensor weight, torch::Tensor bias) {
    int B = x.size(0), N = x.size(1), D = x.size(2);
    auto out = torch::zeros({B, N, 3 * D}, x.options());
    int total = B * N * 3 * D;
    int threads = 256, blocks = (total + threads - 1) / threads;
    qkv_proj_kernel<<<blocks, threads>>>(x.data_ptr<float>(), weight.data_ptr<float>(), bias.data_ptr<float>(), out.data_ptr<float>(), B, N, D);
    return out;
}
