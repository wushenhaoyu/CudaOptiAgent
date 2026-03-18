#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void cls_posadd_kernel(
    const float* x, const float* cls, const float* pos, float* out,
    int B, int Np, int D) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int N = Np + 1;
    int total = B * N * D;
    if (idx >= total) return;
    int d = idx % D;
    int n = (idx / D) % N;
    int b = idx / (N * D);

    float v = (n == 0) ? cls[d] : x[(b * Np + (n - 1)) * D + d];
    out[idx] = v + pos[n * D + d];
}

torch::Tensor CLS_PosAdd_Elementwise(torch::Tensor x, torch::Tensor cls_token, torch::Tensor pos_embedding) {
    int B = x.size(0), Np = x.size(1), D = x.size(2);
    auto out = torch::zeros({B, Np + 1, D}, x.options());
    int total = B * (Np + 1) * D;
    int threads = 256, blocks = (total + threads - 1) / threads;
    cls_posadd_kernel<<<blocks, threads>>>(
        x.data_ptr<float>(), cls_token.data_ptr<float>(), pos_embedding.data_ptr<float>(), out.data_ptr<float>(),
        B, Np, D);
    return out;
}
