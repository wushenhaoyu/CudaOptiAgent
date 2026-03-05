#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <cmath>

namespace {

__global__ void layernorm_channel_reduction_kernel(
    const float* __restrict__ x,      // [N, C, D, H, W]
    const float* __restrict__ gamma,  // [C]
    const float* __restrict__ beta,   // [C]
    float* __restrict__ y,            // [N, C, D, H, W]
    int N, int C, int D, int H, int W,
    float eps
) {
    int linear = blockIdx.x * blockDim.x + threadIdx.x;
    int nhw = N * D * H * W;
    if (linear >= nhw) return;

    int t = linear;
    int w = t % W; t /= W;
    int h = t % H; t /= H;
    int d = t % D; t /= D;
    int n = t;

    float mean = 0.0f;
    for (int c = 0; c < C; ++c) {
        int idx = ((((n * C + c) * D + d) * H + h) * W + w);
        mean += x[idx];
    }
    mean /= (float)C;

    float var = 0.0f;
    for (int c = 0; c < C; ++c) {
        int idx = ((((n * C + c) * D + d) * H + h) * W + w);
        float v = x[idx] - mean;
        var += v * v;
    }
    var /= (float)C;
    float inv_std = rsqrtf(var + eps);

    for (int c = 0; c < C; ++c) {
        int idx = ((((n * C + c) * D + d) * H + h) * W + w);
        float norm = (x[idx] - mean) * inv_std;
        y[idx] = norm * gamma[c] + beta[c];
    }
}

} // namespace

torch::Tensor layernorm_channel_reduction(
    torch::Tensor x,
    torch::Tensor weight,
    torch::Tensor bias,
    std::vector<int64_t> normalized_shape,
    double eps
) {
    TORCH_CHECK(x.is_cuda(), "x must be CUDA");
    TORCH_CHECK(weight.is_cuda(), "weight must be CUDA");
    TORCH_CHECK(bias.is_cuda(), "bias must be CUDA");
    TORCH_CHECK(x.scalar_type() == torch::kFloat32, "x must be float32");
    TORCH_CHECK(weight.scalar_type() == torch::kFloat32, "weight must be float32");
    TORCH_CHECK(bias.scalar_type() == torch::kFloat32, "bias must be float32");
    TORCH_CHECK(normalized_shape.size() == 1, "normalized_shape must be [C]");

    auto x_c = x.contiguous();
    auto w_c = weight.contiguous();
    auto b_c = bias.contiguous();

    int N = (int)x_c.size(0);
    int C = (int)x_c.size(1);
    int D = (int)x_c.size(2);
    int H = (int)x_c.size(3);
    int W = (int)x_c.size(4);

    TORCH_CHECK((int)normalized_shape[0] == C, "normalized_shape mismatch with channels");

    auto y = torch::empty_like(x_c);

    int total = N * D * H * W;
    int threads = 128;
    int blocks = (total + threads - 1) / threads;

    layernorm_channel_reduction_kernel<<<blocks, threads, 0, at::cuda::getDefaultCUDAStream()>>>(
        x_c.data_ptr<float>(),
        w_c.data_ptr<float>(),
        b_c.data_ptr<float>(),
        y.data_ptr<float>(),
        N, C, D, H, W,
        (float)eps
    );

    return y;
}
