#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

__device__ float gelu_head(float x) {
    return 0.5f * x * (1.0f + erff(x * 0.70710678118f));
}

__global__ void head_fc1_gelu_kernel(const float* x, const float* w, const float* b, float* y, int B, int D, int M) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = B * M;
    if (idx >= total) return;

    int m = idx % M;
    int bs = idx / M;

    float sum = b ? b[m] : 0.f;
    const float* xv = x + bs * D;
    const float* ww = w + m * D;
    for (int k = 0; k < D; ++k) sum += xv[k] * ww[k];
    y[idx] = gelu_head(sum);
}

torch::Tensor Head_FC1_GEMM_plus_GELU(torch::Tensor x, torch::Tensor w, torch::Tensor b) {
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(w.is_cuda(), "w must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "x must be float32");
    TORCH_CHECK(w.dtype() == torch::kFloat32, "w must be float32");
    TORCH_CHECK(x.dim() == 2, "x must be 2D");
    TORCH_CHECK(w.dim() == 2, "w must be 2D");

    auto x_cont = x.contiguous();
    int B = x_cont.size(0);
    int D = x_cont.size(1);

    auto w_cont = w.contiguous();
    TORCH_CHECK(w_cont.size(1) == D, "w.size(1) must match x.size(1)");
    int M = w_cont.size(0);

    const float* b_ptr = nullptr;
    if (b.defined() && b.numel() > 0) {
        TORCH_CHECK(b.is_cuda(), "b must be a CUDA tensor when provided");
        TORCH_CHECK(b.dtype() == torch::kFloat32, "b must be float32 when provided");
        TORCH_CHECK(b.dim() == 1, "b must be 1D when provided");
        TORCH_CHECK(b.size(0) == M, "b.size(0) must match w.size(0)");
        auto b_cont = b.contiguous();
        b_ptr = b_cont.data_ptr<float>();

        auto out = torch::zeros({B, M}, x_cont.options());
        int total = B * M;
        int threads = 256, blocks = (total + threads - 1) / threads;
        head_fc1_gelu_kernel<<<blocks, threads>>>(
            x_cont.data_ptr<float>(),
            w_cont.data_ptr<float>(),
            b_ptr,
            out.data_ptr<float>(),
            B, D, M
        );
        cudaError_t err = cudaGetLastError();
        TORCH_CHECK(err == cudaSuccess, "head_fc1_gelu_kernel launch failed: ", cudaGetErrorString(err));
        return out;
    } else {
        auto out = torch::zeros({B, M}, x_cont.options());
        int total = B * M;
        int threads = 256, blocks = (total + threads - 1) / threads;
        head_fc1_gelu_kernel<<<blocks, threads>>>(
            x_cont.data_ptr<float>(),
            w_cont.data_ptr<float>(),
            nullptr,
            out.data_ptr<float>(),
            B, D, M
        );
        cudaError_t err = cudaGetLastError();
        TORCH_CHECK(err == cudaSuccess, "head_fc1_gelu_kernel launch failed: ", cudaGetErrorString(err));
        return out;
    }
}