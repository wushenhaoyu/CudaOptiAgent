#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <cstdint>
#include <climits>

__global__ void patch_embedding_gemm_kernel(
    const float* img, const float* w, const float* b, float* out,
    int64_t B, int64_t C, int64_t H, int64_t W, int64_t P, int64_t Np, int64_t patch_dim, int64_t D) {

    int64_t idx = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
    int64_t total = B * Np * D;
    if (idx >= total) return;

    int64_t d = idx % D;
    int64_t np = (idx / D) % Np;
    int64_t bs = idx / (Np * D);

    int64_t nH = H / P;
    int64_t nW = W / P;

    float bias_val = 0.f;
    if (b != nullptr) bias_val = b[d];
    float sum = bias_val;

    for (int64_t k = 0; k < patch_dim; ++k) {
        int64_t global_k = np * patch_dim + k;

        int64_t ix = global_k % P;
        int64_t rem = global_k / P;
        int64_t iy = rem % P;
        rem /= P;
        int64_t nw = rem % nW;
        rem /= nW;
        int64_t nh = rem % nH;
        int64_t c = rem / nH;

        int64_t in_y = nh * P + iy;
        int64_t in_x = nw * P + ix;
        int64_t in_idx = ((bs * C + c) * H + in_y) * W + in_x;

        sum += img[in_idx] * w[d * patch_dim + k];
    }

    out[idx] = sum;
}

torch::Tensor PatchEmbedding_GEMM(torch::Tensor img, torch::Tensor weight, torch::Tensor bias, int64_t patch_size) {
    TORCH_CHECK(img.is_cuda(), "img must be a CUDA tensor");
    TORCH_CHECK(weight.is_cuda(), "weight must be a CUDA tensor");
    TORCH_CHECK(!bias.defined() || bias.is_cuda(), "bias must be a CUDA tensor when defined");

    TORCH_CHECK(
        img.scalar_type() == torch::kFloat32 &&
        weight.scalar_type() == torch::kFloat32 &&
        (!bias.defined() || bias.scalar_type() == torch::kFloat32),
        "tensors must be float32"
    );

    TORCH_CHECK(img.dim() == 4, "img must have shape [B, C, H, W]");
    TORCH_CHECK(weight.dim() == 2, "weight must have shape [D, patch_dim]");

    auto img_contig = img.contiguous();
    auto w_contig = weight.contiguous();

    int64_t B = img_contig.size(0);
    int64_t C = img_contig.size(1);
    int64_t H = img_contig.size(2);
    int64_t W = img_contig.size(3);
    int64_t P = patch_size;

    TORCH_CHECK(P > 0, "patch_size must be > 0");
    TORCH_CHECK(H % P == 0 && W % P == 0, "patch_size must divide H and W");

    int64_t Np = (H / P) * (W / P);
    int64_t patch_dim = C * P * P;
    int64_t D = w_contig.size(0);

    TORCH_CHECK(w_contig.size(1) == patch_dim,
                "weight second dimension must equal C * patch_size * patch_size");

    const float* bptr = nullptr;
    torch::Tensor b_contig;
    if (bias.defined() && bias.numel() > 0) {
        TORCH_CHECK(bias.dim() == 1, "bias must be 1D when provided");
        TORCH_CHECK(bias.size(0) == D, "bias size must match weight.size(0)");
        b_contig = bias.contiguous();
        bptr = b_contig.data_ptr<float>();
    }

    auto out = torch::zeros({B, Np, D}, img_contig.options());

    int64_t total = B * Np * D;
    int threads = 256;
    int64_t blocks_ll = (total + threads - 1) / threads;

    TORCH_CHECK(blocks_ll <= INT_MAX, "Kernel launch blocks exceed INT_MAX");
    int blocks = static_cast<int>(blocks_ll);

    patch_embedding_gemm_kernel<<<blocks, threads>>>(
        img_contig.data_ptr<float>(),
        w_contig.data_ptr<float>(),
        bptr,
        out.data_ptr<float>(),
        B, C, H, W, P, Np, patch_dim, D
    );

    cudaError_t launch_err = cudaGetLastError();
    TORCH_CHECK(launch_err == cudaSuccess, "patch_embedding_gemm_kernel launch failed: ", cudaGetErrorString(launch_err));

    cudaError_t err = cudaDeviceSynchronize();
    TORCH_CHECK(err == cudaSuccess, "patch_embedding_gemm_kernel execution failed: ", cudaGetErrorString(err));

    return out;
}