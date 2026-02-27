
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include <vector>
#include <cmath>

#define BLOCK_SIZE 256

__device__ __forceinline__ float gelu_exact(float x) {
    const float sqrt_2 = 1.4142135623730951f;
    return 0.5f * x * (1.0f + erff(x / sqrt_2));
}

// Kernel: compute ConvTranspose3d forward (no bias added here)
__global__ void conv_transpose3d_kernel(
    const float* __restrict__ input,    // [N, C_in, D_in, H_in, W_in]
    const float* __restrict__ weight,   // [C_in, C_out_group, kD, kH, kW]
    float* __restrict__ output,         // [N, C_out, D_out, H_out, W_out]
    int N,
    int C_in,
    int C_out,
    int C_out_group,
    int groups,
    int D_in, int H_in, int W_in,
    int D_out, int H_out, int W_out,
    int kD, int kH, int kW,
    int stride_d, int stride_h, int stride_w,
    int pad_d, int pad_h, int pad_w
) {
    long long total = (long long)N * C_out * D_out * H_out * W_out;
    long long tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= total) return;

    long long tmp = tid;
    int x = tmp % W_out; tmp /= W_out;
    int y = tmp % H_out; tmp /= H_out;
    int z = tmp % D_out; tmp /= D_out;
    int co = tmp % C_out; tmp /= C_out;
    int n = (int)tmp;

    int group = co / C_out_group;
    int co_group_idx = co % C_out_group;
    int C_in_group = C_in / groups;

    long long in_stride_c = (long long)D_in * H_in * W_in;
    long long in_stride_n = (long long)C_in * in_stride_c;

    long long out_stride_c = (long long)D_out * H_out * W_out;
    long long out_stride_n = (long long)C_out * out_stride_c;

    long long out_index_base = (long long)n * out_stride_n + (long long)co * out_stride_c + (long long)z * (long long)H_out * (long long)W_out + (long long)y * (long long)W_out + (long long)x;

    float acc = 0.0f;

    for (int kd = 0; kd < kD; ++kd) {
        int numer_d = z + pad_d - kd;
        if (numer_d % stride_d != 0) continue;
        int id = numer_d / stride_d;
        if (id < 0 || id >= D_in) continue;

        for (int kh = 0; kh < kH; ++kh) {
            int numer_h = y + pad_h - kh;
            if (numer_h % stride_h != 0) continue;
            int ih = numer_h / stride_h;
            if (ih < 0 || ih >= H_in) continue;

            for (int kw = 0; kw < kW; ++kw) {
                int numer_w = x + pad_w - kw;
                if (numer_w % stride_w != 0) continue;
                int iw = numer_w / stride_w;
                if (iw < 0 || iw >= W_in) continue;

                for (int ci_local = 0; ci_local < C_in_group; ++ci_local) {
                    int ci = group * C_in_group + ci_local;

                    long long in_index = (long long)n * in_stride_n + (long long)ci * in_stride_c + (long long)id * (long long)H_in * (long long)W_in + (long long)ih * (long long)W_in + (long long)iw;

                    long long w_index = ((long long)ci * (long long)C_out_group + (long long)co_group_idx) * (long long)(kD * kH * kW)
                                        + (long long)kd * (long long)(kH * kW)
                                        + (long long)kh * (long long)kW
                                        + (long long)kw;

                    acc += input[in_index] * weight[w_index];
                }
            }
        }
    }

    output[out_index_base] = acc;
}

// LayerNorm over last 1 dim: [ ..., W ]
__global__ void add_bias_sum_layernorm_L1_kernel(
    const float* __restrict__ x,    // [N, C, D, H, W]
    const float* __restrict__ bias, // [C]
    float sum_scalar,
    const float* __restrict__ gamma, // [W]
    const float* __restrict__ beta,  // [W]
    float* __restrict__ y,          // [N, C, D, H, W]
    int N, int C, int D, int H, int W,
    float eps
) {
    long long outer = (long long)N * C * D * H; // rows
    long long tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= outer) return;

    long long tmp = tid;
    int h = tmp % H; tmp /= H;
    int d = tmp % D; tmp /= D;
    int c = tmp % C; tmp /= C;
    int n = (int)tmp;

    long long stride_W = 1;
    long long stride_H = W;
    long long stride_D = (long long)H * W;
    long long stride_C = (long long)D * H * W;
    long long stride_N = (long long)C * stride_C;

    long long base = (long long)n * stride_N + (long long)c * stride_C + (long long)d * stride_D + (long long)h * stride_H;

    float b = bias[c];
    float mean = 0.0f;
    for (int w = 0; w < W; ++w) {
        float v = x[base + w * stride_W] + b + sum_scalar;
        mean += v;
    }
    mean /= (float)W;

    float var = 0.0f;
    for (int w = 0; w < W; ++w) {
        float v = x[base + w * stride_W] + b + sum_scalar;
        float dlt = v - mean;
        var += dlt * dlt;
    }
    var /= (float)W;
    float inv_std = rsqrtf(var + eps);

    for (int w = 0; w < W; ++w) {
        float v = x[base + w * stride_W] + b + sum_scalar;
        float nrm = (v - mean) * inv_std;
        y[base + w * stride_W] = nrm * gamma[w] + beta[w];
    }
}

// LayerNorm over last 2 dims: [ ..., H, W ]
__global__ void add_bias_sum_layernorm_L2_kernel(
    const float* __restrict__ x,    // [N, C, D, H, W]
    const float* __restrict__ bias, // [C]
    float sum_scalar,
    const float* __restrict__ gamma, // [H, W] flattened row-major
    const float* __restrict__ beta,  // [H, W]
    float* __restrict__ y,          // [N, C, D, H, W]
    int N, int C, int D, int H, int W,
    float eps
) {
    long long outer = (long long)N * C * D;
    long long tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= outer) return;

    long long tmp = tid;
    int d = tmp % D; tmp /= D;
    int c = tmp % C; tmp /= C;
    int n = (int)tmp;

    long long stride_W = 1;
    long long stride_H = W;
    long long stride_D = (long long)H * W;
    long long stride_C = (long long)D * H * W;
    long long stride_N = (long long)C * stride_C;

    float b = bias[c];

    long long base = (long long)n * stride_N + (long long)c * stride_C + (long long)d * stride_D;

    int S = H * W;

    float mean = 0.0f;
    for (int idx = 0; idx < S; ++idx) {
        int h = idx / W;
        int w = idx % W;
        long long off = base + (long long)h * stride_H + (long long)w * stride_W;
        float v = x[off] + b + sum_scalar;
        mean += v;
    }
    mean /= (float)S;

    float var = 0.0f;
    for (int idx = 0; idx < S; ++idx) {
        int h = idx / W;
        int w = idx % W;
        long long off = base + (long long)h * stride_H + (long long)w * stride_W;
        float v = x[off] + b + sum_scalar;
        float dlt = v - mean;
        var += dlt * dlt;
    }
    var /= (float)S;
    float inv_std = rsqrtf(var + eps);

    for (int idx = 0; idx < S; ++idx) {
        int h = idx / W;
        int w = idx % W;
        long long off = base + (long long)h * stride_H + (long long)w * stride_W;
        float v = x[off] + b + sum_scalar;
        float nrm = (v - mean) * inv_std;
        float g = gamma[idx];
        float be = beta[idx];
        y[off] = nrm * g + be;
    }
}

// LayerNorm over last 3 dims: [ ..., D, H, W ]
__global__ void add_bias_sum_layernorm_L3_kernel(
    const float* __restrict__ x,    // [N, C, D, H, W]
    const float* __restrict__ bias, // [C]
    float sum_scalar,
    const float* __restrict__ gamma, // [D, H, W] flattened
    const float* __restrict__ beta,  // [D, H, W]
    float* __restrict__ y,          // [N, C, D, H, W]
    int N, int C, int D, int H, int W,
    float eps
) {
    long long outer = (long long)N * C;
    long long tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= outer) return;

    int c = tid % C;
    int n = (int)(tid / C);

    long long stride_W = 1;
    long long stride_H = W;
    long long stride_D = (long long)H * W;
    long long stride_C = (long long)D * H * W;
    long long stride_N = (long long)C * stride_C;

    long long base = (long long)n * stride_N + (long long)c * stride_C;

    int S = D * H * W;
    float b = bias[c];

    float mean = 0.0f;
    for (int idx = 0; idx < S; ++idx) {
        int d = idx / (H * W);
        int rem = idx % (H * W);
        int h = rem / W;
        int w = rem % W;
        long long off = base + (long long)d * stride_D + (long long)h * stride_H + (long long)w * stride_W;
        float v = x[off] + b + sum_scalar;
        mean += v;
    }
    mean /= (float)S;

    float var = 0.0f;
    for (int idx = 0; idx < S; ++idx) {
        int d = idx / (H * W);
        int rem = idx % (H * W);
        int h = rem / W;
        int w = rem % W;
        long long off = base + (long long)d * stride_D + (long long)h * stride_H + (long long)w * stride_W;
        float v = x[off] + b + sum_scalar;
        float dlt = v - mean;
        var += dlt * dlt;
    }
    var /= (float)S;
    float inv_std = rsqrtf(var + eps);

    for (int idx = 0; idx < S; ++idx) {
        int d = idx / (H * W);
        int rem = idx % (H * W);
        int h = rem / W;
        int w = rem % W;
        long long off = base + (long long)d * stride_D + (long long)h * stride_H + (long long)w * stride_W;
        float v = x[off] + b + sum_scalar;
        float nrm = (v - mean) * inv_std;
        float g = gamma[idx];
        float be = beta[idx];
        y[off] = nrm * g + be;
    }
}

// LayerNorm over last 4 dims: [ ..., C, D, H, W ]
__global__ void add_bias_sum_layernorm_L4_kernel(
    const float* __restrict__ x,    // [N, C, D, H, W]
    const float* __restrict__ bias, // [C]
    float sum_scalar,
    const float* __restrict__ gamma, // [C, D, H, W] flattened
    const float* __restrict__ beta,  // [C, D, H, W]
    float* __restrict__ y,          // [N, C, D, H, W]
    int N, int C, int D, int H, int W,
    float eps
) {
    long long outer = (long long)N;
    long long tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= outer) return;

    int n = (int)tid;

    long long stride_W = 1;
    long long stride_H = W;
    long long stride_D = (long long)H * W;
    long long stride_C = (long long)D * H * W;
    long long stride_N = (long long)C * stride_C;

    long long base = (long long)n * stride_N;

    int S = C * D * H * W;

    float mean = 0.0f;
    for (int idx = 0; idx < S; ++idx) {
        int c = idx / (D * H * W);
        int rem_cd = idx % (D * H * W);
        int d = rem_cd / (H * W);
        int rem_dh = rem_cd % (H * W);
        int h = rem_dh / W;
        int w = rem_dh % W;
        long long off = base + (long long)c * stride_C + (long long)d * stride_D + (long long)h * stride_H + (long long)w * stride_W;
        float v = x[off] + bias[c] + sum_scalar;
        mean += v;
    }
    mean /= (float)S;

    float var = 0.0f;
    for (int idx = 0; idx < S; ++idx) {
        int c = idx / (D * H * W);
        int rem_cd = idx % (D * H * W);
        int d = rem_cd / (H * W);
        int rem_dh = rem_cd % (H * W);
        int h = rem_dh / W;
        int w = rem_dh % W;
        long long off = base + (long long)c * stride_C + (long long)d * stride_D + (long long)h * stride_H + (long long)w * stride_W;
        float v = x[off] + bias[c] + sum_scalar;
        float dlt = v - mean;
        var += dlt * dlt;
    }
    var /= (float)S;
    float inv_std = rsqrtf(var + eps);

    for (int idx = 0; idx < S; ++idx) {
        int c = idx / (D * H * W);
        int rem_cd = idx % (D * H * W);
        int d = rem_cd / (H * W);
        int rem_dh = rem_cd % (H * W);
        int h = rem_dh / W;
        int w = rem_dh % W;
        long long off = base + (long long)c * stride_C + (long long)d * stride_D + (long long)h * stride_H + (long long)w * stride_W;
        float v = x[off] + bias[c] + sum_scalar;
        float nrm = (v - mean) * inv_std;
        float g = gamma[idx];
        float be = beta[idx];
        y[off] = nrm * g + be;
    }
}

// Kernel: AvgPool3d followed by GELU
__global__ void avgpool3d_gelu_kernel(
    const float* __restrict__ norm_out,  // [N, C, D, H, W]
    float* __restrict__ output,          // [N, C, Dp, Hp, Wp]
    int N, int C, int D, int H, int W,
    int kD, int kH, int kW,
    int sD, int sH, int sW,
    int Dp, int Hp, int Wp
) {
    long long total = (long long)N * C * Dp * Hp * Wp;
    long long tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= total) return;

    long long tmp = tid;
    int x = tmp % Wp; tmp /= Wp;
    int y = tmp % Hp; tmp /= Hp;
    int z = tmp % Dp; tmp /= Dp;
    int c = tmp % C;  tmp /= C;
    int n = (int)tmp;

    long long in_stride_c = (long long)D * H * W;
    long long in_stride_n = (long long)C * in_stride_c;

    int z0 = z * sD;
    int y0 = y * sH;
    int x0 = x * sW;

    float sum = 0.0f;
    for (int kd = 0; kd < kD; ++kd) {
        int iz = z0 + kd;
        for (int kh = 0; kh < kH; ++kh) {
            int iy = y0 + kh;
            for (int kw = 0; kw < kW; ++kw) {
                int ix = x0 + kw;
                long long in_index = (long long)n * in_stride_n + (long long)c * in_stride_c + (long long)iz * (long long)H * (long long)W + (long long)iy * (long long)W + (long long)ix;
                sum += norm_out[in_index];
            }
        }
    }
    float avg = sum / (float)(kD * kH * kW);
    output[tid] = gelu_exact(avg);
}

torch::Tensor ConvTranspose3d_Sum_LayerNorm_AvgPool_GELU(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    c10::IntArrayRef kernel_size,      // not used directly; inferred from weight
    c10::IntArrayRef stride,
    c10::IntArrayRef padding,
    c10::IntArrayRef output_padding,
    int64_t groups,
    torch::Tensor sum_weight,
    torch::Tensor norm_weight,
    torch::Tensor norm_bias,
    double norm_eps,
    c10::IntArrayRef pool_kernel_size,
    c10::IntArrayRef pool_stride
) {
    TORCH_CHECK(input.is_cuda(), "input must be CUDA tensor");
    TORCH_CHECK(weight.is_cuda(), "weight must be CUDA tensor");
    TORCH_CHECK(bias.is_cuda(), "bias must be CUDA tensor");
    TORCH_CHECK(norm_weight.is_cuda(), "norm_weight must be CUDA tensor");
    TORCH_CHECK(norm_bias.is_cuda(), "norm_bias must be CUDA tensor");

    TORCH_CHECK(input.dtype() == torch::kFloat32, "Only float32 is supported");
    TORCH_CHECK(weight.dtype() == torch::kFloat32, "Only float32 weight is supported");
    TORCH_CHECK(bias.dtype() == torch::kFloat32, "Only float32 bias is supported");
    TORCH_CHECK(norm_weight.dtype() == torch::kFloat32, "Only float32 norm_weight is supported");
    TORCH_CHECK(norm_bias.dtype() == torch::kFloat32, "Only float32 norm_bias is supported");

    auto in = input.contiguous();
    auto w = weight.contiguous();
    auto b = bias.contiguous();
    auto gamma = norm_weight.contiguous();
    auto beta = norm_bias.contiguous();

    int N = in.size(0);
    int C_in = in.size(1);
    int D_in = in.size(2);
    int H_in = in.size(3);
    int W_in = in.size(4);

    int kD = w.size(2);
    int kH = w.size(3);
    int kW = w.size(4);

    int sD = stride.size() >= 1 ? (int)stride[0] : 1;
    int sH = stride.size() >= 2 ? (int)stride[1] : 1;
    int sW = stride.size() >= 3 ? (int)stride[2] : 1;

    int pD = padding.size() >= 1 ? (int)padding[0] : 0;
    int pH = padding.size() >= 2 ? (int)padding[1] : 0;
    int pW = padding.size() >= 3 ? (int)padding[2] : 0;

    int oD = output_padding.size() >= 1 ? (int)output_padding[0] : 0;
    int oH = output_padding.size() >= 2 ? (int)output_padding[1] : 0;
    int oW = output_padding.size() >= 3 ? (int)output_padding[2] : 0;

    int C_out_group = w.size(1);
    int C_out = C_out_group * (int)groups;

    TORCH_CHECK(C_in % groups == 0, "C_in must be divisible by groups");
    TORCH_CHECK(b.numel() == C_out, "ConvTranspose bias size must match out channels");

    int D_out = (D_in - 1) * sD - 2 * pD + kD + oD;
    int H_out = (H_in - 1) * sH - 2 * pH + kH + oH;
    int W_out = (W_in - 1) * sW - 2 * pW + kW + oW;

    TORCH_CHECK(gamma.dim() >= 1 && gamma.dim() <= 4, "LayerNorm normalized_shape rank must be 1..4 for 5D input");
    TORCH_CHECK(beta.dim() == gamma.dim(), "LayerNorm weight/bias must have same rank");
    TORCH_CHECK(beta.sizes().vec() == gamma.sizes().vec(), "LayerNorm weight/bias shapes must match");

    int L = gamma.dim();
    // Validate that gamma matches last L dims of [N, C_out, D_out, H_out, W_out]
    std::vector<int64_t> last_dims = {C_out, D_out, H_out, W_out};
    for (int i = 0; i < L; ++i) {
        TORCH_CHECK(gamma.size(i) == last_dims[4 - L + i - 1 + 1], "LayerNorm normalized_shape must match the last dimensions of the input");
        // The expression simplifies to last_dims[i + (4 - L + 1) - 1] but we keep vector simple:
        // We'll map indices explicitly below when launching kernels.
    }

    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(in.device());

    auto conv_out = torch::empty({N, C_out, D_out, H_out, W_out}, options);
    auto norm_out = torch::empty({N, C_out, D_out, H_out, W_out}, options);

    int pkD = pool_kernel_size.size() >= 1 ? (int)pool_kernel_size[0] : 1;
    int pkH = pool_kernel_size.size() >= 2 ? (int)pool_kernel_size[1] : 1;
    int pkW = pool_kernel_size.size() >= 3 ? (int)pool_kernel_size[2] : 1;

    int psD = pool_stride.size() >= 1 ? (int)pool_stride[0] : pkD;
    int psH = pool_stride.size() >= 2 ? (int)pool_stride[1] : pkH;
    int psW = pool_stride.size() >= 3 ? (int)pool_stride[2] : pkW;

    int Dp = (D_out - pkD) / psD + 1;
    int Hp = (H_out - pkH) / psH + 1;
    int Wp = (W_out - pkW) / psW + 1;

    auto output = torch::empty({N, C_out, Dp, Hp, Wp}, options);

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();

    {
        long long total = (long long)N * C_out * D_out * H_out * W_out;
        dim3 block(BLOCK_SIZE);
        dim3 grid((unsigned int)((total + BLOCK_SIZE - 1) / BLOCK_SIZE));
        conv_transpose3d_kernel<<<grid, block, 0, stream>>>(
            in.data_ptr<float>(),
            w.data_ptr<float>(),
            conv_out.data_ptr<float>(),
            N,
            C_in,
            C_out,
            C_out_group,
            (int)groups,
            D_in, H_in, W_in,
            D_out, H_out, W_out,
            kD, kH, kW,
            sD, sH, sW,
            pD, pH, pW
        );
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    {
        float sum_scalar = 0.0f;
        if (sum_weight.numel() >= 1) {
            sum_scalar = sum_weight.item<float>();
        }
        dim3 block(BLOCK_SIZE);
        if (L == 1) {
            TORCH_CHECK(gamma.size(0) == W_out, "normalized_shape mismatch for last 1 dim (W)");
            long long outer = (long long)N * C_out * D_out * H_out;
            dim3 grid((unsigned int)((outer + BLOCK_SIZE - 1) / BLOCK_SIZE));
            add_bias_sum_layernorm_L1_kernel<<<grid, block, 0, stream>>>(
                conv_out.data_ptr<float>(),
                b.data_ptr<float>(),
                sum_scalar,
                gamma.data_ptr<float>(),
                beta.data_ptr<float>(),
                norm_out.data_ptr<float>(),
                N, C_out, D_out, H_out, W_out,
                (float)norm_eps
            );
        } else if (L == 2) {
            TORCH_CHECK(gamma.size(0) == H_out && gamma.size(1) == W_out, "normalized_shape mismatch for last 2 dims (H, W)");
            long long outer = (long long)N * C_out * D_out;
            dim3 grid((unsigned int)((outer + BLOCK_SIZE - 1) / BLOCK_SIZE));
            add_bias_sum_layernorm_L2_kernel<<<grid, block, 0, stream>>>(
                conv_out.data_ptr<float>(),
                b.data_ptr<float>(),
                sum_scalar,
                gamma.data_ptr<float>(),
                beta.data_ptr<float>(),
                norm_out.data_ptr<float>(),
                N, C_out, D_out, H_out, W_out,
                (float)norm_eps
            );
        } else if (L == 3) {
            TORCH_CHECK(gamma.size(0) == D_out && gamma.size(1) == H_out && gamma.size(2) == W_out, "normalized_shape mismatch for last 3 dims (D, H, W)");
            long long outer = (long long)N * C_out;
            dim3 grid((unsigned int)((outer + BLOCK_SIZE - 1) / BLOCK_SIZE));
            add_bias_sum_layernorm_L3_kernel<<<grid, block, 0, stream>>>(
                conv_out.data_ptr<float>(),
                b.data_ptr<float>(),
                sum_scalar,
                gamma.data_ptr<float>(),
                beta.data_ptr<float>(),
                norm_out.data_ptr<float>(),
                N, C_out, D_out, H_out, W_out,
                (float)norm_eps
            );
        } else { // L == 4
            TORCH_CHECK(gamma.size(0) == C_out && gamma.size(1) == D_out && gamma.size(2) == H_out && gamma.size(3) == W_out, "normalized_shape mismatch for last 4 dims (C, D, H, W)");
            long long outer = (long long)N;
            dim3 grid((unsigned int)((outer + BLOCK_SIZE - 1) / BLOCK_SIZE));
            add_bias_sum_layernorm_L4_kernel<<<grid, block, 0, stream>>>(
                conv_out.data_ptr<float>(),
                b.data_ptr<float>(),
                sum_scalar,
                gamma.data_ptr<float>(),
                beta.data_ptr<float>(),
                norm_out.data_ptr<float>(),
                N, C_out, D_out, H_out, W_out,
                (float)norm_eps
            );
        }
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    {
        long long total = (long long)N * C_out * Dp * Hp * Wp;
        dim3 block(BLOCK_SIZE);
        dim3 grid((unsigned int)((total + BLOCK_SIZE - 1) / BLOCK_SIZE));
        avgpool3d_gelu_kernel<<<grid, block, 0, stream>>>(
            norm_out.data_ptr<float>(),
            output.data_ptr<float>(),
            N, C_out, D_out, H_out, W_out,
            pkD, pkH, pkW,
            psD, psH, psW,
            Dp, Hp, Wp
        );
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("ConvTranspose3d_Sum_LayerNorm_AvgPool_GELU", &ConvTranspose3d_Sum_LayerNorm_AvgPool_GELU, "Fused ConvTranspose3d + Sum + LayerNorm + AvgPool3d + GELU");
}