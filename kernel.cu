
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include <vector>
#include <math_constants.h>

#define THREADS_PER_BLOCK 256

// Utility: GELU approximation (GeLU(x) = 0.5 * x * (1 + tanh(sqrt(2/pi)*(x + 0.044715*x^3))))
__device__ __forceinline__ float gelu_approx(float x) {
    const float kAlpha = 0.7978845608028654f; // sqrt(2/pi)
    const float kBeta = 0.044715f;
    float x3 = x * x * x;
    float tanh_arg = kAlpha * (x + kBeta * x3);
    float tanh_res = tanhf(tanh_arg);
    return 0.5f * x * (1.0f + tanh_res);
}

// Kernel 1: ConvTranspose3d forward
// Input: input [N, C_in, D_in, H_in, W_in]
// Weight: [C_in, C_out, kD, kH, kW]
// Bias: [C_out] or nullptr
// Output: [N, C_out, D_out, H_out, W_out]
// Parameters: stride, padding, output_padding (all 3D tuples)
// We implement naive direct conv_transpose3d (backward conv) without shared memory or advanced tiling for correctness.
// Each thread computes one output element (n, c_out, d_out, h_out, w_out).
// We parallelize over output elements.
__global__ void conv_transpose3d_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int N,
    int C_in,
    int D_in,
    int H_in,
    int W_in,
    int C_out,
    int kD,
    int kH,
    int kW,
    int D_out,
    int H_out,
    int W_out,
    int stride_d,
    int stride_h,
    int stride_w,
    int pad_d,
    int pad_h,
    int pad_w,
    int out_pad_d,
    int out_pad_h,
    int out_pad_w
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * C_out * D_out * H_out * W_out;
    if (idx >= total) return;

    // Decode idx to n, c_out, d_out, h_out, w_out
    int w_out = idx % W_out; idx /= W_out;
    int h_out = idx % H_out; idx /= H_out;
    int d_out = idx % D_out; idx /= D_out;
    int c_out = idx % C_out; idx /= C_out;
    int n = idx;

    float val = 0.0f;

    // For conv_transpose3d, output[n, c_out, d_out, h_out, w_out] =
    // sum_{c_in, kd, kh, kw} input[n, c_in, d_in, h_in, w_in] * weight[c_in, c_out, kd, kh, kw]
    // where d_in = (d_out + pad_d - kd) / stride_d, similarly for h_in, w_in
    // and d_in, h_in, w_in must be integer and in range.

    for (int c_in = 0; c_in < C_in; c_in++) {
        for (int kd = 0; kd < kD; kd++) {
            int d_in = (d_out + pad_d - kd);
            if (d_in < 0) continue;
            if (d_in % stride_d != 0) continue;
            d_in /= stride_d;
            if (d_in >= D_in) continue;

            for (int kh = 0; kh < kH; kh++) {
                int h_in = (h_out + pad_h - kh);
                if (h_in < 0) continue;
                if (h_in % stride_h != 0) continue;
                h_in /= stride_h;
                if (h_in >= H_in) continue;

                for (int kw = 0; kw < kW; kw++) {
                    int w_in = (w_out + pad_w - kw);
                    if (w_in < 0) continue;
                    if (w_in % stride_w != 0) continue;
                    w_in /= stride_w;
                    if (w_in >= W_in) continue;

                    float inp_val = input[((n * C_in + c_in) * D_in + d_in) * H_in * W_in + h_in * W_in + w_in];
                    float w_val = weight[((c_in * C_out + c_out) * kD + kd) * kH * kW + kh * kW + kw];
                    val += inp_val * w_val;
                }
            }
        }
    }

    if (bias != nullptr) {
        val += bias[c_out];
    }

    output[((n * C_out + c_out) * D_out + d_out) * H_out * W_out + h_out * W_out + w_out] = val;
}

// Kernel 2: sum_weight add (injective)
// sum_weight is a scalar float tensor (broadcasted to all elements of output)
__global__ void add_sum_weight_kernel(float* __restrict__ output, float sum_weight, int total) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    output[idx] += sum_weight;
}

// Kernel 3: LayerNorm forward
// Input and output shape: [N, C, D, H, W]
// LayerNorm over last dimension = C dimension only (norm_shape = (C,))
// Actually LayerNorm norm_shape = (C,), so normalize over channel dim only per spatial location.
// But PyTorch LayerNorm with norm_shape=(C,) normalizes over last dimension(s) of input shape.
// Here input shape is 5D: (N, C, D, H, W)
// norm_shape=(C,) means normalize over dimension 1 only? No, LayerNorm normalizes over last dims matching norm_shape.
// So norm_shape=(C,) means normalize over dimension 1 only if input shape is (N, D, H, W, C).
// But input is (N, C, D, H, W).
// So we must permute or treat carefully.

// According to PyTorch docs, LayerNorm normalizes over last len(norm_shape) dims.
// Here norm_shape=(C,), so normalize over last dim of input.
// Our input is (N, C, D, H, W), so last dim is W, not C.
// So the LayerNorm in the original model is applied on shape (N, C, D, H, W) with norm_shape=(C,).
// This means the input is permuted or LayerNorm is applied per spatial location over channels.

// So we must normalize over channel dimension per spatial location (d,h,w) for each batch element.

// So for each (n,d,h,w), compute mean and variance over c in [0,C-1].

// LayerNorm formula:
// y = (x - mean) / sqrt(var + eps) * weight + bias
// weight and bias shape: (C,)
// eps = 1e-5 (PyTorch default)

__global__ void layernorm_channel_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int N,
    int C,
    int D,
    int H,
    int W,
    float eps
) {
    // Each thread processes one (n,d,h,w) location, normalizing over C channels.

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * D * H * W;
    if (idx >= total) return;

    int w = idx % W; idx /= W;
    int h = idx % H; idx /= H;
    int d = idx % D; idx /= D;
    int n = idx;

    // Compute mean over C
    float mean = 0.0f;
    for (int c = 0; c < C; c++) {
        mean += input[((n * C + c) * D + d) * H * W + h * W + w];
    }
    mean /= C;

    // Compute variance
    float var = 0.0f;
    for (int c = 0; c < C; c++) {
        float val = input[((n * C + c) * D + d) * H * W + h * W + w];
        float diff = val - mean;
        var += diff * diff;
    }
    var /= C;

    float inv_std = rsqrtf(var + eps);

    // Normalize and apply weight and bias per channel
    for (int c = 0; c < C; c++) {
        float val = input[((n * C + c) * D + d) * H * W + h * W + w];
        float norm_val = (val - mean) * inv_std;
        float wgt = weight[c];
        float bs = bias[c];
        output[((n * C + c) * D + d) * H * W + h * W + w] = norm_val * wgt + bs;
    }
}

// Kernel 4: AvgPool3d forward
// Input and output shapes: input [N, C, D_in, H_in, W_in], output [N, C, D_out, H_out, W_out]
// kernel_size = (kD, kH, kW)
// stride = kernel_size (default for AvgPool3d if stride not specified)
// No padding for avgpool in this model
// Each thread computes one output element
__global__ void avgpool3d_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int N,
    int C,
    int D_in,
    int H_in,
    int W_in,
    int D_out,
    int H_out,
    int W_out,
    int kD,
    int kH,
    int kW
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * C * D_out * H_out * W_out;
    if (idx >= total) return;

    int w_out = idx % W_out; idx /= W_out;
    int h_out = idx % H_out; idx /= H_out;
    int d_out = idx % D_out; idx /= D_out;
    int c = idx % C; idx /= C;
    int n = idx;

    int d_start = d_out * kD;
    int h_start = h_out * kH;
    int w_start = w_out * kW;

    float sum = 0.0f;
    int count = 0;
    for (int kd = 0; kd < kD; kd++) {
        int d_in = d_start + kd;
        if (d_in >= D_in) continue;
        for (int kh = 0; kh < kH; kh++) {
            int h_in = h_start + kh;
            if (h_in >= H_in) continue;
            for (int kw = 0; kw < kW; kw++) {
                int w_in = w_start + kw;
                if (w_in >= W_in) continue;
                float val = input[((n * C + c) * D_in + d_in) * H_in * W_in + h_in * W_in + w_in];
                sum += val;
                count++;
            }
        }
    }
    output[idx] = sum / count;
}

// Kernel 5: GELU activation (injective)
// Input and output shape same
__global__ void gelu_kernel(float* __restrict__ data, int total) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    data[idx] = gelu_approx(data[idx]);
}

// Main fused function exposed to Python
torch::Tensor ConvTranspose3d_Sum_LayerNorm_AvgPool_GELU(
    torch::Tensor input,           // [N, C_in, D_in, H_in, W_in]
    torch::Tensor weight,          // [C_in, C_out, kD, kH, kW]
    torch::Tensor bias,            // [C_out] or None
    torch::Tensor sum_weight,      // scalar tensor
    torch::Tensor norm_weight,     // [C_out]
    torch::Tensor norm_bias,       // [C_out]
    c10::IntArrayRef pool_kernel_size, // (kD, kH, kW)
    c10::IntArrayRef stride,       // (stride_d, stride_h, stride_w)
    c10::IntArrayRef padding,      // (pad_d, pad_h, pad_w)
    c10::IntArrayRef output_padding // (out_pad_d, out_pad_h, out_pad_w)
) {
    // Check input dims
    TORCH_CHECK(input.dim() == 5, "Input must be 5D");
    TORCH_CHECK(weight.dim() == 5, "Weight must be 5D");
    TORCH_CHECK(norm_weight.dim() == 1, "norm_weight must be 1D");
    TORCH_CHECK(norm_bias.dim() == 1, "norm_bias must be 1D");
    TORCH_CHECK(sum_weight.numel() == 1, "sum_weight must be scalar");

    int N = input.size(0);
    int C_in = input.size(1);
    int D_in = input.size(2);
    int H_in = input.size(3);
    int W_in = input.size(4);

    int C_in_w = weight.size(0);
    int C_out = weight.size(1);
    int kD = weight.size(2);
    int kH = weight.size(3);
    int kW = weight.size(4);

    TORCH_CHECK(C_in == C_in_w, "Input channels and weight channels mismatch");
    TORCH_CHECK(norm_weight.size(0) == C_out, "norm_weight size mismatch");
    TORCH_CHECK(norm_bias.size(0) == C_out, "norm_bias size mismatch");

    int stride_d = stride[0];
    int stride_h = stride[1];
    int stride_w = stride[2];
    int pad_d = padding[0];
    int pad_h = padding[1];
    int pad_w = padding[2];
    int out_pad_d = output_padding[0];
    int out_pad_h = output_padding[1];
    int out_pad_w = output_padding[2];

    // Compute output spatial dims for conv_transpose3d
    int D_out = (D_in - 1) * stride_d - 2 * pad_d + kD + out_pad_d;
    int H_out = (H_in - 1) * stride_h - 2 * pad_h + kH + out_pad_h;
    int W_out = (W_in - 1) * stride_w - 2 * pad_w + kW + out_pad_w;

    // Allocate output tensor for conv_transpose3d
    auto options = torch::TensorOptions().dtype(input.dtype()).device(input.device());
    auto conv_out = torch::empty({N, C_out, D_out, H_out, W_out}, options);

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();

    // Launch conv_transpose3d kernel
    int total_conv = N * C_out * D_out * H_out * W_out;
    int blocks_conv = (total_conv + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    conv_transpose3d_kernel<<<blocks_conv, THREADS_PER_BLOCK, 0, stream>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.defined() ? bias.data_ptr<float>() : nullptr,
        conv_out.data_ptr<float>(),
        N, C_in, D_in, H_in, W_in,
        C_out, kD, kH, kW,
        D_out, H_out, W_out,
        stride_d, stride_h, stride_w,
        pad_d, pad_h, pad_w,
        out_pad_d, out_pad_h, out_pad_w
    );

    // Add sum_weight scalar to conv_out
    int total_sum = total_conv;
    int blocks_sum = (total_sum + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    add_sum_weight_kernel<<<blocks_sum, THREADS_PER_BLOCK, 0, stream>>>(
        conv_out.data_ptr<float>(),
        sum_weight.item<float>(),
        total_sum
    );

    // LayerNorm over channel dim per spatial location
    // Output tensor for layernorm
    auto norm_out = torch::empty_like(conv_out);
    int total_ln = N * D_out * H_out * W_out;
    int blocks_ln = (total_ln + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    layernorm_channel_kernel<<<blocks_ln, THREADS_PER_BLOCK, 0, stream>>>(
        conv_out.data_ptr<float>(),
        norm_weight.data_ptr<float>(),
        norm_bias.data_ptr<float>(),
        norm_out.data_ptr<float>(),
        N, C_out, D_out, H_out, W_out,
        1e-5f
    );

    // AvgPool3d forward
    int kD_pool = pool_kernel_size[0];
    int kH_pool = pool_kernel_size[1];
    int kW_pool = pool_kernel_size[2];

    int D_pool_out = D_out / kD_pool;
    int H_pool_out = H_out / kH_pool;
    int W_pool_out = W_out / kW_pool;

    auto pool_out = torch::empty({N, C_out, D_pool_out, H_pool_out, W_pool_out}, options);

    int total_pool = N * C_out * D_pool_out * H_pool_out * W_pool_out;
    int blocks_pool = (total_pool + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    avgpool3d_kernel<<<blocks_pool, THREADS_PER_BLOCK, 0, stream>>>(
        norm_out.data_ptr<float>(),
        pool_out.data_ptr<float>(),
        N, C_out,
        D_out, H_out, W_out,
        D_pool_out, H_pool_out, W_pool_out,
        kD_pool, kH_pool, kW_pool
    );

    // GELU activation inplace on pool_out
    int total_gelu = total_pool;
    int blocks_gelu = (total_gelu + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    gelu_kernel<<<blocks_gelu, THREADS_PER_BLOCK, 0, stream>>>(
        pool_out.data_ptr<float>(),
        total_gelu
    );

    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return pool_out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("ConvTranspose3d_Sum_LayerNorm_AvgPool_GELU", &ConvTranspose3d_Sum_LayerNorm_AvgPool_GELU, "Fused ConvTranspose3d + Sum + LayerNorm + AvgPool3d + GELU");
}