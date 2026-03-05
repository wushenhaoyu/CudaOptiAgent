#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <string>
#include <cmath>

namespace {

__device__ __forceinline__ float gelu_exact(float x) {
    const float inv_sqrt2 = 0.7071067811865475f;
    return 0.5f * x * (1.0f + erff(x * inv_sqrt2));
}

__device__ __forceinline__ float gelu_tanh(float x) {
    const float k0 = 0.7978845608028654f; // sqrt(2/pi)
    const float k1 = 0.044715f;
    float v = k0 * (x + k1 * x * x * x);
    return 0.5f * x * (1.0f + tanhf(v));
}

__global__ void avgpool3d_gelu_kernel(
    const float* __restrict__ x,  // [N, C, Di, Hi, Wi]
    float* __restrict__ y,        // [N, C, Do, Ho, Wo]
    int N, int C, int Di, int Hi, int Wi,
    int Do, int Ho, int Wo,
    int Kd, int Kh, int Kw,
    int Sd, int Sh, int Sw,
    int Pd, int Ph, int Pw,
    bool count_include_pad,
    int divisor_override,
    int gelu_mode // 0 exact, 1 tanh
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * C * Do * Ho * Wo;
    if (idx >= total) return;

    int t = idx;
    int ow = t % Wo; t /= Wo;
    int oh = t % Ho; t /= Ho;
    int od = t % Do; t /= Do;
    int c  = t % C;  t /= C;
    int n  = t;

    int dstart = od * Sd - Pd;
    int hstart = oh * Sh - Ph;
    int wstart = ow * Sw - Pw;
    int dend = min(dstart + Kd, Di);
    int hend = min(hstart + Kh, Hi);
    int wend = min(wstart + Kw, Wi);
    dstart = max(dstart, 0);
    hstart = max(hstart, 0);
    wstart = max(wstart, 0);

    float sum = 0.0f;
    int valid_count = max(dend - dstart, 0) * max(hend - hstart, 0) * max(wend - wstart, 0);

    for (int d = dstart; d < dend; ++d) {
        for (int h = hstart; h < hend; ++h) {
            for (int w = wstart; w < wend; ++w) {
                int x_idx = ((((n * C + c) * Di + d) * Hi + h) * Wi + w);
                sum += x[x_idx];
            }
        }
    }

    int divv;
    if (divisor_override > 0) {
        divv = divisor_override;
    } else if (count_include_pad) {
        divv = Kd * Kh * Kw;
    } else {
        divv = valid_count > 0 ? valid_count : 1;
    }

    float avg = sum / (float)divv;
    y[idx] = (gelu_mode == 1) ? gelu_tanh(avg) : gelu_exact(avg);
}

} // namespace

torch::Tensor avgpool3d_gelu(
    torch::Tensor x,
    std::vector<int64_t> kernel_size,
    std::vector<int64_t> stride,
    std::vector<int64_t> padding,
    bool ceil_mode,
    bool count_include_pad,
    c10::optional<int64_t> divisor_override,
    std::string approximate
) {
    TORCH_CHECK(x.is_cuda(), "x must be CUDA");
    TORCH_CHECK(x.scalar_type() == torch::kFloat32, "x must be float32");
    TORCH_CHECK(kernel_size.size() == 3, "kernel_size must be 3D");

    auto x_c = x.contiguous();

    int N = (int)x_c.size(0);
    int C = (int)x_c.size(1);
    int Di = (int)x_c.size(2);
    int Hi = (int)x_c.size(3);
    int Wi = (int)x_c.size(4);

    int Kd = (int)kernel_size[0];
    int Kh = (int)kernel_size[1];
    int Kw = (int)kernel_size[2];

    int Sd = stride.size() == 0 ? Kd : (int)stride[0];
    int Sh = stride.size() == 0 ? Kh : (int)stride[1];
    int Sw = stride.size() == 0 ? Kw : (int)stride[2];

    int Pd = padding.size() == 0 ? 0 : (int)padding[0];
    int Ph = padding.size() == 0 ? 0 : (int)padding[1];
    int Pw = padding.size() == 0 ? 0 : (int)padding[2];

    auto out_size = [&](int in, int k, int s, int p) -> int {
        float numer = (float)(in + 2 * p - k);
        if (ceil_mode) return (int)floorf((numer + s - 1) / s) + 1;
        return (int)floorf(numer / s) + 1;
    };

    int Do = out_size(Di, Kd, Sd, Pd);
    int Ho = out_size(Hi, Kh, Sh, Ph);
    int Wo = out_size(Wi, Kw, Sw, Pw);

    auto y = torch::empty({N, C, Do, Ho, Wo}, x_c.options());

    int div_override = divisor_override.has_value() ? (int)divisor_override.value() : -1;
    int gelu_mode = (approximate == "tanh") ? 1 : 0;

    int total = N * C * Do * Ho * Wo;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;

    avgpool3d_gelu_kernel<<<blocks, threads, 0, at::cuda::getDefaultCUDAStream()>>>(
        x_c.data_ptr<float>(),
        y.data_ptr<float>(),
        N, C, Di, Hi, Wi,
        Do, Ho, Wo,
        Kd, Kh, Kw,
        Sd, Sh, Sw,
        Pd, Ph, Pw,
        count_include_pad,
        div_override,
        gelu_mode
    );

    return y;
}
