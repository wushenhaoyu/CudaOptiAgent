#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <cmath>

namespace {

__global__ void conv_transpose3d_add_kernel(
    const float* __restrict__ x,          // [N, Cin, Di, Hi, Wi]
    const float* __restrict__ w,          // [Cin, Cout/groups, Kd, Kh, Kw]
    const float* __restrict__ b,          // [Cout] or nullptr
    float* __restrict__ y,                // [N, Cout, Do, Ho, Wo]
    int N, int Cin, int Di, int Hi, int Wi,
    int Cout, int Do, int Ho, int Wo,
    int Kd, int Kh, int Kw,
    int Sd, int Sh, int Sw,
    int Pd, int Ph, int Pw,
    int Od, int Oh, int Ow,
    int G, int Dd, int Dh, int Dw,
    float sum_weight
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * Cout * Do * Ho * Wo;
    if (idx >= total) return;

    int tmp = idx;
    int ow = tmp % Wo; tmp /= Wo;
    int oh = tmp % Ho; tmp /= Ho;
    int od = tmp % Do; tmp /= Do;
    int oc = tmp % Cout; tmp /= Cout;
    int n  = tmp;

    int cout_per_group = Cout / G;
    int cin_per_group = Cin / G;
    int g = oc / cout_per_group;
    int ocg = oc - g * cout_per_group;

    float acc = 0.0f;

    for (int icg = 0; icg < cin_per_group; ++icg) {
        int ic = g * cin_per_group + icg;
        for (int kd = 0; kd < Kd; ++kd) {
            int t_d = od + Pd - kd * Dd;
            if (t_d < 0 || (t_d % Sd) != 0) continue;
            int id = t_d / Sd;
            if (id < 0 || id >= Di) continue;

            for (int kh = 0; kh < Kh; ++kh) {
                int t_h = oh + Ph - kh * Dh;
                if (t_h < 0 || (t_h % Sh) != 0) continue;
                int ih = t_h / Sh;
                if (ih < 0 || ih >= Hi) continue;

                for (int kw = 0; kw < Kw; ++kw) {
                    int t_w = ow + Pw - kw * Dw;
                    if (t_w < 0 || (t_w % Sw) != 0) continue;
                    int iw = t_w / Sw;
                    if (iw < 0 || iw >= Wi) continue;

                    int x_idx = (((n * Cin + ic) * Di + id) * Hi + ih) * Wi + iw;
                    int w_idx = ((((ic * cout_per_group + ocg) * Kd + kd) * Kh + kh) * Kw + kw);
                    acc += x[x_idx] * w[w_idx];
                }
            }
        }
    }

    if (b) acc += b[oc];
    acc += sum_weight;
    y[idx] = acc;
}

} // namespace

torch::Tensor conv_transpose3d_add(
    torch::Tensor x,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor sum_weight,
    std::vector<int64_t> stride,
    std::vector<int64_t> padding,
    std::vector<int64_t> output_padding,
    int64_t groups,
    std::vector<int64_t> dilation
) {
    TORCH_CHECK(x.is_cuda(), "x must be CUDA");
    TORCH_CHECK(weight.is_cuda(), "weight must be CUDA");
    TORCH_CHECK(!bias.defined() || bias.is_cuda(), "bias must be CUDA when defined");
    TORCH_CHECK(sum_weight.is_cuda(), "sum_weight must be CUDA");
    TORCH_CHECK(x.scalar_type() == torch::kFloat32, "x must be float32");
    TORCH_CHECK(weight.scalar_type() == torch::kFloat32, "weight must be float32");
    TORCH_CHECK(!bias.defined() || bias.scalar_type() == torch::kFloat32, "bias must be float32");
    TORCH_CHECK(sum_weight.scalar_type() == torch::kFloat32, "sum_weight must be float32");
    TORCH_CHECK(stride.size() == 3 && padding.size() == 3 && output_padding.size() == 3 && dilation.size() == 3, "3D params required");

    auto x_c = x.contiguous();
    auto w_c = weight.contiguous();
    auto b_c = bias.defined() ? bias.contiguous() : torch::Tensor();
    auto sw_c = sum_weight.contiguous();

    int N = (int)x_c.size(0);
    int Cin = (int)x_c.size(1);
    int Di = (int)x_c.size(2);
    int Hi = (int)x_c.size(3);
    int Wi = (int)x_c.size(4);

    int Kd = (int)w_c.size(2);
    int Kh = (int)w_c.size(3);
    int Kw = (int)w_c.size(4);

    int Sd = (int)stride[0], Sh = (int)stride[1], Sw = (int)stride[2];
    int Pd = (int)padding[0], Ph = (int)padding[1], Pw = (int)padding[2];
    int Od = (int)output_padding[0], Oh = (int)output_padding[1], Ow = (int)output_padding[2];
    int Dd = (int)dilation[0], Dh = (int)dilation[1], Dw = (int)dilation[2];
    int G = (int)groups;

    int Cout = (int)w_c.size(1) * G;
    int Do = (Di - 1) * Sd - 2 * Pd + Dd * (Kd - 1) + Od + 1;
    int Ho = (Hi - 1) * Sh - 2 * Ph + Dh * (Kh - 1) + Oh + 1;
    int Wo = (Wi - 1) * Sw - 2 * Pw + Dw * (Kw - 1) + Ow + 1;

    auto y = torch::empty({N, Cout, Do, Ho, Wo}, x_c.options());

    float sw = sw_c.item<float>();
    const float* bptr = b_c.defined() ? b_c.data_ptr<float>() : nullptr;

    int total = N * Cout * Do * Ho * Wo;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;

    conv_transpose3d_add_kernel<<<blocks, threads, 0, at::cuda::getDefaultCUDAStream()>>>(
        x_c.data_ptr<float>(),
        w_c.data_ptr<float>(),
        bptr,
        y.data_ptr<float>(),
        N, Cin, Di, Hi, Wi,
        Cout, Do, Ho, Wo,
        Kd, Kh, Kw,
        Sd, Sh, Sw,
        Pd, Ph, Pw,
        Od, Oh, Ow,
        G, Dd, Dh, Dw,
        sw
    );

    return y;
}
