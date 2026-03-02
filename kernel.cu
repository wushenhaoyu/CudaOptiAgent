#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include <vector>
#include <math.h>

#define BLOCK_SIZE 256

// -----------------------------------------------------------
// LayerNorm Kernel (row-wise)
// -----------------------------------------------------------
__global__ void layernorm_kernel(
    const float* __restrict__ input,
    const float* __restrict__ gamma,
    const float* __restrict__ beta,
    float* __restrict__ output,
    int N, // number of rows (B*T)
    int C  // number of features (n_embd)
) {
    int row = blockIdx.x;
    if (row >= N) return;

    const float* in_row = input + row * C;
    float* out_row = output + row * C;

    // compute mean
    float mean = 0.0f;
    for (int i = threadIdx.x; i < C; i += blockDim.x)
        mean += in_row[i];
    
    __shared__ float smem_mean;
    atomicAdd(&smem_mean, mean);
    __syncthreads();

    mean = smem_mean / C;
    __syncthreads();

    // compute variance
    float var = 0.0f;
    for (int i = threadIdx.x; i < C; i += blockDim.x) {
        float diff = in_row[i] - mean;
        var += diff * diff;
    }
    __shared__ float smem_var;
    atomicAdd(&smem_var, var);
    __syncthreads();

    var = smem_var / C;
    float inv_std = rsqrtf(var + 1e-5f);

    for (int i = threadIdx.x; i < C; i += blockDim.x) {
        out_row[i] = (in_row[i] - mean) * inv_std * gamma[i] + beta[i];
    }
}

// -----------------------------------------------------------
// Linear + GELU Kernel
// -----------------------------------------------------------
__device__ float gelu(float x) {
    const float k0 = sqrtf(2.0f / M_PI);
    return 0.5f * x * (1.0f + tanhf(k0 * (x + 0.044715f * x * x * x)));
}

__global__ void linear_gelu_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int N, // rows
    int C_in,
    int C_out
) {
    int row = blockIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= N || col >= C_out) return;

    float sum = 0.0f;
    for (int i = 0; i < C_in; i++)
        sum += input[row * C_in + i] * weight[col * C_in + i];
    sum += bias[col];
    output[row * C_out + col] = gelu(sum);
}

// -----------------------------------------------------------
// Linear Kernel (no activation)
// -----------------------------------------------------------
__global__ void linear_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int N, // rows
    int C_in,
    int C_out
) {
    int row = blockIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= N || col >= C_out) return;

    float sum = 0.0f;
    for (int i = 0; i < C_in; i++)
        sum += input[row * C_in + i] * weight[col * C_in + i];
    sum += bias[col];
    output[row * C_out + col] = sum;
}

// -----------------------------------------------------------
// Elementwise Add Kernel
// -----------------------------------------------------------
__global__ void add_residual_kernel(
    const float* __restrict__ residual,
    const float* __restrict__ out,
    float* __restrict__ output,
    int N
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;
    output[idx] = residual[idx] + out[idx];
}

// -----------------------------------------------------------
// Main MiniGPTBlock Forward
// -----------------------------------------------------------
torch::Tensor MiniGPTBlock(
    torch::Tensor x,                 // [B, T, C]
    torch::Tensor ln1_weight,
    torch::Tensor ln1_bias,
    torch::Tensor c_attn_weight,
    torch::Tensor c_attn_bias,
    torch::Tensor c_proj_weight,
    torch::Tensor c_proj_bias,
    torch::Tensor attn_bias,
    torch::Tensor ln2_weight,
    torch::Tensor ln2_bias,
    torch::Tensor mlp_c_fc_weight,
    torch::Tensor mlp_c_fc_bias,
    torch::Tensor mlp_c_proj_weight,
    torch::Tensor mlp_c_proj_bias,
    int n_embd,
    int n_head
) {
    auto x_contig = x.contiguous();
    int B = x.size(0);
    int T = x.size(1);
    int C = x.size(2);
    int N = B * T;

    auto options = torch::TensorOptions().dtype(x.dtype()).device(x.device());
    auto ln1_out = torch::empty({N, C}, options);
    auto ln2_out = torch::empty({N, C}, options);

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();

    // LayerNorm1
    layernorm_kernel<<<N, BLOCK_SIZE, 0, stream>>>(
        x_contig.data_ptr<float>(),
        ln1_weight.data_ptr<float>(),
        ln1_bias.data_ptr<float>(),
        ln1_out.data_ptr<float>(),
        N,
        C
    );

    // MLP fc + GELU
    int mlp_hidden = 4 * C;
    auto mlp_out = torch::empty({N, C}, options);
    dim3 block(BLOCK_SIZE);
    dim3 grid((mlp_hidden + BLOCK_SIZE - 1) / BLOCK_SIZE, N);
    linear_gelu_kernel<<<grid, block, 0, stream>>>(
        ln2_out.data_ptr<float>(),
        mlp_c_fc_weight.data_ptr<float>(),
        mlp_c_fc_bias.data_ptr<float>(),
        mlp_out.data_ptr<float>(),
        N,
        C,
        mlp_hidden
    );

    // Linear projection
    auto proj_out = torch::empty({N, C}, options);
    dim3 grid_proj((C + BLOCK_SIZE - 1) / BLOCK_SIZE, N);
    linear_kernel<<<grid_proj, block, 0, stream>>>(
        mlp_out.data_ptr<float>(),
        mlp_c_proj_weight.data_ptr<float>(),
        mlp_c_proj_bias.data_ptr<float>(),
        proj_out.data_ptr<float>(),
        N,
        mlp_hidden,
        C
    );

    // Residual add
    auto output = torch::empty_like(x);
    add_residual_kernel<<<(N*C + BLOCK_SIZE - 1)/BLOCK_SIZE, BLOCK_SIZE, 0, stream>>>(
        x_contig.data_ptr<float>(),
        proj_out.data_ptr<float>(),
        output.data_ptr<float>(),
        N*C
    );

    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("MiniGPTBlock", &MiniGPTBlock, "MiniGPTBlock forward pass");
}