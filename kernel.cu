#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>

#define BLOCK_SIZE 256

// Kernel for fused linear + ReLU layers
// Computes: output = ReLU(input @ weight^T + bias)
__global__ void fused_linear_relu_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int batch_size,
    int input_dim,
    int output_dim) {

    int batch_idx = blockIdx.y;
    int out_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (batch_idx < batch_size && out_idx < output_dim) {
        float sum = 0.0f;
        for (int i = 0; i < input_dim; i++) {
            float x = input[batch_idx * input_dim + i];
            float w = weight[out_idx * input_dim + i];
            sum += x * w;
        }
        sum += bias[out_idx];
        output[batch_idx * output_dim + out_idx] = fmaxf(sum, 0.0f);
    }
}

// Kernel for final linear layer (no ReLU)
// Computes: output = input @ weight^T + bias
__global__ void linear_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int batch_size,
    int input_dim,
    int output_dim) {

    int batch_idx = blockIdx.y;
    int out_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (batch_idx < batch_size && out_idx < output_dim) {
        float sum = 0.0f;
        for (int i = 0; i < input_dim; i++) {
            float x = input[batch_idx * input_dim + i];
            float w = weight[out_idx * input_dim + i];
            sum += x * w;
        }
        sum += bias[out_idx];
        output[batch_idx * output_dim + out_idx] = sum;
    }
}

torch::Tensor DeepNarrowMLP(
    torch::Tensor input,
    torch::Tensor weight0, torch::Tensor bias0,
    torch::Tensor weight1, torch::Tensor bias1,
    torch::Tensor weight2, torch::Tensor bias2,
    torch::Tensor weight3, torch::Tensor bias3,
    torch::Tensor weight4, torch::Tensor bias4,
    torch::Tensor weight5, torch::Tensor bias5,
    torch::Tensor weight6, torch::Tensor bias6,
    torch::Tensor weight7, torch::Tensor bias7,
    torch::Tensor weight8, torch::Tensor bias8,
    torch::Tensor weight9, torch::Tensor bias9,
    torch::Tensor weight10, torch::Tensor bias10,
    torch::Tensor weight11, torch::Tensor bias11,
    torch::Tensor weight12, torch::Tensor bias12,
    torch::Tensor weight13, torch::Tensor bias13,
    torch::Tensor weight14, torch::Tensor bias14,
    torch::Tensor weight15, torch::Tensor bias15,
    torch::Tensor weight16, torch::Tensor bias16) {

    // Ensure all weights and biases are contiguous and on CUDA device
    weight0 = weight0.contiguous();
    bias0 = bias0.contiguous();
    weight1 = weight1.contiguous();
    bias1 = bias1.contiguous();
    weight2 = weight2.contiguous();
    bias2 = bias2.contiguous();
    weight3 = weight3.contiguous();
    bias3 = bias3.contiguous();
    weight4 = weight4.contiguous();
    bias4 = bias4.contiguous();
    weight5 = weight5.contiguous();
    bias5 = bias5.contiguous();
    weight6 = weight6.contiguous();
    bias6 = bias6.contiguous();
    weight7 = weight7.contiguous();
    bias7 = bias7.contiguous();
    weight8 = weight8.contiguous();
    bias8 = bias8.contiguous();
    weight9 = weight9.contiguous();
    bias9 = bias9.contiguous();
    weight10 = weight10.contiguous();
    bias10 = bias10.contiguous();
    weight11 = weight11.contiguous();
    bias11 = bias11.contiguous();
    weight12 = weight12.contiguous();
    bias12 = bias12.contiguous();
    weight13 = weight13.contiguous();
    bias13 = bias13.contiguous();
    weight14 = weight14.contiguous();
    bias14 = bias14.contiguous();
    weight15 = weight15.contiguous();
    bias15 = bias15.contiguous();
    weight16 = weight16.contiguous();
    bias16 = bias16.contiguous();

    int batch_size = input.size(0);

    int input_size = input.size(1);
    int hidden_size = weight0.size(0);
    int output_size = weight16.size(0);

    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);

    torch::Tensor h0 = torch::empty({batch_size, hidden_size}, options);
    torch::Tensor h1 = torch::empty({batch_size, hidden_size}, options);
    torch::Tensor h2 = torch::empty({batch_size, hidden_size}, options);
    torch::Tensor h3 = torch::empty({batch_size, hidden_size}, options);
    torch::Tensor h4 = torch::empty({batch_size, hidden_size}, options);
    torch::Tensor h5 = torch::empty({batch_size, hidden_size}, options);
    torch::Tensor h6 = torch::empty({batch_size, hidden_size}, options);
    torch::Tensor h7 = torch::empty({batch_size, hidden_size}, options);
    torch::Tensor h8 = torch::empty({batch_size, hidden_size}, options);
    torch::Tensor h9 = torch::empty({batch_size, hidden_size}, options);
    torch::Tensor h10 = torch::empty({batch_size, hidden_size}, options);
    torch::Tensor h11 = torch::empty({batch_size, hidden_size}, options);
    torch::Tensor h12 = torch::empty({batch_size, hidden_size}, options);
    torch::Tensor h13 = torch::empty({batch_size, hidden_size}, options);
    torch::Tensor h14 = torch::empty({batch_size, hidden_size}, options);
    torch::Tensor h15 = torch::empty({batch_size, hidden_size}, options);

    torch::Tensor output = torch::empty({batch_size, output_size}, options);

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();

    dim3 block(BLOCK_SIZE, 1, 1);

    auto launch_fused_linear_relu = [&](const torch::Tensor& in, const torch::Tensor& w, const torch::Tensor& b, torch::Tensor& out) {
        int out_dim = w.size(0);
        int in_dim = w.size(1);
        dim3 grid((out_dim + BLOCK_SIZE - 1) / BLOCK_SIZE, batch_size, 1);
        fused_linear_relu_kernel<<<grid, block, 0, stream>>>(
            in.data_ptr<float>(),
            w.data_ptr<float>(),
            b.data_ptr<float>(),
            out.data_ptr<float>(),
            batch_size,
            in_dim,
            out_dim);
    };

    auto launch_linear = [&](const torch::Tensor& in, const torch::Tensor& w, const torch::Tensor& b, torch::Tensor& out) {
        int out_dim = w.size(0);
        int in_dim = w.size(1);
        dim3 grid((out_dim + BLOCK_SIZE - 1) / BLOCK_SIZE, batch_size, 1);
        linear_kernel<<<grid, block, 0, stream>>>(
            in.data_ptr<float>(),
            w.data_ptr<float>(),
            b.data_ptr<float>(),
            out.data_ptr<float>(),
            batch_size,
            in_dim,
            out_dim);
    };

    launch_fused_linear_relu(input, weight0, bias0, h0);
    launch_fused_linear_relu(h0, weight1, bias1, h1);
    launch_fused_linear_relu(h1, weight2, bias2, h2);
    launch_fused_linear_relu(h2, weight3, bias3, h3);
    launch_fused_linear_relu(h3, weight4, bias4, h4);
    launch_fused_linear_relu(h4, weight5, bias5, h5);
    launch_fused_linear_relu(h5, weight6, bias6, h6);
    launch_fused_linear_relu(h6, weight7, bias7, h7);
    launch_fused_linear_relu(h7, weight8, bias8, h8);
    launch_fused_linear_relu(h8, weight9, bias9, h9);
    launch_fused_linear_relu(h9, weight10, bias10, h10);
    launch_fused_linear_relu(h10, weight11, bias11, h11);
    launch_fused_linear_relu(h11, weight12, bias12, h12);
    launch_fused_linear_relu(h12, weight13, bias13, h13);
    launch_fused_linear_relu(h13, weight14, bias14, h14);
    launch_fused_linear_relu(h14, weight15, bias15, h15);
    launch_linear(h15, weight16, bias16, output);

    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("DeepNarrowMLP", &DeepNarrowMLP, "Deep Narrow MLP forward with fused linear+ReLU layers");
}