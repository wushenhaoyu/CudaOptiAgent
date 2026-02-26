#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include <vector>

#define TILE_SIZE 32
#define BLOCK_SIZE 256

// Kernel for linear layer
__global__ void linear_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int batch_size,
    int input_dim,
    int output_dim) {
    
    int batch_idx = blockIdx.y;
    int output_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (output_idx < output_dim && batch_idx < batch_size) {
        float sum = 0.0f;
        for (int i = 0; i < input_dim; i++) {
            float w = weight[output_idx * input_dim + i];
            float x = input[batch_idx * input_dim + i];
            sum += w * x;
        }
        output[batch_idx * output_dim + output_idx] = sum + bias[output_idx];
    }
}
__global__ void fused_linear_relu_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int batch_size,
    int input_dim,
    int output_dim) {
    
    int batch_idx = blockIdx.y;
    int output_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (output_idx < output_dim && batch_idx < batch_size) {
        float sum = 0.0f;
        #pragma unroll 4
        for (int i = 0; i < input_dim; i++) {
            float w = weight[output_idx * input_dim + i];
            float x = input[batch_idx * input_dim + i];
            sum += w * x;
        }
        sum += bias[output_idx];
        // Apply ReLU
        output[batch_idx * output_dim + output_idx] = fmaxf(0.0f, sum);
    }
}


// Main MLP function
torch::Tensor MLP(
    torch::Tensor input,
    torch::Tensor weight1,
    torch::Tensor bias1,
    torch::Tensor weight2,
    torch::Tensor bias2,
    torch::Tensor weight3,
    torch::Tensor bias3) {
    
    int batch_size = input.size(0);
    int input_size = input.size(1);
    
    int hidden1_size = weight1.size(0);
    int hidden2_size = weight2.size(0);
    int output_size = weight3.size(0);
    
    auto hidden1 = torch::empty({batch_size, hidden1_size}, 
                                torch::dtype(torch::kFloat32).device(torch::kCUDA));
    auto hidden2 = torch::empty({batch_size, hidden2_size}, 
                                torch::dtype(torch::kFloat32).device(torch::kCUDA));
    auto output = torch::empty({batch_size, output_size}, 
                               torch::dtype(torch::kFloat32).device(torch::kCUDA));
    
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();

    dim3 block1(BLOCK_SIZE, 1, 1);
    dim3 grid1((hidden1_size + BLOCK_SIZE - 1) / BLOCK_SIZE, batch_size, 1);
    fused_linear_relu_kernel<<<grid1, block1, 0, stream>>>(
        input.data_ptr<float>(),
        weight1.data_ptr<float>(),
        bias1.data_ptr<float>(),
        hidden1.data_ptr<float>(),
        batch_size,
        input_size,
        hidden1_size
    );
    // Layer 2: fused linear2 + relu2
    dim3 block2(BLOCK_SIZE, 1, 1);
    dim3 grid2((hidden2_size + BLOCK_SIZE - 1) / BLOCK_SIZE, batch_size, 1);
    fused_linear_relu_kernel<<<grid2, block2, 0, stream>>>(
        hidden1.data_ptr<float>(),
        weight2.data_ptr<float>(),
        bias2.data_ptr<float>(),
        hidden2.data_ptr<float>(),
        batch_size,
        hidden1_size,
        hidden2_size
    );
    
    dim3 block3(BLOCK_SIZE, 1, 1);
    dim3 grid3((output_size + BLOCK_SIZE - 1) / BLOCK_SIZE, batch_size, 1);
    linear_kernel<<<grid3, block3, 0, stream>>>(
        hidden2.data_ptr<float>(),
        weight3.data_ptr<float>(),
        bias3.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        hidden2_size,
        output_size
    );
    
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("MLP", &MLP, "MLP forward pass with fused linear2+relu2");
}