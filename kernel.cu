#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include <vector>

#define BLOCK_SIZE 256
#define TILE_SIZE 32

__global__ void LinearReLUKernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int batch_size,
    int in_features,
    int out_features,
    bool apply_relu) {
    
    int batch_idx = blockIdx.y;
    int out_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (batch_idx >= batch_size || out_idx >= out_features) return;
    
    extern __shared__ float shared[];
    float* shared_input = shared;
    float* shared_weight = shared + TILE_SIZE;
    
    float sum = 0.0f;
    
    for (int tile = 0; tile < (in_features + TILE_SIZE - 1) / TILE_SIZE; ++tile) {
        int k_start = tile * TILE_SIZE;
        
        // Each thread loads one element if threadIdx.x < TILE_SIZE
        if (threadIdx.x < TILE_SIZE) {
            int k_idx = k_start + threadIdx.x;
            if (k_idx < in_features) {
                shared_input[threadIdx.x] = input[batch_idx * in_features + k_idx];
                shared_weight[threadIdx.x] = weight[out_idx * in_features + k_idx];
            } else {
                shared_input[threadIdx.x] = 0.0f;
                shared_weight[threadIdx.x] = 0.0f;
            }
        }
        
        __syncthreads();
        
        for (int k = 0; k < TILE_SIZE; ++k) {
            sum += shared_input[k] * shared_weight[k];
        }
        
        __syncthreads();
    }
    
    sum += bias[out_idx];
    
    if (apply_relu) {
        sum = fmaxf(0.0f, sum);
    }
    
    output[batch_idx * out_features + out_idx] = sum;
}

void LaunchLinearReLU(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    int batch_size,
    int in_features,
    int out_features,
    bool apply_relu) {
    
    dim3 block(BLOCK_SIZE);
    dim3 grid((out_features + BLOCK_SIZE - 1) / BLOCK_SIZE, batch_size);
    size_t shared_mem = 2 * TILE_SIZE * sizeof(float);
    
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
    
    LinearReLUKernel<<<grid, block, shared_mem, stream>>>(
        input, weight, bias, output,
        batch_size, in_features, out_features, apply_relu
    );
    
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

torch::Tensor MLP(
    torch::Tensor x,
    std::vector<torch::Tensor> weights,
    std::vector<torch::Tensor> biases) {
    
    AT_ASSERTM(x.is_cuda(), "input must be CUDA tensor");
    AT_ASSERTM(weights.size() == biases.size(), "weights and biases must have same length");
    
    for (size_t i = 0; i < weights.size(); ++i) {
        AT_ASSERTM(weights[i].is_cuda(), "weight must be CUDA tensor");
        AT_ASSERTM(biases[i].is_cuda(), "bias must be CUDA tensor");
    }
    
    int batch_size = x.size(0);
    int in_features = x.size(1);
    
    torch::Tensor current_input = x;
    torch::Tensor current_output;
    
    int num_layers = weights.size();
    
    for (int layer = 0; layer < num_layers; ++layer) {
        int out_features = weights[layer].size(0);
        bool is_last_layer = (layer == num_layers - 1);
        
        current_output = torch::empty({batch_size, out_features},
                                     torch::dtype(torch::kFloat32).device(torch::kCUDA));
        
        LaunchLinearReLU(
            current_input.data_ptr<float>(),
            weights[layer].data_ptr<float>(),
            biases[layer].data_ptr<float>(),
            current_output.data_ptr<float>(),
            batch_size,
            in_features,
            out_features,
            !is_last_layer
        );
        
        current_input = current_output;
        in_features = out_features;
    }
    
    return current_output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("MLP", &MLP, "Multi-Layer Perceptron with ReLU");
}