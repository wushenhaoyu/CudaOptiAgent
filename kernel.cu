#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include <vector>

#define BLOCK_SIZE 256
#define TILE_SIZE 16

// Kernel for linear layer (GEMM + bias)
template<bool apply_relu>
__global__ void linear_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    const int batch_size,
    const int in_features,
    const int out_features) {
    
    // 2D block for tiled matrix multiplication
    __shared__ float tile_A[TILE_SIZE][TILE_SIZE];
    __shared__ float tile_B[TILE_SIZE][TILE_SIZE];
    
    const int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    const int col = blockIdx.x * TILE_SIZE + threadIdx.x;
    
    float sum = 0.0f;
    
    // Loop over tiles
    for (int t = 0; t < (in_features + TILE_SIZE - 1) / TILE_SIZE; ++t) {
        // Load tile from input (batch_size x in_features)
        const int input_col = t * TILE_SIZE + threadIdx.x;
        if (row < batch_size && input_col < in_features) {
            tile_A[threadIdx.y][threadIdx.x] = input[row * in_features + input_col];
        } else {
            tile_A[threadIdx.y][threadIdx.x] = 0.0f;
        }
        
        // Load tile from weight (out_features x in_features, but transposed)
        const int weight_row = col;
        const int weight_col = t * TILE_SIZE + threadIdx.y;
        if (weight_row < out_features && weight_col < in_features) {
            tile_B[threadIdx.y][threadIdx.x] = weight[weight_row * in_features + weight_col];
        } else {
            tile_B[threadIdx.y][threadIdx.x] = 0.0f;
        }
        
        __syncthreads();
        
        // Compute partial sum
        for (int k = 0; k < TILE_SIZE; ++k) {
            sum += tile_A[threadIdx.y][k] * tile_B[k][threadIdx.x];
        }
        
        __syncthreads();
    }
    
    // Write result with optional ReLU
    if (row < batch_size && col < out_features) {
        float result = sum + bias[col];
        if (apply_relu) {
            result = fmaxf(0.0f, result);
        }
        output[row * out_features + col] = result;
    }
}

// In-place ReLU kernel
__global__ void relu_kernel(
    float* tensor,
    const int num_elements) {
    
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_elements) {
        tensor[idx] = fmaxf(0.0f, tensor[idx]);
    }
}

// Special fused kernel for linear2 + relu2 (as specified in fusion plan)
__global__ void fused_linear2_relu2_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    const int batch_size,
    const int in_features,
    const int out_features) {
    
    __shared__ float tile_A[TILE_SIZE][TILE_SIZE];
    __shared__ float tile_B[TILE_SIZE][TILE_SIZE];
    
    const int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    const int col = blockIdx.x * TILE_SIZE + threadIdx.x;
    
    float sum = 0.0f;
    
    // Loop over tiles
    for (int t = 0; t < (in_features + TILE_SIZE - 1) / TILE_SIZE; ++t) {
        // Load tile from input
        const int input_col = t * TILE_SIZE + threadIdx.x;
        if (row < batch_size && input_col < in_features) {
            tile_A[threadIdx.y][threadIdx.x] = input[row * in_features + input_col];
        } else {
            tile_A[threadIdx.y][threadIdx.x] = 0.0f;
        }
        
        // Load tile from weight
        const int weight_row = col;
        const int weight_col = t * TILE_SIZE + threadIdx.y;
        if (weight_row < out_features && weight_col < in_features) {
            tile_B[threadIdx.y][threadIdx.x] = weight[weight_row * in_features + weight_col];
        } else {
            tile_B[threadIdx.y][threadIdx.x] = 0.0f;
        }
        
        __syncthreads();
        
        // Compute partial sum
        for (int k = 0; k < TILE_SIZE; ++k) {
            sum += tile_A[threadIdx.y][k] * tile_B[k][threadIdx.x];
        }
        
        __syncthreads();
    }
    
    // Write result with ReLU (fused as per fusion plan)
    if (row < batch_size && col < out_features) {
        float result = sum + bias[col];
        result = fmaxf(0.0f, result);  // ReLU activation
        output[row * out_features + col] = result;
    }
}

torch::Tensor MLP(
    torch::Tensor input,
    const std::vector<torch::Tensor>& weights,
    const std::vector<torch::Tensor>& biases) {
    
    AT_ASSERTM(input.is_cuda(), "input must be a CUDA tensor");
    AT_ASSERTM(input.dim() == 2, "input must be 2D");
    AT_ASSERTM(weights.size() == biases.size(), "Number of weights and biases must match");
    
    const int batch_size = input.size(0);
    const int num_layers = weights.size();
    
    // Validate all weights and biases are on CUDA
    for (int i = 0; i < num_layers; ++i) {
        AT_ASSERTM(weights[i].is_cuda(), "weights must be CUDA tensors");
        AT_ASSERTM(biases[i].is_cuda(), "biases must be CUDA tensors");
        AT_ASSERTM(weights[i].dim() == 2, "weights must be 2D");
        AT_ASSERTM(biases[i].dim() == 1, "biases must be 1D");
    }
    
    std::vector<torch::Tensor> activations;
    activations.push_back(input);
    
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
    
    // Process each layer according to fusion plan
    for (int layer_idx = 0; layer_idx < num_layers; ++layer_idx) {
        const int in_features = activations.back().size(1);
        const int out_features = weights[layer_idx].size(0);
        
        AT_ASSERTM(weights[layer_idx].size(1) == in_features, 
                  "Weight matrix dimensions mismatch");
        AT_ASSERTM(biases[layer_idx].size(0) == out_features,
                  "Bias vector dimensions mismatch");
        
        auto output = torch::empty({batch_size, out_features}, 
                                  torch::dtype(torch::kFloat32).device(torch::kCUDA));
        
        const dim3 block(TILE_SIZE, TILE_SIZE);
        const dim3 grid(
            (out_features + TILE_SIZE - 1) / TILE_SIZE,
            (batch_size + TILE_SIZE - 1) / TILE_SIZE
        );
        
        // According to fusion plan: layer 2 (index 1) uses fused linear+relu
        // Other layers use separate kernels
        if (layer_idx == 1) {  // linear2 + relu2 (fused as per fusion plan)
            fused_linear2_relu2_kernel<<<grid, block, 0, stream>>>(
                activations.back().data_ptr<float>(),
                weights[layer_idx].data_ptr<float>(),
                biases[layer_idx].data_ptr<float>(),
                output.data_ptr<float>(),
                batch_size,
                in_features,
                out_features
            );
        } else if (layer_idx == num_layers - 1) {  // Last layer: linear only
            linear_kernel<false><<<grid, block, 0, stream>>>(
                activations.back().data_ptr<float>(),
                weights[layer_idx].data_ptr<float>(),
                biases[layer_idx].data_ptr<float>(),
                output.data_ptr<float>(),
                batch_size,
                in_features,
                out_features
            );
        } else {  // Other layers: linear + separate relu
            linear_kernel<false><<<grid, block, 0, stream>>>(
                activations.back().data_ptr<float>(),
                weights[layer_idx].data_ptr<float>(),
                biases[layer_idx].data_ptr<float>(),
                output.data_ptr<float>(),
                batch_size,
                in_features,
                out_features
            );
            
            // Apply ReLU in-place (except for last layer)
            const int num_elements = batch_size * out_features;
            const int grid_size_relu = (num_elements + BLOCK_SIZE - 1) / BLOCK_SIZE;
            relu_kernel<<<grid_size_relu, BLOCK_SIZE, 0, stream>>>(
                output.data_ptr<float>(),
                num_elements
            );
        }
        
        activations.push_back(output);
    }
    
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    
    return activations.back();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("MLP", &MLP, "Multi-layer perceptron with specified fusion");
}