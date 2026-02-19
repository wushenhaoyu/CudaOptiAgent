#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <vector>
#include <cassert>

// ============================================
// CUDA Kernels for Conv2d + Mish + Mish
// ============================================

// Mish activation function: x * tanh(ln(1 + exp(x)))
// Optimized version to avoid numerical issues
__device__ __forceinline__ float mish_func(float x) {
    float e = expf(x);
    float softplus = log1pf(e);  // ln(1 + e)
    return x * tanhf(softplus);
}

// Mish activation kernel (in-place)
__global__ void mish_kernel(
    float* data,
    int num_elements
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_elements) {
        data[idx] = mish_func(data[idx]);
    }
}

// Main kernel: Conv2d + 2x Mish (fused)
// Implements direct convolution with kernel_size=3
// Input:  [N, C, H, W]
// Weight: [K, C, 3, 3]
// Output: [N, K, H-2, W-2] (no padding, stride=1)
template <int TILE_SIZE = 32>
__global__ void conv2d_mish_mish_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int N, int C, int H, int W, int K
) {
    // Output dimensions
    const int OH = H - 2;
    const int OW = W - 2;
    
    // Tile dimensions
    const int TILE_H = TILE_SIZE;
    const int TILE_W = TILE_SIZE;
    
    // Shared memory for input tile
    __shared__ float input_tile[TILE_H + 2][TILE_W + 2];
    
    // Thread position in output space
    const int ow = blockIdx.x * TILE_W + threadIdx.x;
    const int oh = blockIdx.y * TILE_H + threadIdx.y;
    const int n = blockIdx.z;
    
    // Early exit if out of bounds
    if (ow >= OW || oh >= OH || n >= N) return;
    
    // Accumulator for each output channel
    float acc = 0.0f;
    
    // Load input tile to shared memory
    // Each thread loads one element
    for (int c = 0; c < C; c++) {
        // Load input tile (including halo for convolution)
        if (threadIdx.x < TILE_W + 2 && threadIdx.y < TILE_H + 2) {
            int h_idx = oh + threadIdx.y;
            int w_idx = ow + threadIdx.x;
            
            // Check bounds
            if (h_idx < H && w_idx < W) {
                int input_idx = ((n * C + c) * H + h_idx) * W + w_idx;
                input_tile[threadIdx.y][threadIdx.x] = input[input_idx];
            } else {
                input_tile[threadIdx.y][threadIdx.x] = 0.0f;
            }
        }
        
        __syncthreads();
        
        // Convolution for all output channels
        for (int k = 0; k < K; k++) {
            // Reset accumulator for this output channel
            if (c == 0) {
                acc = bias ? bias[k] : 0.0f;
            }
            
            // Perform 3x3 convolution
            #pragma unroll
            for (int kh = 0; kh < 3; kh++) {
                #pragma unroll
                for (int kw = 0; kw < 3; kw++) {
                    int weight_idx = ((k * C + c) * 3 + kh) * 3 + kw;
                    float input_val = input_tile[threadIdx.y + kh][threadIdx.x + kw];
                    float weight_val = weight[weight_idx];
                    acc += input_val * weight_val;
                }
            }
            
            // After processing all input channels, apply Mish twice
            if (c == C - 1) {
                // First Mish
                float mish1 = mish_func(acc);
                // Second Mish
                float mish2 = mish_func(mish1);
                
                // Write output
                int output_idx = ((n * K + k) * OH + oh) * OW + ow;
                output[output_idx] = mish2;
            }
        }
        
        __syncthreads();
    }
}

// Wrapper function for the fused kernel
torch::Tensor conv2d_mish_mish_forward(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias
) {
    // Check input dimensions
    TORCH_CHECK(input.dim() == 4, "Input must be 4D (N, C, H, W)");
    TORCH_CHECK(weight.dim() == 4, "Weight must be 4D (K, C, KH, KW)");
    
    // Get dimensions
    const int N = input.size(0);
    const int C = input.size(1);
    const int H = input.size(2);
    const int W = input.size(3);
    const int K = weight.size(0);
    const int kH = weight.size(2);
    const int kW = weight.size(3);
    
    // Verify kernel size is 3x3
    TORCH_CHECK(kH == 3 && kW == 3, "Kernel size must be 3x3");
    TORCH_CHECK(C == weight.size(1), "Input channels must match weight channels");
    
    // Create output tensor
    const int OH = H - 2;  // no padding, stride=1
    const int OW = W - 2;
    auto output = torch::zeros({N, K, OH, OW}, input.options());
    
    // Setup CUDA grid and blocks
    const int TILE_SIZE = 16;
    dim3 block_dim(TILE_SIZE, TILE_SIZE, 1);
    dim3 grid_dim(
        (OW + TILE_SIZE - 1) / TILE_SIZE,
        (OH + TILE_SIZE - 1) / TILE_SIZE,
        N
    );
    
    // Get pointers
    float* input_ptr = input.data_ptr<float>();
    float* weight_ptr = weight.data_ptr<float>();
    float* bias_ptr = bias.defined() ? bias.data_ptr<float>() : nullptr;
    float* output_ptr = output.data_ptr<float>();
    
    // Launch kernel
    conv2d_mish_mish_kernel<TILE_SIZE><<<grid_dim, block_dim>>>(
        input_ptr,
        weight_ptr,
        bias_ptr,
        output_ptr,
        N, C, H, W, K
    );
    
    return output;
}

// ============================================
// PyBind11 Module Definition
// ============================================

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("Conv2d_Mish_Mish", &conv2d_mish_mish_forward,
          "Fused Conv2d + Mish + Mish operation");
}