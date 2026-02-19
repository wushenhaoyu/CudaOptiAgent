#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>

#define TILE_H 16
#define TILE_W 16
#define BLOCK_CHANNELS 8

__global__ void Conv2D_ReLU_BiasAdd_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ conv_bias,
    const float* __restrict__ bias,
    float* __restrict__ output,
    const int batch_size,
    const int in_channels,
    const int out_channels,
    const int height,
    const int width,
    const int kernel_size,
    const int output_height,
    const int output_width) {
    
    // Calculate input tile dimensions with halo for convolution
    const int TILE_H_IN = TILE_H + kernel_size - 1;
    const int TILE_W_IN = TILE_W + kernel_size - 1;
    
    // Dynamic shared memory for input tile: (TILE_H_IN * TILE_W_IN * BLOCK_CHANNELS)
    extern __shared__ float shared_mem[];
    float* input_tile = shared_mem;
    
    const int block_channel = blockIdx.z % ((out_channels + BLOCK_CHANNELS - 1) / BLOCK_CHANNELS);
    const int batch = blockIdx.z / ((out_channels + BLOCK_CHANNELS - 1) / BLOCK_CHANNELS);
    
    const int h_tile_out = blockIdx.y * TILE_H;
    const int w_tile_out = blockIdx.x * TILE_W;
    const int h_tile_in = h_tile_out;  // No padding, so input tile starts at same position
    const int w_tile_in = w_tile_out;
    
    // Each thread loads multiple elements from input to shared memory
    const int elements_per_thread = (TILE_H_IN * TILE_W_IN + blockDim.x * blockDim.y - 1) / 
                                    (blockDim.x * blockDim.y);
    
    // Accumulators for each output channel in this block
    float acc[BLOCK_CHANNELS];
    for (int c = 0; c < BLOCK_CHANNELS; ++c) {
        acc[c] = 0.0f;
    }
    
    // Loop over input channel blocks
    for (int in_ch_block = 0; in_ch_block < (in_channels + BLOCK_CHANNELS - 1) / BLOCK_CHANNELS; ++in_ch_block) {
        
        // Load input tile into shared memory
        const int in_ch_base = in_ch_block * BLOCK_CHANNELS;
        
        // Each thread loads multiple elements for current input channel block
        for (int elem = 0; elem < elements_per_thread; ++elem) {
            const int linear_idx = threadIdx.y * blockDim.x + threadIdx.x + elem * blockDim.x * blockDim.y;
            
            if (linear_idx < TILE_H_IN * TILE_W_IN) {
                const int h_in_tile = linear_idx / TILE_W_IN;
                const int w_in_tile = linear_idx % TILE_W_IN;
                const int h_global = h_tile_in + h_in_tile;
                const int w_global = w_tile_in + w_in_tile;
                
                // Load BLOCK_CHANNELS channels for this spatial position
                for (int ch_offset = 0; ch_offset < BLOCK_CHANNELS; ++ch_offset) {
                    const int in_ch = in_ch_base + ch_offset;
                    float value = 0.0f;
                    
                    if (in_ch < in_channels && h_global < height && w_global < width) {
                        const int input_idx = ((batch * in_channels + in_ch) * height + h_global) * width + w_global;
                        value = input[input_idx];
                    }
                    
                    // Store in shared memory with proper indexing
                    const int shared_idx = (h_in_tile * TILE_W_IN + w_in_tile) * BLOCK_CHANNELS + ch_offset;
                    input_tile[shared_idx] = value;
                }
            }
        }
        
        __syncthreads();
        
        // Only threads within the output tile compute
        const int h_out = h_tile_out + threadIdx.y;
        const int w_out = w_tile_out + threadIdx.x;
        
        if (h_out < output_height && w_out < output_width) {
            // For each output channel in this block
            for (int out_ch_offset = 0; out_ch_offset < BLOCK_CHANNELS; ++out_ch_offset) {
                const int out_ch = block_channel * BLOCK_CHANNELS + out_ch_offset;
                
                if (out_ch < out_channels) {
                    float sum = 0.0f;
                    
                    // For each input channel in current block
                    for (int ch_offset = 0; ch_offset < BLOCK_CHANNELS; ++ch_offset) {
                        const int in_ch = in_ch_base + ch_offset;
                        
                        if (in_ch < in_channels) {
                            // Convolution over kernel window
                            for (int kh = 0; kh < kernel_size; ++kh) {
                                for (int kw = 0; kw < kernel_size; ++kw) {
                                    // Access shared memory with proper bounds checking
                                    const int h_in_tile = threadIdx.y + kh;
                                    const int w_in_tile = threadIdx.x + kw;
                                    
                                    if (h_in_tile < TILE_H_IN && w_in_tile < TILE_W_IN) {
                                        const int shared_idx = (h_in_tile * TILE_W_IN + w_in_tile) * BLOCK_CHANNELS + ch_offset;
                                        const float input_val = input_tile[shared_idx];
                                        
                                        const int weight_idx = (((out_ch * in_channels + in_ch) * kernel_size + kh) * kernel_size + kw);
                                        const float weight_val = weight[weight_idx];
                                        
                                        sum += input_val * weight_val;
                                    }
                                }
                            }
                        }
                    }
                    
                    // Accumulate across input channel blocks
                    if (in_ch_block == 0) {
                        acc[out_ch_offset] = sum;
                    } else {
                        acc[out_ch_offset] += sum;
                    }
                }
            }
        }
        
        __syncthreads();
    }
    
    // Write results with ReLU and bias addition
    const int h_out = h_tile_out + threadIdx.y;
    const int w_out = w_tile_out + threadIdx.x;
    
    if (h_out < output_height && w_out < output_width) {
        for (int out_ch_offset = 0; out_ch_offset < BLOCK_CHANNELS; ++out_ch_offset) {
            const int out_ch = block_channel * BLOCK_CHANNELS + out_ch_offset;
            
            if (out_ch < out_channels) {
                float result = acc[out_ch_offset];
                
                // Add convolution bias
                result += conv_bias[out_ch];
                
                // Apply ReLU
                result = fmaxf(0.0f, result);
                
                // Add bias
                result += bias[out_ch];
                
                // Write to output
                const int output_idx = ((batch * out_channels + out_ch) * output_height + h_out) * output_width + w_out;
                output[output_idx] = result;
            }
        }
    }
}

torch::Tensor Conv2D_ReLU_BiasAdd(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor conv_bias,
    torch::Tensor bias) {
    
    AT_ASSERTM(input.is_cuda(), "input must be a CUDA tensor");
    AT_ASSERTM(weight.is_cuda(), "weight must be a CUDA tensor");
    AT_ASSERTM(conv_bias.is_cuda(), "conv_bias must be a CUDA tensor");
    AT_ASSERTM(bias.is_cuda(), "bias must be a CUDA tensor");
    
    const int batch_size = input.size(0);
    const int in_channels = input.size(1);
    const int height = input.size(2);
    const int width = input.size(3);
    
    const int out_channels = weight.size(0);
    const int kernel_size = weight.size(2);
    
    AT_ASSERTM(weight.size(1) == in_channels, "Weight input channels must match input channels");
    AT_ASSERTM(conv_bias.size(0) == out_channels, "Conv bias size must match output channels");
    AT_ASSERTM(bias.size(0) == out_channels, "Bias size must match output channels");
    
    const int output_height = height - kernel_size + 1;
    const int output_width = width - kernel_size + 1;
    
    auto output = torch::zeros({batch_size, out_channels, output_height, output_width}, 
                              torch::dtype(torch::kFloat32).device(torch::kCUDA));
    
    const dim3 block_size(TILE_W, TILE_H, 1);
    const dim3 grid_size(
        (output_width + TILE_W - 1) / TILE_W,
        (output_height + TILE_H - 1) / TILE_H,
        batch_size * ((out_channels + BLOCK_CHANNELS - 1) / BLOCK_CHANNELS)
    );
    
    // Calculate required shared memory size
    const int TILE_H_IN = TILE_H + kernel_size - 1;
    const int TILE_W_IN = TILE_W + kernel_size - 1;
    const size_t shared_mem_size = TILE_H_IN * TILE_W_IN * BLOCK_CHANNELS * sizeof(float);
    
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
    
    Conv2D_ReLU_BiasAdd_kernel<<<grid_size, block_size, shared_mem_size, stream>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        conv_bias.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        in_channels,
        out_channels,
        height,
        width,
        kernel_size,
        output_height,
        output_width
    );
    
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("Conv2D_ReLU_BiasAdd", &Conv2D_ReLU_BiasAdd, "Conv2D + ReLU + BiasAdd");
}