#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

// Optimized matrix multiplication using shared memory and tiling
template <int BLOCK_SIZE>
__global__ void square_matmul_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int N) {
    
    // Shared memory for tiles
    __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];
    
    int block_row = blockIdx.y;
    int block_col = blockIdx.x;
    
    int row = block_row * BLOCK_SIZE + threadIdx.y;
    int col = block_col * BLOCK_SIZE + threadIdx.x;
    
    float sum = 0.0f;
    
    // Loop over tiles
    for (int tile = 0; tile < gridDim.x; ++tile) {
        // Load tile from A
        int A_row = row;
        int A_col = tile * BLOCK_SIZE + threadIdx.x;
        if (A_row < N && A_col < N) {
            As[threadIdx.y][threadIdx.x] = A[A_row * N + A_col];
        } else {
            As[threadIdx.y][threadIdx.x] = 0.0f;
        }
        
        // Load tile from B
        int B_row = tile * BLOCK_SIZE + threadIdx.y;
        int B_col = col;
        if (B_row < N && B_col < N) {
            Bs[threadIdx.y][threadIdx.x] = B[B_row * N + B_col];
        } else {
            Bs[threadIdx.y][threadIdx.x] = 0.0f;
        }
        
        __syncthreads();
        
        // Compute partial sum for this tile
        #pragma unroll
        for (int k = 0; k < BLOCK_SIZE; ++k) {
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }
        
        __syncthreads();
    }
    
    // Write result
    if (row < N && col < N) {
        C[row * N + col] = sum;
    }
}

// Wrapper function with fixed block size
torch::Tensor Square_matrix_multiplication_(
    torch::Tensor A,
    torch::Tensor B) {
    
    // Input validation
    TORCH_CHECK(A.dim() == 2, "A must be 2D tensor");
    TORCH_CHECK(B.dim() == 2, "B must be 2D tensor");
    TORCH_CHECK(A.size(0) == A.size(1), "A must be square matrix");
    TORCH_CHECK(B.size(0) == B.size(1), "B must be square matrix");
    TORCH_CHECK(A.size(0) == B.size(0), "A and B must have same size");
    TORCH_CHECK(A.scalar_type() == torch::kFloat32, "A must be float32");
    TORCH_CHECK(B.scalar_type() == torch::kFloat32, "B must be float32");
    TORCH_CHECK(A.is_cuda(), "A must be CUDA tensor");
    TORCH_CHECK(B.is_cuda(), "B must be CUDA tensor");
    
    int N = A.size(0);
    auto C = torch::zeros({N, N}, A.options());
    
    // Set kernel configuration
    const int BLOCK_SIZE = 32;
    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid((N + BLOCK_SIZE - 1) / BLOCK_SIZE,
              (N + BLOCK_SIZE - 1) / BLOCK_SIZE);
    
    // Launch kernel
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    square_matmul_kernel<BLOCK_SIZE><<<grid, block, 0, stream>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        N);
    
    // Check for kernel launch errors
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    
    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("Square_matrix_multiplication_", &Square_matrix_multiplication_,
          "CUDA-optimized square matrix multiplication");
}