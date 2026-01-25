#include <torch/extension.h>
#include <cstdint>

torch::Tensor Square_matrix_multiplication_(torch::Tensor A, torch::Tensor B) {
    // Get tensor dimensions
    int64_t N = A.size(0);
    
    // Create output tensor
    torch::Tensor C = torch::zeros({N, N}, A.options());
    
    // Get data pointers
    float* A_data = A.data_ptr<float>();
    float* B_data = B.data_ptr<float>();
    float* C_data = C.data_ptr<float>();
    
    // Naive matrix multiplication (triple nested loop)
    for (int64_t i = 0; i < N; i++) {
        for (int64_t j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int64_t k = 0; k < N; k++) {
                sum += A_data[i * N + k] * B_data[k * N + j];
            }
            C_data[i * N + j] = sum;
        }
    }
    
    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("Square_matrix_multiplication_", &Square_matrix_multiplication_,
          "CPU reference implementation of square matrix multiplication");
}