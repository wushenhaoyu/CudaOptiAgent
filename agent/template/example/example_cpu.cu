#include <torch/extension.h>
#include <vector>

static inline int64_t get_index(const int64_t* strides, 
                               int64_t i, int64_t j, int64_t k, int64_t l) {
    return i * strides[0] + j * strides[1] + k * strides[2] + l * strides[3];
}

torch::Tensor Conv2D_ReLU_BiasAdd(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor conv_bias,
    torch::Tensor bias) {
    
    // Extract tensor information
    float* input_ptr = input.data_ptr<float>();
    float* weight_ptr = weight.data_ptr<float>();
    float* conv_bias_ptr = conv_bias.data_ptr<float>();
    float* bias_ptr = bias.data_ptr<float>();
    
    auto input_sizes = input.sizes();
    auto weight_sizes = weight.sizes();
    auto bias_sizes = bias.sizes();
    
    auto input_strides = input.strides();
    auto weight_strides = weight.strides();
    auto bias_strides = bias.strides();
    
    // Input dimensions
    const int64_t N = input_sizes[0];
    const int64_t C_in = input_sizes[1];
    const int64_t H_in = input_sizes[2];
    const int64_t W_in = input_sizes[3];
    
    // Convolution parameters
    const int64_t C_out = weight_sizes[0];
    const int64_t K = weight_sizes[2];
    
    // Output spatial dimensions for valid convolution (padding=0, stride=1)
    const int64_t H_out = H_in - K + 1;
    const int64_t W_out = W_in - K + 1;
    
    // Create output tensor with correct spatial dimensions
    auto output = torch::zeros({N, C_out, H_out, W_out}, input.options());
    float* output_ptr = output.data_ptr<float>();
    auto output_strides = output.strides();
    
    // Main computation loop
    for (int64_t n = 0; n < N; ++n) {
        for (int64_t c_out = 0; c_out < C_out; ++c_out) {
            for (int64_t h_out = 0; h_out < H_out; ++h_out) {
                for (int64_t w_out = 0; w_out < W_out; ++w_out) {
                    
                    float conv_result = conv_bias_ptr[c_out];
                    
                    for (int64_t c_in = 0; c_in < C_in; ++c_in) {
                        for (int64_t kh = 0; kh < K; ++kh) {
                            for (int64_t kw = 0; kw < K; ++kw) {
                                
                                int64_t h_in = h_out + kh;
                                int64_t w_in = w_out + kw;
                                
                                int64_t input_idx = get_index(input_strides.data(), n, c_in, h_in, w_in);
                                int64_t weight_idx = get_index(weight_strides.data(), c_out, c_in, kh, kw);
                                
                                conv_result += input_ptr[input_idx] * weight_ptr[weight_idx];
                            }
                        }
                    }
                    
                    float relu_result = conv_result > 0.0f ? conv_result : 0.0f;
                    int64_t bias_idx = get_index(bias_strides.data(), c_out, 0, 0, 0);
                    float result = relu_result + bias_ptr[bias_idx];
                    
                    int64_t output_idx = get_index(output_strides.data(), n, c_out, h_out, w_out);
                    output_ptr[output_idx] = result;
                }
            }
        }
    }
    
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("Conv2D_ReLU_BiasAdd", &Conv2D_ReLU_BiasAdd, "Conv2D + ReLU + BiasAdd");
}