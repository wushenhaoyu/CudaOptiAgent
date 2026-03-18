#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <float.h>
#include <math.h>

__global__ void softmax_kernel(const float* x, float* y, int rows, int cols) {
    int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= rows) return;

    // Each row corresponds to one (batch, head, query_position),
    // and softmax is reduced across key_sequence_length (cols = N).
    const float* row = x + r * cols;
    float* out = y + r * cols;

    // Numerically stable max reduction per row
    float m = -FLT_MAX;
    for (int i = 0; i < cols; ++i) {
        m = fmaxf(m, row[i]);
    }

    // Exponentiate shifted values and accumulate sum per row
    float s = 0.0f;
    for (int i = 0; i < cols; ++i) {
        float e = expf(row[i] - m);
        out[i] = e;
        s += e;
    }

    // Normalize
    float inv = 1.0f / s;
    for (int i = 0; i < cols; ++i) {
        out[i] *= inv;
    }
}

torch::Tensor Softmax_Reduction(torch::Tensor attn_scores) {
    int B = attn_scores.size(0);
    int H = attn_scores.size(1);
    int Nq = attn_scores.size(2);
    int Nk = attn_scores.size(3);

    auto out = torch::zeros_like(attn_scores);

    // rows = (B * H * query_position), cols = key_sequence_length
    int rows = B * H * Nq;
    int cols = Nk;

    int threads = 256;
    int blocks = (rows + threads - 1) / threads;
    softmax_kernel<<<blocks, threads>>>(
        attn_scores.data_ptr<float>(),
        out.data_ptr<float>(),
        rows,
        cols
    );
    return out;
}