#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_FLOAT(x) TORCH_CHECK(x.scalar_type() == at::kFloat, #x " must be float32")
#define EPS 1e-5f

__device__ __forceinline__ int idx4(int n, int c, int h, int w, int C, int H, int W) {
    return ((n * C + c) * H + h) * W + w;
}

__global__ void conv2d_bn_relu_k(
    const float* x, const float* w,
    const float* bn_w, const float* bn_b, const float* bn_m, const float* bn_v,
    float* y,
    int N, int Cin, int H, int W, int Cout,
    int Kh, int Kw, int pad, int stride, int Hout, int Wout,
    bool apply_relu)
{
    int linear = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * Cout * Hout * Wout;
    if (linear >= total) return;

    int ow = linear % Wout;
    int oh = (linear / Wout) % Hout;
    int oc = (linear / (Wout * Hout)) % Cout;
    int n  = linear / (Wout * Hout * Cout);

    float acc = 0.0f;
    for (int ic = 0; ic < Cin; ++ic) {
        for (int kh = 0; kh < Kh; ++kh) {
            int ih = oh * stride + kh - pad;
            if (ih < 0 || ih >= H) continue;
            for (int kw = 0; kw < Kw; ++kw) {
                int iw = ow * stride + kw - pad;
                if (iw < 0 || iw >= W) continue;
                float xv = x[idx4(n, ic, ih, iw, Cin, H, W)];
                float wv = w[(((oc * Cin + ic) * Kh + kh) * Kw + kw)];
                acc += xv * wv;
            }
        }
    }

    float scale = bn_w[oc] * rsqrtf(bn_v[oc] + EPS);
    float bias = bn_b[oc] - bn_m[oc] * scale;
    float outv = acc * scale + bias;
    if (apply_relu) outv = outv > 0.0f ? outv : 0.0f;
    y[idx4(n, oc, oh, ow, Cout, Hout, Wout)] = outv;
}

__global__ void maxpool3x3s2_k(const float* x, float* y, int N, int C, int H, int W, int Hout, int Wout) {
    int linear = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * C * Hout * Wout;
    if (linear >= total) return;

    int ow = linear % Wout;
    int oh = (linear / Wout) % Hout;
    int c  = (linear / (Wout * Hout)) % C;
    int n  = linear / (Wout * Hout * C);

    float m = -3.402823e38f;
    for (int kh = 0; kh < 3; ++kh) {
        int ih = oh * 2 + kh - 1;
        if (ih < 0 || ih >= H) continue;
        for (int kw = 0; kw < 3; ++kw) {
            int iw = ow * 2 + kw - 1;
            if (iw < 0 || iw >= W) continue;
            float v = x[idx4(n, c, ih, iw, C, H, W)];
            m = v > m ? v : m;
        }
    }
    y[idx4(n, c, oh, ow, C, Hout, Wout)] = m;
}

__global__ void concat_channel_k(const float* a, const float* b, float* out, int N, int Ca, int Cb, int H, int W) {
    int linear = blockIdx.x * blockDim.x + threadIdx.x;
    int Cout = Ca + Cb;
    int total = N * Cout * H * W;
    if (linear >= total) return;
    int w = linear % W;
    int h = (linear / W) % H;
    int c = (linear / (W * H)) % Cout;
    int n = linear / (W * H * Cout);
    if (c < Ca) out[idx4(n, c, h, w, Cout, H, W)] = a[idx4(n, c, h, w, Ca, H, W)];
    else out[idx4(n, c, h, w, Cout, H, W)] = b[idx4(n, c - Ca, h, w, Cb, H, W)];
}

__global__ void avgpool2x2s2_k(const float* x, float* y, int N, int C, int H, int W, int Hout, int Wout) {
    int linear = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * C * Hout * Wout;
    if (linear >= total) return;
    int ow = linear % Wout;
    int oh = (linear / Wout) % Hout;
    int c  = (linear / (Wout * Hout)) % C;
    int n  = linear / (Wout * Hout * C);

    int ih0 = oh * 2;
    int iw0 = ow * 2;
    float s = 0.0f;
    s += x[idx4(n, c, ih0, iw0, C, H, W)];
    s += x[idx4(n, c, ih0, iw0 + 1, C, H, W)];
    s += x[idx4(n, c, ih0 + 1, iw0, C, H, W)];
    s += x[idx4(n, c, ih0 + 1, iw0 + 1, C, H, W)];
    y[idx4(n, c, oh, ow, C, Hout, Wout)] = 0.25f * s;
}

__global__ void final_bn_relu_gap_k(
    const float* x,
    const float* bn_w, const float* bn_b, const float* bn_m, const float* bn_v,
    float* y, int N, int C, int H, int W)
{
    int linear = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * C;
    if (linear >= total) return;
    int c = linear % C;
    int n = linear / C;

    float scale = bn_w[c] * rsqrtf(bn_v[c] + EPS);
    float bias = bn_b[c] - bn_m[c] * scale;

    float s = 0.0f;
    int HW = H * W;
    for (int i = 0; i < HW; ++i) {
        int h = i / W;
        int w = i % W;
        float v = x[idx4(n, c, h, w, C, H, W)];
        v = v * scale + bias;
        v = v > 0.0f ? v : 0.0f;
        s += v;
    }
    y[n * C + c] = s / (float)HW;
}

__global__ void linear_k(const float* x, const float* w, const float* b, float* y, int N, int Cin, int Cout) {
    int linear = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * Cout;
    if (linear >= total) return;
    int oc = linear % Cout;
    int n = linear / Cout;

    float acc = b[oc];
    for (int ic = 0; ic < Cin; ++ic) {
        acc += x[n * Cin + ic] * w[oc * Cin + ic];
    }
    y[n * Cout + oc] = acc;
}

static at::Tensor conv_bn_relu(
    const at::Tensor& x, const at::Tensor& w,
    const at::Tensor& bn_w, const at::Tensor& bn_b, const at::Tensor& bn_m, const at::Tensor& bn_v,
    int pad, int stride, bool relu)
{
    int N = x.size(0), Cin = x.size(1), H = x.size(2), W = x.size(3);
    int Cout = w.size(0), Kh = w.size(2), Kw = w.size(3);
    int Hout = (H + 2 * pad - Kh) / stride + 1;
    int Wout = (W + 2 * pad - Kw) / stride + 1;
    auto y = torch::zeros({N, Cout, Hout, Wout}, x.options());
    int total = N * Cout * Hout * Wout;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    conv2d_bn_relu_k<<<blocks, threads, 0, at::cuda::getDefaultCUDAStream()>>>(
        x.data_ptr<float>(), w.data_ptr<float>(),
        bn_w.data_ptr<float>(), bn_b.data_ptr<float>(), bn_m.data_ptr<float>(), bn_v.data_ptr<float>(),
        y.data_ptr<float>(), N, Cin, H, W, Cout, Kh, Kw, pad, stride, Hout, Wout, relu
    );
    return y;
}

static at::Tensor maxpool3x3s2(const at::Tensor& x) {
    int N = x.size(0), C = x.size(1), H = x.size(2), W = x.size(3);
    int Hout = (H + 2 - 3) / 2 + 1;
    int Wout = (W + 2 - 3) / 2 + 1;
    auto y = torch::zeros({N, C, Hout, Wout}, x.options());
    int total = N * C * Hout * Wout;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    maxpool3x3s2_k<<<blocks, threads, 0, at::cuda::getDefaultCUDAStream()>>>(
        x.data_ptr<float>(), y.data_ptr<float>(), N, C, H, W, Hout, Wout
    );
    return y;
}

static at::Tensor concat_channel(const at::Tensor& a, const at::Tensor& b) {
    int N = a.size(0), Ca = a.size(1), H = a.size(2), W = a.size(3), Cb = b.size(1);
    auto y = torch::zeros({N, Ca + Cb, H, W}, a.options());
    int total = N * (Ca + Cb) * H * W;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    concat_channel_k<<<blocks, threads, 0, at::cuda::getDefaultCUDAStream()>>>(
        a.data_ptr<float>(), b.data_ptr<float>(), y.data_ptr<float>(), N, Ca, Cb, H, W
    );
    return y;
}

static at::Tensor avgpool2x2s2(const at::Tensor& x) {
    int N = x.size(0), C = x.size(1), H = x.size(2), W = x.size(3);
    int Hout = H / 2, Wout = W / 2;
    auto y = torch::zeros({N, C, Hout, Wout}, x.options());
    int total = N * C * Hout * Wout;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    avgpool2x2s2_k<<<blocks, threads, 0, at::cuda::getDefaultCUDAStream()>>>(
        x.data_ptr<float>(), y.data_ptr<float>(), N, C, H, W, Hout, Wout
    );
    return y;
}

at::Tensor DenseNet121(
    const at::Tensor& input,
    const at::Tensor& p0,
    const at::Tensor& p1, const at::Tensor& p2, const at::Tensor& p3, const at::Tensor& p4,
    const std::vector<at::Tensor>& params_tail)
{
    CHECK_CUDA(input); CHECK_CONTIGUOUS(input); CHECK_FLOAT(input);
    CHECK_CUDA(p0); CHECK_CUDA(p1); CHECK_CUDA(p2); CHECK_CUDA(p3); CHECK_CUDA(p4);
    auto x = conv_bn_relu(input, p0, p1, p2, p3, p4, 3, 2, true);
    x = maxpool3x3s2(x);

    int idx = 0;
    auto get = [&](int i) -> const at::Tensor& { return params_tail[i]; };

    int block_layers[4] = {6, 12, 24, 16};
    for (int b = 0; b < 4; ++b) {
        for (int l = 0; l < block_layers[b]; ++l) {
            auto y = conv_bn_relu(x, get(idx + 4), get(idx + 0), get(idx + 1), get(idx + 2), get(idx + 3), 1, 1, false);
            idx += 5;
            x = concat_channel(x, y);
        }
        if (b != 3) {
            x = conv_bn_relu(x, get(idx + 4), get(idx + 0), get(idx + 1), get(idx + 2), get(idx + 3), 0, 1, false);
            idx += 5;
            x = avgpool2x2s2(x);
        }
    }

    auto final_bn_w = get(idx + 0);
    auto final_bn_b = get(idx + 1);
    auto final_bn_m = get(idx + 2);
    auto final_bn_v = get(idx + 3);
    auto fc_w = get(idx + 4);
    auto fc_b = get(idx + 5);

    int N = x.size(0), C = x.size(1), H = x.size(2), W = x.size(3);
    auto pooled = torch::zeros({N, C}, x.options());
    {
        int total = N * C;
        int threads = 256;
        int blocks = (total + threads - 1) / threads;
        final_bn_relu_gap_k<<<blocks, threads, 0, at::cuda::getDefaultCUDAStream()>>>(
            x.data_ptr<float>(),
            final_bn_w.data_ptr<float>(), final_bn_b.data_ptr<float>(),
            final_bn_m.data_ptr<float>(), final_bn_v.data_ptr<float>(),
            pooled.data_ptr<float>(), N, C, H, W
        );
    }

    int Cout = fc_w.size(0);
    auto out = torch::zeros({N, Cout}, x.options());
    {
        int total = N * Cout;
        int threads = 256;
        int blocks = (total + threads - 1) / threads;
        linear_k<<<blocks, threads, 0, at::cuda::getDefaultCUDAStream()>>>(
            pooled.data_ptr<float>(), fc_w.data_ptr<float>(), fc_b.data_ptr<float>(), out.data_ptr<float>(), N, C, Cout
        );
    }
    return out;
}

at::Tensor DenseNet121_entry(
    const at::Tensor& input,
    const at::Tensor& p0,
    const at::Tensor& p1, const at::Tensor& p2, const at::Tensor& p3, const at::Tensor& p4,
    const at::Tensor& p5, const at::Tensor& p6, const at::Tensor& p7, const at::Tensor& p8, const at::Tensor& p9,
    const at::Tensor& p10, const at::Tensor& p11, const at::Tensor& p12, const at::Tensor& p13, const at::Tensor& p14,
    const at::Tensor& p15, const at::Tensor& p16, const at::Tensor& p17, const at::Tensor& p18, const at::Tensor& p19,
    const at::Tensor& p20, const at::Tensor& p21, const at::Tensor& p22, const at::Tensor& p23, const at::Tensor& p24,
    const at::Tensor& p25, const at::Tensor& p26, const at::Tensor& p27, const at::Tensor& p28, const at::Tensor& p29,
    const at::Tensor& p30, const at::Tensor& p31, const at::Tensor& p32, const at::Tensor& p33, const at::Tensor& p34,
    const at::Tensor& p35, const at::Tensor& p36, const at::Tensor& p37, const at::Tensor& p38, const at::Tensor& p39,
    const at::Tensor& p40, const at::Tensor& p41, const at::Tensor& p42, const at::Tensor& p43, const at::Tensor& p44,
    const at::Tensor& p45, const at::Tensor& p46, const at::Tensor& p47, const at::Tensor& p48, const at::Tensor& p49,
    const at::Tensor& p50, const at::Tensor& p51, const at::Tensor& p52, const at::Tensor& p53, const at::Tensor& p54,
    const at::Tensor& p55, const at::Tensor& p56, const at::Tensor& p57, const at::Tensor& p58, const at::Tensor& p59,
    const at::Tensor& p60, const at::Tensor& p61, const at::Tensor& p62, const at::Tensor& p63, const at::Tensor& p64,
    const at::Tensor& p65, const at::Tensor& p66, const at::Tensor& p67, const at::Tensor& p68, const at::Tensor& p69,
    const at::Tensor& p70, const at::Tensor& p71, const at::Tensor& p72, const at::Tensor& p73, const at::Tensor& p74,
    const at::Tensor& p75, const at::Tensor& p76, const at::Tensor& p77, const at::Tensor& p78, const at::Tensor& p79,
    const at::Tensor& p80, const at::Tensor& p81, const at::Tensor& p82, const at::Tensor& p83, const at::Tensor& p84,
    const at::Tensor& p85, const at::Tensor& p86, const at::Tensor& p87, const at::Tensor& p88, const at::Tensor& p89,
    const at::Tensor& p90, const at::Tensor& p91, const at::Tensor& p92, const at::Tensor& p93, const at::Tensor& p94,
    const at::Tensor& p95, const at::Tensor& p96, const at::Tensor& p97, const at::Tensor& p98, const at::Tensor& p99,
    const at::Tensor& p100, const at::Tensor& p101, const at::Tensor& p102, const at::Tensor& p103, const at::Tensor& p104,
    const at::Tensor& p105, const at::Tensor& p106, const at::Tensor& p107, const at::Tensor& p108, const at::Tensor& p109,
    const at::Tensor& p110, const at::Tensor& p111, const at::Tensor& p112, const at::Tensor& p113, const at::Tensor& p114,
    const at::Tensor& p115, const at::Tensor& p116, const at::Tensor& p117, const at::Tensor& p118, const at::Tensor& p119,
    const at::Tensor& p120, const at::Tensor& p121, const at::Tensor& p122, const at::Tensor& p123, const at::Tensor& p124,
    const at::Tensor& p125, const at::Tensor& p126, const at::Tensor& p127, const at::Tensor& p128, const at::Tensor& p129,
    const at::Tensor& p130, const at::Tensor& p131, const at::Tensor& p132, const at::Tensor& p133, const at::Tensor& p134,
    const at::Tensor& p135, const at::Tensor& p136, const at::Tensor& p137, const at::Tensor& p138, const at::Tensor& p139,
    const at::Tensor& p140, const at::Tensor& p141, const at::Tensor& p142, const at::Tensor& p143, const at::Tensor& p144,
    const at::Tensor& p145, const at::Tensor& p146, const at::Tensor& p147, const at::Tensor& p148, const at::Tensor& p149,
    const at::Tensor& p150, const at::Tensor& p151, const at::Tensor& p152, const at::Tensor& p153, const at::Tensor& p154,
    const at::Tensor& p155, const at::Tensor& p156, const at::Tensor& p157, const at::Tensor& p158, const at::Tensor& p159,
    const at::Tensor& p160, const at::Tensor& p161, const at::Tensor& p162, const at::Tensor& p163, const at::Tensor& p164,
    const at::Tensor& p165, const at::Tensor& p166, const at::Tensor& p167, const at::Tensor& p168, const at::Tensor& p169,
    const at::Tensor& p170, const at::Tensor& p171, const at::Tensor& p172, const at::Tensor& p173, const at::Tensor& p174,
    const at::Tensor& p175, const at::Tensor& p176, const at::Tensor& p177, const at::Tensor& p178, const at::Tensor& p179,
    const at::Tensor& p180, const at::Tensor& p181, const at::Tensor& p182, const at::Tensor& p183, const at::Tensor& p184,
    const at::Tensor& p185, const at::Tensor& p186, const at::Tensor& p187, const at::Tensor& p188, const at::Tensor& p189,
    const at::Tensor& p190, const at::Tensor& p191, const at::Tensor& p192, const at::Tensor& p193, const at::Tensor& p194,
    const at::Tensor& p195, const at::Tensor& p196, const at::Tensor& p197, const at::Tensor& p198, const at::Tensor& p199,
    const at::Tensor& p200, const at::Tensor& p201, const at::Tensor& p202, const at::Tensor& p203, const at::Tensor& p204,
    const at::Tensor& p205, const at::Tensor& p206, const at::Tensor& p207, const at::Tensor& p208, const at::Tensor& p209,
    const at::Tensor& p210, const at::Tensor& p211, const at::Tensor& p212, const at::Tensor& p213, const at::Tensor& p214,
    const at::Tensor& p215, const at::Tensor& p216, const at::Tensor& p217, const at::Tensor& p218, const at::Tensor& p219,
    const at::Tensor& p220, const at::Tensor& p221, const at::Tensor& p222, const at::Tensor& p223, const at::Tensor& p224,
    const at::Tensor& p225, const at::Tensor& p226, const at::Tensor& p227, const at::Tensor& p228, const at::Tensor& p229,
    const at::Tensor& p230, const at::Tensor& p231, const at::Tensor& p232, const at::Tensor& p233, const at::Tensor& p234,
    const at::Tensor& p235, const at::Tensor& p236, const at::Tensor& p237, const at::Tensor& p238, const at::Tensor& p239,
    const at::Tensor& p240, const at::Tensor& p241, const at::Tensor& p242, const at::Tensor& p243, const at::Tensor& p244,
    const at::Tensor& p245, const at::Tensor& p246, const at::Tensor& p247, const at::Tensor& p248, const at::Tensor& p249,
    const at::Tensor& p250, const at::Tensor& p251, const at::Tensor& p252, const at::Tensor& p253, const at::Tensor& p254,
    const at::Tensor& p255, const at::Tensor& p256, const at::Tensor& p257, const at::Tensor& p258, const at::Tensor& p259,
    const at::Tensor& p260, const at::Tensor& p261, const at::Tensor& p262, const at::Tensor& p263, const at::Tensor& p264,
    const at::Tensor& p265, const at::Tensor& p266, const at::Tensor& p267, const at::Tensor& p268, const at::Tensor& p269,
    const at::Tensor& p270, const at::Tensor& p271, const at::Tensor& p272, const at::Tensor& p273, const at::Tensor& p274,
    const at::Tensor& p275, const at::Tensor& p276, const at::Tensor& p277, const at::Tensor& p278, const at::Tensor& p279,
    const at::Tensor& p280, const at::Tensor& p281, const at::Tensor& p282, const at::Tensor& p283, const at::Tensor& p284,
    const at::Tensor& p285, const at::Tensor& p286, const at::Tensor& p287, const at::Tensor& p288, const at::Tensor& p289,
    const at::Tensor& p290, const at::Tensor& p291, const at::Tensor& p292, const at::Tensor& p293, const at::Tensor& p294,
    const at::Tensor& p295, const at::Tensor& p296, const at::Tensor& p297, const at::Tensor& p298, const at::Tensor& p299,
    const at::Tensor& p300, const at::Tensor& p301, const at::Tensor& p302, const at::Tensor& p303, const at::Tensor& p304,
    const at::Tensor& p305, const at::Tensor& p306, const at::Tensor& p307, const at::Tensor& p308, const at::Tensor& p309)
{
    std::vector<at::Tensor> tail = {
        p5,p6,p7,p8,p9,p10,p11,p12,p13,p14,p15,p16,p17,p18,p19,p20,p21,p22,p23,p24,p25,p26,p27,p28,p29,p30,p31,p32,p33,p34,
        p35,p36,p37,p38,p39,p40,p41,p42,p43,p44,p45,p46,p47,p48,p49,p50,p51,p52,p53,p54,p55,p56,p57,p58,p59,p60,p61,p62,p63,p64,
        p65,p66,p67,p68,p69,p70,p71,p72,p73,p74,p75,p76,p77,p78,p79,p80,p81,p82,p83,p84,p85,p86,p87,p88,p89,p90,p91,p92,p93,p94,
        p95,p96,p97,p98,p99,p100,p101,p102,p103,p104,p105,p106,p107,p108,p109,p110,p111,p112,p113,p114,p115,p116,p117,p118,p119,p120,p121,p122,p123,p124,
        p125,p126,p127,p128,p129,p130,p131,p132,p133,p134,p135,p136,p137,p138,p139,p140,p141,p142,p143,p144,p145,p146,p147,p148,p149,p150,p151,p152,p153,p154,
        p155,p156,p157,p158,p159,p160,p161,p162,p163,p164,p165,p166,p167,p168,p169,p170,p171,p172,p173,p174,p175,p176,p177,p178,p179,p180,p181,p182,p183,p184,
        p185,p186,p187,p188,p189,p190,p191,p192,p193,p194,p195,p196,p197,p198,p199,p200,p201,p202,p203,p204,p205,p206,p207,p208,p209,p210,p211,p212,p213,p214,
        p215,p216,p217,p218,p219,p220,p221,p222,p223,p224,p225,p226,p227,p228,p229,p230,p231,p232,p233,p234,p235,p236,p237,p238,p239,p240,p241,p242,p243,p244,
        p245,p246,p247,p248,p249,p250,p251,p252,p253,p254,p255,p256,p257,p258,p259,p260,p261,p262,p263,p264,p265,p266,p267,p268,p269,p270,p271,p272,p273,p274,
        p275,p276,p277,p278,p279,p280,p281,p282,p283,p284,p285,p286,p287,p288,p289,p290,p291,p292,p293,p294,p295,p296,p297,p298,p299,p300,p301,p302,p303,p304,p305,p306,p307,p308,p309
    };
    return DenseNet121(input, p0, p1, p2, p3, p4, tail);
}

PYBIND11_MODULE(DenseNet121, m) {
    m.def("DenseNet121", &DenseNet121_entry, "DenseNet121");
}