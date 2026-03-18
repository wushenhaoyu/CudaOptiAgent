#include <pybind11/pybind11.h>
#include <torch/extension.h>

#include "PatchEmbedding_GEMM.cu"
#include "CLS_PosAdd_Elementwise.cu"
#include "SelfAttn_QKV_Proj.cu"
#include "SelfAttn_QK_Scores.cu"
#include "Softmax_Reduction.cu"
#include "SelfAttn_AttnV_MatMul.cu"
#include "OutProj_GEMM_plus_Residual.cu"
#include "LayerNorm_Reduction.cu"
#include "MLP_FC1_GEMM_plus_GELU.cu"
#include "MLP_FC2_GEMM_plus_Residual.cu"
#include "ToCLS_Slice.cu"
#include "Head_FC1_GEMM_plus_GELU.cu"
#include "Head_FC2_GEMM.cu"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("PatchEmbedding_GEMM", &PatchEmbedding_GEMM, "PatchEmbedding_GEMM");
    m.def("CLS_PosAdd_Elementwise", &CLS_PosAdd_Elementwise, "CLS_PosAdd_Elementwise");
    m.def("SelfAttn_QKV_Proj", &SelfAttn_QKV_Proj, "SelfAttn_QKV_Proj");
    m.def("SelfAttn_QK_Scores", &SelfAttn_QK_Scores, "SelfAttn_QK_Scores");
    m.def("Softmax_Reduction", &Softmax_Reduction, "Softmax_Reduction");
    m.def("SelfAttn_AttnV_MatMul", &SelfAttn_AttnV_MatMul, "SelfAttn_AttnV_MatMul");
    m.def("OutProj_GEMM_plus_Residual", &OutProj_GEMM_plus_Residual, "OutProj_GEMM_plus_Residual");
    m.def("LayerNorm_Reduction", &LayerNorm_Reduction, "LayerNorm_Reduction");
    m.def("MLP_FC1_GEMM_plus_GELU", &MLP_FC1_GEMM_plus_GELU, "MLP_FC1_GEMM_plus_GELU");
    m.def("MLP_FC2_GEMM_plus_Residual", &MLP_FC2_GEMM_plus_Residual, "MLP_FC2_GEMM_plus_Residual");
    m.def("ToCLS_Slice", &ToCLS_Slice, "ToCLS_Slice");
    m.def("Head_FC1_GEMM_plus_GELU", &Head_FC1_GEMM_plus_GELU, "Head_FC1_GEMM_plus_GELU");
    m.def("Head_FC2_GEMM", &Head_FC2_GEMM, "Head_FC2_GEMM");
}
