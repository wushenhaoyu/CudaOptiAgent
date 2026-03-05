#include "conv_transpose3d_add.cu"
#include "layernorm_channel_reduction.cu"
#include "avgpool3d_gelu.cu"

#include <pybind11/pybind11.h>

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("conv_transpose3d_add", &conv_transpose3d_add, "conv_transpose3d_add");
    m.def("layernorm_channel_reduction", &layernorm_channel_reduction, "layernorm_channel_reduction");
    m.def("avgpool3d_gelu", &avgpool3d_gelu, "avgpool3d_gelu");
}
