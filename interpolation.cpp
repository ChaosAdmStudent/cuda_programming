#include "utils.h"

// This function calls a cuda extension  
torch::Tensor trilinear_interpolation_fw(
    torch::Tensor feats, 
    torch::Tensor points 
){ 
    CHECK_INPUT(feats);  
    CHECK_INPUT(points);

    return trilinear_fw_cu(feats, points); 
}

torch::Tensor trilinear_interpolation_bw(
    const torch::Tensor dL_dfeat_interp,
    const torch::Tensor feats,
    const torch::Tensor points
){
    CHECK_INPUT(dL_dfeat_interp);
    CHECK_INPUT(feats);
    CHECK_INPUT(points);

    return trilinear_bw_cu(dL_dfeat_interp, feats, points);
} 

// Providing a interface for python to call. Basically allowing python to call above function 
// This is done using Pybind 

/* 
    TORCH_EXTENSION_NAME is a macro that is defined in torch/extension.h. 
    It is used to define the name of the module that will be created by the Pybind11 module. 

    m is the variable that will be used to add definitions to this module 

*/

// This is basically the C++ bridge to python
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) 
{
    m.def("trilinear_interpolation_fw", &trilinear_interpolation_fw);
    m.def("trilinear_interpolation_bw", &trilinear_interpolation_bw); 
} 