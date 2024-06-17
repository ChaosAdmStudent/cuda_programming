#include <torch/extension.h> 
#include "utils.h"


// This function calls a cuda extension  
torch::Tensor trilinear_interpolation(
    torch::Tensor feats, 
    torch::Tensor point 
){ 
    CHECK_INPUT(feats);  
    CHECK_INPUT(point);

    return trilinear_fw_cu(feats, point); 
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
    m.def("trilinear_interpolation", &trilinear_interpolation, "Trilinear interpolation (CUDA)");

} 