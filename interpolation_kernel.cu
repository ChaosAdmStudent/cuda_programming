#include <torch/extension.h> 

torch::Tensor trilinear_fw_cu(
    torch::Tensor feats, 
    torch::Tensor point 
){
    return feats; 
}