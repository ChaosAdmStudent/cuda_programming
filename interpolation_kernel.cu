#include <torch/extension.h> 

torch::Tensor trilinear_fw_cu(
    torch::Tensor feats, 
    torch::Tensor point 
)
{
    /* 
    Inputs:
        feats: (N,8,F) 
        point: (N,3) 
    Outputs: 
        output: (N,F) 
    */
   
    const int N = feats.size(0), F = feats.size(2);

    //feat.options sets the device and dtype of the output tensor to be the same as feats
    torch::Tensor feat_interp = torch::empty({N, F}, feats.options()); 

    const dim3 threads(16,16); // 16 threads in each dimension  
    const dim3 blocks((N + threads.x - 1) / threads.x, (F + threads.y - 1) / threads.y);

    // This macro determines the type of the input tensor and calls function with the corresponding correct type signature
    AT_DISPATCH_FLOATING_TYPES(feats.type(), "trilinear_fw_cu", 
    ([&] { 
        // This launches the kernel. trilinear_fw_kernel is the kernel function 
        trilinear_fw_kernel<scalar_t><<<blocks, threads>>>(
            feats.packed_accessor<scalar_t, 3, torch::RestrictPtrTraits, size_t>(),
            points.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits, size_t>(),
            feat_interp.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits, size_t>()
        ); 
        // It took 3 inputs: the 2 CUDA tensors and the output tensor to be filled
    }));

}