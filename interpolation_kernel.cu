#include <torch/extension.h> 

template <typename scalar_t>
__global__ void trilinear_fw_kernel(
    const torch::PackedTensorAccessor<scalar_t, 3, torch::RestrictPtrTraits, size_t> feats,
    const torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, size_t> points,
    torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, size_t> feat_interp
){
    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    const int f = blockIdx.y * blockDim.y + threadIdx.y;

    // Excluding redundant threads
    if (n>=feats.size(0) || f>=feats.size(2)) return;

    // point: [-1,1], so we normalize to [0,1] 
    // u,v,w are normalized distance from x,y,z axes
    const scalar_t u = (points[n][0]+1)/2;
    const scalar_t v = (points[n][1]+1)/2;
    const scalar_t w = (points[n][2]+1)/2;
    
    // weights for trilinear interpolation
    const scalar_t a = (1-v)*(1-w);
    const scalar_t b = (1-v)*w;
    const scalar_t c = v*(1-w);
    const scalar_t d = 1-a-b-c;
    feat_interp[n][f] = (1-u)*(a*feats[n][0][f] +
                               b*feats[n][1][f] +
                               c*feats[n][2][f] +
                               d*feats[n][3][f]) + 
                            u*(a*feats[n][4][f] +
                               b*feats[n][5][f] +
                               c*feats[n][6][f] +
                               d*feats[n][7][f]);
}


// If returning multiple tensors, use std::vector<torch::Tensor>
torch::Tensor trilinear_fw_cu(
    torch::Tensor feats, 
    torch::Tensor points 
)
{
    /* 
    Inputs:
        feats: (N,8,F) 
        points: (N,3) 
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

    // If multiple  tensors returning, do: 
    // {feat_interp, feat_interp2, feat_interp3} 
    return feat_interp;

}