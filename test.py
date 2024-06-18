import torch 
import cppcuda_tutorial  
import time 

# Pure python version of trilinear interpolation for testing 
def trilinear_interpolation(feats, points):
    """
    Inputs: 
        feats: (N, 8, F) tensor 
        points: (N, 3) tensor
    Outputs:
        feats_interp: (N, F) tensor 
    """

    u = (points[:, 0:1]+1)/2
    v = (points[:, 1:2]+1)/2
    w = (points[:, 2:3]+1)/2
    a = (1-v)*(1-w)
    b = (1-v)*w
    c = v*(1-w)
    d = 1-a-b-c

    feats_interp = (1-u)*(a*feats[:, 0] +
                          b*feats[:, 1] +
                          c*feats[:, 2] +
                          d*feats[:, 3]) + \
                       u*(a*feats[:, 4] +
                          b*feats[:, 5] +
                          c*feats[:, 6] +
                          d*feats[:, 7])
    
    return feats_interp

# Pytorch wrap for fw and bw pass in trilinear interpolation
class Trilinear_interpolation_cuda(torch.autograd.Function):
    @staticmethod # ctx is the context object to save tensors for backward pass
    def forward(ctx, feats, points):
        feat_interp = cppcuda_tutorial.trilinear_interpolation_fw(feats, points)

        ctx.save_for_backward(feats, points) # backprop has u,v,w which is extracted from points. feats is needed for redundant thread detection

        return feat_interp

    @staticmethod
    def backward(ctx, dL_dfeat_interp):
        feats, points = ctx.saved_tensors

        dL_dfeats = cppcuda_tutorial.trilinear_interpolation_bw(dL_dfeat_interp.contiguous(), feats, points)

        # We have to return as many values as we passed to forward (except ctx) 
        # We assumed points are fixed, so dL_dpoints is None 
        return dL_dfeats, None

if __name__ == '__main__': 

    device = 'cuda' if torch.cuda.is_available() else 'cpu' 

    N = 65536 
    F = 256 

    rand = torch.rand(N,8,F).to(device)  
    feats = rand.clone().requires_grad_()  # Used for Pytorch 
    feats2 = rand.clone().requires_grad_() # Used for CUDA
    points = torch.rand(N,3).to(device)*2 - 1

    # Checking time for CUDA 
    t = time.time() 
    # We can't directly use forward function anymore and need to use the class instead
    # out_cuda = cppcuda_tutorial.trilinear_interpolation_fw(feats2, points)     
    out_cuda = Trilinear_interpolation_cuda.apply(feats2, points)
    torch.cuda.synchronize() 
    print('CUDA fw time: ', time.time() - t)
    
    # Checking time for CPU 
    t = time.time()
    out_py = trilinear_interpolation(feats, points) 
    torch.cuda.synchronize() 
    print('Pytorch fw time: ', time.time() - t)

    print('fw all close: ',torch.allclose(out_cuda, out_py))  

    t = time.time() 
    loss = out_py.sum() 
    loss.backward() 
    torch.cuda.synchronize() 
    print('Pytorch bw time: ', time.time() - t)

    t = time.time() 
    loss2 = out_cuda.sum() 
    loss2.backward()
    torch.cuda.synchronize() 
    print('CUDA bw time: ', time.time() - t) 

    print('bw all close: ', torch.allclose(feats.grad, feats2.grad))
