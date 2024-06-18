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

if __name__ == '__main__': 

    device = 'cuda' if torch.cuda.is_available() else 'cpu' 

    N = 65536 
    F = 256 

    feats = torch.rand(N,8,F).to(device) 
    points = torch.rand(N,3).to(device)*2 - 1

    # Checking time for CUDA 
    t = time.time() 
    out_cuda = cppcuda_tutorial.trilinear_interpolation(feats, points)  
    torch.cuda.synchronize() 
    print('CUDA time: ', time.time() - t)
    
    # Checking time for CPU 
    t = time.time()
    out_py = trilinear_interpolation(feats, points) 
    torch.cuda.synchronize() 
    print('Pytorch time: ', time.time() - t)

    print(torch.allclose(out_cuda, out_py)) 
