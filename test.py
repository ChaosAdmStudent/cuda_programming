import torch 
import cppcuda_tutorial  


if __name__ == '__main__': 

    device = 'cuda' if torch.cuda.is_available() else 'cpu' 

    feats = torch.ones(2).to(device) 
    points = torch.zeros(2).to(device) 

    out = cppcuda_tutorial.trilinear_interpolation(feats, points) 

    print(out) 