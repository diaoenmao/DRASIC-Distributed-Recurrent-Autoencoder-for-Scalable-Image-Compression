import config
import torch
import numpy as np

device = config.PARAM['device']

def decorrelated_backward_propagation(grad):
    x = grad.view(grad.size(0),-1).t().cpu().numpy()
    cov = np.cov(x,rowvar=True)
    U,S,V = np.linalg.svd(cov)
    not_retained = S.cumsum(axis=0)>0.95*S.sum()
    U[:,not_retained] = 0
    whiten_matrix = U
    #whiten_matrix = U.dot(U.T)
    decorrelated_x = whiten_matrix.dot(x)
    decorrelated_x = torch.from_numpy(decorrelated_x).float().t().to(device)
    x = decorrelated_x.view(grad.size())
    return x
