import torch
import torch.nn as nn

from functions import Quantize as QuantizeFunction


class Quantize(nn.Module):
    def __init__(self,inplace=False):
        super().__init__()
        self.inplace = inplace
        
    def forward(self, x):
        return QuantizeFunction.apply(x, self.inplace)
