import torch
from torch.autograd.function import Function

class Quantize(Function):

    def __init__(self):
        super(Quantize, self).__init__()   
        
    @staticmethod
    def forward(ctx, input, inplace=False):
        if(inplace):
            output = input.round_()
        else:
            output = torch.round(input)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None 