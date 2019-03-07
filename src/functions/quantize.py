import torch
from torch.autograd.function import Function

class Quantize(Function):
    def __init__(self):
        super(Quantize, self).__init__()   
        
    @staticmethod
    def forward(ctx, input, num_level, is_training):
        if is_training:
            prob = input.new(input.size()).uniform_()
            x = input.clone()
            if(num_level%2==0):
                x[((x+1)/2>=prob)&(x<=1)&(x>=-1)] = 1
                x[((x+1)/2<prob)&(x<=1)&(x>=-1)] = -1
                x[((x-x.floor())>=prob)&((x>1)|(x<-1))] = x.ceil()[((x-x.floor())>=prob)&((x>1)|(x<-1))]
                x[((x-x.floor())<prob)&((x>1)|(x<-1))] = x.floor()[((x-x.floor())<prob)&((x>1)|(x<-1))]
            else:
                x[(x-x.floor())>=prob] = x.ceil()[(x-x.floor())>=prob]
                x[(x-x.floor())<prob] = x.floor()[(x-x.floor())<prob]
        else:
            if(num_level%2==0):
                x = input.clone()
                x[(x>=0)&(x<=1)] = 1
                x[(x>=-1)&(x<0)] = -1
                x[(x>1)|(x<-1)] = x[(x>1)|(x<-1)].round()
            else:
                x = input.round()
        return x

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None, None