import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import collections.abc as container_abcs
from torch.autograd import Variable
from itertools import repeat

def _ntuple(n):
    def parse(x):
        if isinstance(x, container_abcs.Iterable):
            return x
        return tuple(repeat(x, n))
    return parse   

class _oConvNd(nn.Module):

    def __init__(self, max_in_channels, max_out_channels, max_kernel_size, bias):
        super(_oConvNd, self).__init__()
        self.max_in_channels = max_in_channels
        self.max_out_channels = max_out_channels
        self.max_kernel_size = max_kernel_size
        self.weight = nn.Parameter(torch.Tensor(max_out_channels, max_in_channels, *max_kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(max_out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def extra_repr(self):
        s = ('{max_in_channels}, {max_out_channels}, kernel_size={max_kernel_size}')
        if self.bias is None:
            s += ', bias=False'
        return s.format(**self.__dict__)    
        
class oConvNd(_oConvNd):

    def __init__(self, max_N, max_in_channels, max_out_channels, max_kernel_size, bias): 
        self.max_N = max_N
        self.max_ntuple = _ntuple(max_N)
        super(oConvNd, self).__init__(max_in_channels, max_out_channels, self.max_ntuple(max_kernel_size), bias)
                       
    def forward(self, input, protocol):
        N = protocol['N']
        in_coordinates = protocol['in_coordinates']
        out_coordinates = protocol['out_coordinates']
        ntuple = _ntuple(N)
        stride = ntuple(protocol['stride'])
        padding = ntuple(protocol['padding'])
        dilation = ntuple(protocol['dilation'])
        bias = self.bias[out_coordinates] if protocol['bias'] else None
        if(N == 1):
            conv = F.conv1d
        elif(N == 2):
            conv = F.conv2d
        elif(N == 3):
            conv = F.conv3d
        else:
            raise ValueError('data dimension not supported')
        return conv(input, self.weight[out_coordinates.view(-1,1),in_coordinates.view(1,-1),], bias, stride, padding, dilation, 1)
        