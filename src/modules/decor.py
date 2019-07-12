import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import ntuple
from hooks import decorrelated_backward_propagation

class _decorConvNd(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias):
        super(_decorConvNd, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.weight = nn.Parameter(torch.Tensor(out_channels, in_channels // groups, *kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight,a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def extra_repr(self):
        s = ('{in_channels}, {out_channels}, kernel_size={kernel_size}'
             ', stride={stride}')
        if self.padding != (0,) * len(self.padding):
            s += ', padding={padding}'
        if self.dilation != (1,) * len(self.dilation):
            s += ', dilation={dilation}'
        if self.groups != 1:
            s += ', groups={groups}'
        if self.bias is None:
            s += ', bias=False'
        return s.format(**self.__dict__)  

class decorConv1d(_decorConvNd):

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias):
        _tuple = ntuple(1)
        kernel_size = _tuple(kernel_size)
        stride = _tuple(stride)
        padding = _tuple(padding)
        dilation = _tuple(dilation)
        super(decorConv1d, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
                       
    def forward(self, input, permutation):
        self.expanded_weight = self.weight.unsqueeze(dim=0).expand(input.size(0),*self.weight.size())
        self.h = self.expanded_weight.register_hook(decorrelated_backward_propagation)
        output = []
        for i in range(input.size(0)):
            o = F.conv1d(input=input[[i]], weight=self.expanded_weight[i], bias=self.bias,
                        stride=self.stride, padding=self.padding, dilation=self.dilation, groups=self.groups)
            output.append(o)
        output = torch.cat(output,dim=0)
        return output

class decorConv2d(_decorConvNd):

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, dilation=1, groups=1, bias=False):
        _tuple = ntuple(2)
        kernel_size = _tuple(kernel_size)
        stride = _tuple(stride)
        padding = _tuple(padding)
        dilation = _tuple(dilation)
        super(decorConv2d, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
                       
    def forward(self, input):
        self.expanded_weight = self.weight.unsqueeze(dim=0).expand(input.size(0),*self.weight.size())
        self.h = self.expanded_weight.register_hook(decorrelated_backward_propagation)
        output = []
        for i in range(input.size(0)):
            o = F.conv2d(input=input[[i]], weight=self.expanded_weight[i], bias=self.bias,
                        stride=self.stride, padding=self.padding, dilation=self.dilation, groups=self.groups)
            output.append(o)
        output = torch.cat(output,dim=0)
        return output

class decorConv3d(_decorConvNd):

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias):
        _tuple = ntuple(3)
        kernel_size = _tuple(kernel_size)
        stride = _tuple(stride)
        padding = _tuple(padding)
        dilation = _tuple(dilation)
        super(decorConv3d, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
                       
    def forward(self, input, permutation):
        self.expanded_weight = self.weight.unsqueeze(dim=0).expand(input.size(0),*self.weight.size())
        self.h = self.expanded_weight.register_hook(decorrelated_backward_propagation)
        output = []
        for i in range(input.size(0)):
            o = F.conv3d(input=input[[i]], weight=self.expanded_weight[i], bias=self.bias,
                        stride=self.stride, padding=self.padding, dilation=self.dilation, groups=self.groups)
            output.append(o)
        output = torch.cat(output,dim=0)
        return output
