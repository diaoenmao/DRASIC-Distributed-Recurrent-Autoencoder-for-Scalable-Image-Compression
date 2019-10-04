import torch.nn as nn
from functions import Quantize as QuantizeFunction


class Quantization(nn.Module):
    def __init__(self):
        super(Quantization, self).__init__()

    def forward(self, input):
        x = input
        x = QuantizeFunction.apply(x, self.training)
        return x