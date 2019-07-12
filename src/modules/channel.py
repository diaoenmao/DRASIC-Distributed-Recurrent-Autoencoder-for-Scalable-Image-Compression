import torch
import torch.nn as nn
from functions import channel

class Channel(nn.Module):
    def __init__(self, mode, snr):
        super(Channel, self).__init__()
        self.mode = mode
        self.snr = snr

    def forward(self, input):
        x = channel(input, self.mode, self.snr)
        return x