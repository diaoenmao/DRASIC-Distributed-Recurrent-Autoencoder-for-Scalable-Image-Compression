import torch
import torch.nn as nn
import torch.nn.functional as F


class autoencoder(nn.Module):
    def __init__(self):
        super(autoencoder, self).__init__()
        
        self.conv0 = nn.Conv2d(1, 128, kernel_size=1, stride=1, padding=0)
        self.conv1 = nn.Conv2d(512, 128, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(512, 128, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(512, 128, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(128, 32, kernel_size=1, stride=1, padding=0)
        
        self.conv5 = nn.Conv2d(32, 128, kernel_size=1, stride=1, padding=0)
        self.conv6 = nn.Conv2d(128, 512, kernel_size=3, stride=1, padding=1)
        self.conv7 = nn.Conv2d(128, 512, kernel_size=3, stride=1, padding=1)
        self.conv8 = nn.Conv2d(128, 512, kernel_size=3, stride=1, padding=1)
        self.conv9 = nn.Conv2d(128, 1, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x = (x - 0.5) * 2
        x = torch.tanh(self.conv0(x))
        x = pixel_unshuffle(x,2)
        x = torch.tanh(self.conv1(x))
        x = pixel_unshuffle(x,2)
        x = torch.tanh(self.conv2(x))
        x = pixel_unshuffle(x,2)
        x = torch.tanh(self.conv3(x))
        x = torch.tanh(self.conv4(x))
        
        x = torch.tanh(self.conv5(x))
        x = torch.tanh(self.conv6(x))
        x = F.pixel_shuffle(x,2)
        x = torch.tanh(self.conv7(x))
        x = F.pixel_shuffle(x,2)
        x = torch.tanh(self.conv8(x))
        x = F.pixel_shuffle(x,2)
        x = torch.tanh(self.conv9(x))
        x = (x + 1)*0.5
        return x