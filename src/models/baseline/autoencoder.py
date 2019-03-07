import torch
import torch.nn as nn
import torch.nn.functional as F
from functions import pixel_unshuffle

class autoencoder(nn.Module):
    def __init__(self,classes_size):
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

    def compression_loss_fn(self, input, output, protocol):
        if(protocol['loss_mode']['compression'] == 'bce'):
            loss_fn = F.binary_cross_entropy
        elif(protocol['loss_mode']['compression'] == 'mse'):
            loss_fn = F.mse_loss
        elif(protocol['loss_mode']['compression'] == 'mae'):
            loss_fn = F.l1_loss
        else:
            raise ValueError('compression loss mode not supported') 
        if(protocol['tuning_param']['compression'] > 0):
            loss = loss_fn(output['compression']['img'],input['img'],reduction='sum')
            loss = loss/input['img'].size(0)
            loss = loss.mean()
        else:
            loss = torch.tensor(0,device=device,dtype=torch.float32) 
        return loss
        
    def forward(self, input, protocol):
        output = {'compression':{}}
        x = input['img']
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
        output['compression']['img'] = x
        output['loss'] = self.compression_loss_fn(input,output,protocol)
        return output