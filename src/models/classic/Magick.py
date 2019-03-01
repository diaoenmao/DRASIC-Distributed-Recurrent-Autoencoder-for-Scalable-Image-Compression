import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import numpy as np
import config
import os
import io
import time
from PIL import Image
from utils import *

config.init()
device = config.PARAM['device']

class Magick(nn.Module):
    def __init__(self):
        super(Magick, self).__init__()
        self.supported_format = ['jpg','jp2','bpg','webp','png']
        
    def encode(self,input,protocol):
        format = protocol['format']
        if format not in self.supported_format:
            print('not supported format')
            exit()
        filename = protocol['filename']
        for i in range(len(input)):
            save_img(input[i].unsqueeze(0),'./output/tmp/{filename}_{idx}.png'.format(filename=filename,idx=i))
        if(format in ['jpg','jp2','bpg','webp','png']):
            head = 'magick mogrify '
            tail = './output/tmp/*.png'
            quality = protocol['quality']
            sampling_factor = protocol['sampling_factor']
            option = '-format {} -depth 8 '.format(format)
            if(quality is not None):
                option += '-quality {quality} '.format(quality=quality)
            if(sampling_factor is not None and (format in ['jpg','webp'])):
                option += '-sampling-factor {sampling_factor} '.format(sampling_factor=sampling_factor)
            command = '{head}{option}{tail}'.format(head=head, option=option, tail=tail)
        os.system(command)
        code = []
        for i in range(len(input)):
            f = open('./output/tmp/{filename}_{idx}.{format}'.format(filename=filename,idx=i,format=format), 'rb')
            buffer = f.read()
            f.close()
            code.append(np.frombuffer(buffer,dtype=np.uint8))
        return code

    def decode(self,code,protocol):
        format = protocol['format']
        if format not in self.supported_format:
            print('not supported format')
            exit()
        filename = protocol['filename']
        for i in range(len(code)): 
            try:
                f = open('./output/tmp/{filename}_{idx}.{format}'.format(filename=filename,idx=i,format=format), 'wb')
                f.write(code[i])
                f.close()
            except OSError:
                time.sleep(0.1)
                f = open('./output/tmp/{filename}_{idx}.{format}'.format(filename=filename,idx=i,format=format), 'wb')
                f.write(code[i])
                f.close()
        if(format in ['jpg','jp2','bpg','webp','png']):
            command = 'magick mogrify -format png -depth 8 ./output/tmp/*.{format}'.format(format=format)
            os.system(command)
        output = []
        ToTensor = torchvision.transforms.ToTensor()
        for i in range(len(code)):
            output.append(ToTensor(Image.open('./output/tmp/{filename}_{idx}.png'.format(filename=filename,idx=i))).to(device))
        return output

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
            loss = torch.tensor(0,device=device,dtype=torch.float32) 
            for i in range(len(input['img'])):
                loss += loss_fn(output['compression']['img'][i].unsqueeze(0),input['img'][i].unsqueeze(0),reduction='mean')
            loss = loss/len(input['img'])
        else:
            loss = torch.tensor(0,device=device,dtype=torch.float32) 
        return loss
        
    def forward(self,input,protocol):
        output = {'compression':{}}
        output['compression']['code'] = self.encode(input['img'],protocol)
        output['compression']['img'] = self.decode(output['compression']['code'],protocol)
        compression_loss = self.compression_loss_fn(input,output,protocol)
        output['loss'] = protocol['tuning_param']['compression']*compression_loss
        return output


