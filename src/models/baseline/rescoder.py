import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import config
from data import extract_patches_2d, reconstruct_from_patches_2d
from utils import RGB_to_L, L_to_RGB
from modules import Cell

config.init()
device = config.PARAM['device']
code_size = config.PARAM['code_size']
cell_name = config.PARAM['cell_name']

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.encoder_info = self.make_encoder_info()
        self.encoder = self.make_encoder()
        
    def make_encoder_info(self):
        encoder_info = [
        {'input_size':3,'output_size':64,'cell':'Conv2d','kernel_size':3,'stride':1,'padding':1,'dilation':1,'bias':False,'normalization':'bn','activation':'tanh'},
        {'input_size':64,'output_size':128,'num_layer':2,'cell':cell_name,'mode':'pass','normalization':'bn','activation':'tanh','raw':False},
        {'input_size':128,'output_size':256,'num_layer':2,'cell':cell_name,'mode':'downsample','normalization':'bn','activation':'tanh','raw':False},
        {'input_size':256,'output_size':512,'num_layer':2,'cell':cell_name,'mode':'downsample','normalization':'bn','activation':'tanh','raw':False}
        ]
        return encoder_info

    def make_encoder(self):
        encoder = nn.ModuleList([])
        for i in range(len(self.encoder_info)):
            encoder.append(Cell(self.encoder_info[i]))
        return encoder
        
    def forward(self, input, protocol):
        x = L_to_RGB(input) if (protocol['mode'] == 'L') else input
        for i in range(len(self.encoder)):
            x = self.encoder[i](x)
        return x


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.decoder_info = self.make_decoder_info()
        self.decoder = self.make_decoder()
        
    def make_decoder_info(self):
        decoder_info = [
        {'input_size':512,'output_size':256,'num_layer':2,'cell':cell_name,'mode':'upsample','normalization':'bn','activation':'tanh','raw':False},
        {'input_size':256,'output_size':128,'num_layer':2,'cell':cell_name,'mode':'upsample','normalization':'bn','activation':'tanh','raw':False},
        {'input_size':128,'output_size':64,'num_layer':2,'cell':cell_name,'mode':'pass','normalization':'bn','activation':'tanh','raw':False},
        {'input_size':64,'output_size':3,'cell':'Conv2d','kernel_size':3,'stride':1,'padding':1,'dilation':1,'bias':False,'normalization':'none','activation':'tanh'}
        ]
        return decoder_info

    def make_decoder(self):
        decoder = nn.ModuleList([])
        for i in range(len(self.decoder_info)):
            decoder.append(Cell(self.decoder_info[i]))
        return decoder
        
    def forward(self, input, protocol):
        x = input
        for i in range(len(self.decoder)):
            x = self.decoder[i](x)      
        x = RGB_to_L(x) if (protocol['mode'] == 'L') else x
        return x

class Codec(nn.Module):
    def __init__(self):
        super(Codec, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
        
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
            loss = loss.mean()
            loss = loss/input['img'].numel()
        else:
            loss = 0
        return loss
        
class Classifier(nn.Module):
    def __init__(self, classes_size):
        super(Classifier, self).__init__()
        self.classes_size = classes_size
        self.classifier_info = self.make_classifier_info()
        self.classifier = self.make_classifier()
        
    def make_classifier_info(self):
        classifier_info = [
        {'cell':'Conv2d','input_size':512,'output_size':self.classes_size,'num_layers':1,'kernel_size':1,'stride':1,'padding':0,'dilation':1,'bias':False,'normalization':'none','activation':'none'}
        ]
        return classifier_info

    def make_classifier(self):
        classifier = nn.ModuleList([])
        for i in range(len(self.classifier_info)):
            classifier.append(Cell(self.classifier_info[i]))
        return classifier
        
    def classification_loss_fn(self, input, output, protocol):
        if(protocol['loss_mode']['classification'] == 'ce'):
            loss_fn = F.cross_entropy
        else:
            raise ValueError('classification loss mode not supported')
        if(protocol['tuning_param']['classification'] > 0):
            loss = loss_fn(output['classification'],input['label'],reduction='mean')
            loss = loss.mean()
        else:
            loss = torch.tensor(0)
        return loss
        
    def forward(self, input, protocol):
        x = input
        for i in range(len(self.classifier)):
            x = self.classifier[i](x)
        x = F.adaptive_avg_pool2d(x,1)
        x = x.view(x.size(0),self.classes_size)
        return x

class rescoder(nn.Module):
    def __init__(self,classes_size):
        super(rescoder, self).__init__()
        self.classes_size = classes_size
        self.codec = Codec()
        self.classifier = Classifier(classes_size)

    def loss_fn(self, input, output, protocol):
        classification_loss = self.classifier.classification_loss_fn(input,output,protocol)
        loss = protocol['tuning_param']['classification']*classification_loss
        return loss
            
    def forward(self, input, protocol):
        output = {}        
        output['compression'] = {'img':0,'code':0}
        
        img = input['img']
        img = img*2-1
        code = self.codec.encoder(img,protocol)
        reconstruct_img = self.codec.decoder(code,protocol)
        reconstruct_img = (reconstruct_img+1)/2
        output['compression']['code'] = code
        output['compression']['img'] = reconstruct_img        
        output['classification'] = self.classifier(code,protocol)
        compression_loss = self.codec.compression_loss_fn(input,output,protocol)
        classification_loss = self.classifier.classification_loss_fn(input,output,protocol)
        output['loss'] = protocol['tuning_param']['compression']*compression_loss + protocol['tuning_param']['classification']*classification_loss
        return output     
    
    