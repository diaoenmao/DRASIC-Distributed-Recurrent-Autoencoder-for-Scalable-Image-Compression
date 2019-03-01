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
        {'input_size':256,'output_size':512,'num_layer':2,'cell':cell_name,'mode':'downsample','normalization':'bn','activation':'tanh','raw':False},
        {'input_size':512,'output_size':512,'num_layer':2,'cell':cell_name,'mode':'downsample','normalization':'bn','activation':'tanh','raw':False}
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

class LocalizerCell(nn.Module):
    def __init__(self, localizer_cell_info):
        super(LocalizerCell, self).__init__()
        self.localizer_cell_info = localizer_cell_info
        self.localizer_cell = self.make_localizer_cell()
        
    def make_localizer_cell(self):
        localizer_cell = nn.ModuleList([])
        for i in range(len(self.localizer_cell_info)):
            localizer_cell.append(Cell(self.localizer_cell_info[i]))
        return localizer_cell

    def free_hidden(self):
        for i in range(len(self.localizer_cell_info)):
            self.localizer_cell[i].hidden = None
        return
        
    def forward(self, input, protocol):
        x = input
        for i in range(len(self.localizer_cell)):          
            x = self.localizer_cell[i](x,protocol)
        return x
        
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

class myresnet(nn.Module):
    def __init__(self,classes_size):
        super(myresnet, self).__init__()
        self.classes_size = classes_size
        self.encoder = Encoder()
        self.classifier = Classifier(classes_size)

    def loss_fn(self, input, output, protocol):
        classification_loss = self.classifier.classification_loss_fn(input,output,protocol)
        loss = protocol['tuning_param']['classification']*classification_loss
        return loss
            
    def forward(self, input, protocol):
        output = {}        
        output['compression'] = {'img':0,'code':[]}
        
        img = input['img']
        code = self.encoder(img,protocol)
        output['compression']['code'] = code
        output['classification'] = self.classifier(code,protocol)
        classification_loss = self.classifier.classification_loss_fn(input,output,protocol)
        
        output['loss'] = protocol['tuning_param']['classification']*classification_loss
        return output     
    
    