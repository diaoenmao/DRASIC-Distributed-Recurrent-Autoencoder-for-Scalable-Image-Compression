import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import config
from data import extract_patches_2d, reconstruct_from_patches_2d
from utils import RGB_to_L, L_to_RGB
from modules import Cell, Sign
from functions import pixel_unshuffle
config.init()
device = config.PARAM['device']
code_size = config.PARAM['code_size']
cell_name = config.PARAM['cell_name']

class EncoderCell(nn.Module):
    def __init__(self, encoder_cell_info):
        super(EncoderCell, self).__init__()
        self.encoder_cell_info = encoder_cell_info
        self.encoder_cell = self.make_encoder_cell()
        self.hidden = None

    def make_encoder_cell(self):
        encoder_cell = Cell(self.encoder_cell_info)
        return encoder_cell
        
    def free_hidden(self):
        self.hidden = None
        return
        
    def forward(self, input):
        x = input
        self.hidden = self.encoder_cell(x,self.hidden)
        x = self.hidden[0]
        return x

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.encoder_info = self.make_encoder_info()
        self.encoder = self.make_encoder()
        
    def make_encoder_info(self):
        encoder_in_info = [       
        {'input_size':128,'output_size':128,'num_layer':1,'cell':cell_name,'mode':'pass','normalization':'none','activation':'none','raw':True},
        {'input_size':512,'output_size':128,'num_layer':1,'cell':cell_name,'mode':'pass','normalization':'none','activation':'none','raw':True},
        {'input_size':512,'output_size':128,'num_layer':1,'cell':cell_name,'mode':'pass','normalization':'none','activation':'none','raw':True},        
        ]
        encoder_hidden_info = [
        {'input_size':128,'output_size':128,'num_layer':1,'cell':cell_name,'mode':'pass','normalization':'none','activation':'none','raw':True},
        {'input_size':128,'output_size':128,'num_layer':1,'cell':cell_name,'mode':'pass','normalization':'none','activation':'none','raw':True},
        {'input_size':128,'output_size':128,'num_layer':1,'cell':cell_name,'mode':'pass','normalization':'none','activation':'none','raw':True}         
        ]
        encoder_info = [
        {'input_size':3,'output_size':32,'cell':'Conv2d','kernel_size':3,'stride':1,'padding':1,'dilation':1,'bias':False,'normalization':'none','activation':'tanh'},
        {'cell': 'GenericLSTM','in': encoder_in_info[0],'hidden':encoder_hidden_info[0],'activation':'tanh'},
        {'cell': 'GenericLSTM','in': encoder_in_info[1],'hidden':encoder_hidden_info[1],'activation':'tanh'},
        {'cell': 'GenericLSTM','in': encoder_in_info[2],'hidden':encoder_hidden_info[2],'activation':'tanh'},
        {'input_size':128,'output_size':code_size,'cell':'Conv2d','kernel_size':1,'stride':1,'padding':0,'dilation':1,'bias':False,'normalization':'none','activation':'tanh'}
        ]
        return encoder_info

    def make_encoder(self):
        encoder = nn.ModuleList([])
        encoder.append(Cell(self.encoder_info[0]))
        for i in range(1,len(self.encoder_info)-1):
            encoder.append(EncoderCell(self.encoder_info[i]))
        encoder.append(Cell(self.encoder_info[-1]))
        return encoder
        
    def free_hidden(self):
        for i in range(1,len(self.encoder)-1):
            self.encoder[i].free_hidden()
        return
        
    def forward(self, input, protocol):
        x = L_to_RGB(input) if (protocol['mode'] == 'L') else input
        x = self.encoder[0](x)
        for i in range(1,len(self.encoder)-1):
            x = pixel_unshuffle(x,protocol['jump_rate'])
            x = self.encoder[i](x)
        x = self.encoder[-1](x)
        return x

class Embedding(nn.Module):
    def __init__(self):
        super(Embedding, self).__init__()
        self.sign = Sign()

    def forward(self, input):
        x = self.sign(input)
        return x

class DecoderCell(nn.Module):
    def __init__(self, decoder_cell_info):
        super(DecoderCell, self).__init__()
        self.decoder_cell_info = decoder_cell_info
        self.decoder_cell = self.make_decoder_cell()
        self.hidden = None

    def make_decoder_cell(self):
        decoder_cell = Cell(self.decoder_cell_info)
        return decoder_cell
        
    def free_hidden(self):
        self.hidden = None
        return
        
    def forward(self, input):
        x = input
        self.hidden = self.decoder_cell(x,self.hidden)
        x = self.hidden[0]
        return x
        
class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.decoder_info = self.make_decoder_info()
        self.decoder = self.make_decoder()
        
    def make_decoder_info(self):
        decoder_in_info = [ 
        {'input_size':128,'output_size':512,'num_layer':1,'cell':cell_name,'mode':'pass','normalization':'none','activation':'none','raw':False},
        {'input_size':128,'output_size':512,'num_layer':1,'cell':cell_name,'mode':'pass','normalization':'none','activation':'none','raw':False},
        {'input_size':128,'output_size':128,'num_layer':1,'cell':cell_name,'mode':'pass','normalization':'none','activation':'none','raw':False},   
        ]
        decoder_hidden_info = [
        {'input_size':512,'output_size':512,'num_layer':1,'cell':cell_name,'mode':'pass','normalization':'none','activation':'none','raw':False},
        {'input_size':512,'output_size':512,'num_layer':1,'cell':cell_name,'mode':'pass','normalization':'none','activation':'none','raw':False},
        {'input_size':128,'output_size':128,'num_layer':1,'cell':cell_name,'mode':'pass','normalization':'none','activation':'none','raw':False},          
        ]
        decoder_info = [
        {'input_size':code_size,'output_size':128,'cell':'Conv2d','kernel_size':1,'stride':1,'padding':0,'dilation':1,'bias':False,'normalization':'none','activation':'tanh'},
        {'cell': 'GenericLSTM','in': decoder_in_info[0],'hidden':decoder_hidden_info[0],'activation':'tanh'},
        {'cell': 'GenericLSTM','in': decoder_in_info[1],'hidden':decoder_hidden_info[1],'activation':'tanh'},
        {'cell': 'GenericLSTM','in': decoder_in_info[2],'hidden':decoder_hidden_info[2],'activation':'tanh'},
        {'input_size':32,'output_size':3,'cell':'Conv2d','kernel_size':3,'stride':1,'padding':1,'dilation':1,'bias':False,'normalization':'none','activation':'tanh'}
        ]
        return decoder_info

    def make_decoder(self):
        decoder = nn.ModuleList([])
        decoder.append(Cell(self.decoder_info[0]))
        for i in range(1,len(self.decoder_info)-1):
            decoder.append(DecoderCell(self.decoder_info[i]))
        decoder.append(Cell(self.decoder_info[-1]))
        return decoder
    
    def free_hidden(self):
        for i in range(1,len(self.decoder)-1):
            self.decoder[i].free_hidden()
        return
        
    def forward(self, input, protocol):
        x = input
        x = self.decoder[0](x)
        for i in range(1,len(self.decoder)-1):
            x = self.decoder[i](x)
            x = F.pixel_shuffle(x,protocol['jump_rate'])
        x = self.decoder[-1](x)
        x = RGB_to_L(x) if (protocol['mode'] == 'L') else x
        return x
        
class Codec(nn.Module):
    def __init__(self):
        super(Codec, self).__init__()
        self.encoder = Encoder()
        self.embedding = Embedding()
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
            loss = loss_fn(output['compression']['img'],input['img'],reduction='mean')
            loss = loss.mean()
        else:
            loss = 0
        return loss

class ClassifierCell(nn.Module):
    def __init__(self, classifier_cell_info):
        super(ClassifierCell, self).__init__()
        self.classifier_cell_info = classifier_cell_info
        self.classifier_cell = self.make_classifier_cell()
        
    def make_classifier_cell(self):
        classifier_cell = Cell(self.classifier_cell_info)
        return classifier_cell
        
    def forward(self, input):
        x = input
        x = self.classifier_cell(x)
        return x
        
class Classifier(nn.Module):
    def __init__(self, classes_size):
        super(Classifier, self).__init__()
        self.classes_size = classes_size
        self.classifier_info = self.make_classifier_info()
        self.classifier = self.make_classifier()
        
    def make_classifier_info(self):
        classifier_info = [
        {'cell':'Conv2d','input_size':code_size,'output_size':512,'num_layers':1,'kernel_size':3,'stride':1,'padding':1,'dilation':1,'bias':False,'normalization':'none','activation':'tanh'},
        {'cell':'Conv2d','input_size':512,'output_size':self.classes_size,'num_layers':1,'kernel_size':1,'stride':1,'padding':0,'dilation':1,'bias':False,'normalization':'none','activation':'none'}
        ]
        return classifier_info

    def make_classifier(self):
        classifier = nn.ModuleList([])
        for i in range(len(self.classifier_info)):
            classifier.append(ClassifierCell(self.classifier_info[i]))
        return classifier
        
    def classification_loss_fn(self, input, output, protocol):
        if(protocol['loss_mode']['classification'] == 'ce'):
            loss_fn = F.cross_entropy
        else:
            raise ValueError('classification loss mode not supported')
        if(protocol['tuning_param']['classification'] > 0):
            loss = loss_fn(output['classification'],input['label'],reduction='sum')
            loss = loss.mean()
            loss = loss/input['img'].numel()
        else:
            loss = torch.tensor(0)
        return loss
        
    def forward(self, input, protocol):
        x = input
        for i in range(len(self.classifier_info)):
            x = self.classifier[i](x)
        x = F.adaptive_avg_pool2d(x,1)
        x = x.view(x.size(0),self.classes_size)
        return x
        
class testnet(nn.Module):
    def __init__(self,classes_size):
        super(testnet, self).__init__()
        self.classes_size = classes_size
        self.codec = Codec()
            
    def forward(self, input, protocol):
        output = {}        
        output['compression'] = {'img':0,'code':[]}
        
        img = input['img']
        
        compression_loss = 0
        compression_residual = img
        for i in range(protocol['num_iter']):
            if(i==0):
                compression_residual = img*2-1
            encoded = self.codec.encoder(compression_residual,protocol)
            output['compression']['code'].append(self.codec.embedding(encoded))
            decoded = self.codec.decoder(output['compression']['code'][i],protocol)
            if(i==0):
                decoded = (decoded+1)/2
                compression_residual = img - decoded
            else:
                compression_residual = compression_residual - decoded
            output['compression']['img'] = output['compression']['img'] + decoded
            compression_loss = compression_loss + self.codec.compression_loss_fn(input,output,protocol)
        compression_loss = compression_loss/protocol['num_iter']
        output['compression']['code'] = torch.cat(output['compression']['code'],dim=1)
        self.codec.encoder.free_hidden()
        self.codec.decoder.free_hidden()
        
        output['loss'] = protocol['tuning_param']['compression']*compression_loss
        return output     
    
    