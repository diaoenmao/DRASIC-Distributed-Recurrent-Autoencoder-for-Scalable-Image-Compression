import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import config
from data import extract_patches_2d, reconstruct_from_patches_2d
from utils import RGB_to_L, L_to_RGB, apply_along_dim
from modules import Cell, Sign
from functions import pixel_unshuffle
config.init()
device = config.PARAM['device']
code_size = config.PARAM['code_size']
cell_name = config.PARAM['cell_name']

def free_hidden(module):
    for n, m in module.named_children():
        if(hasattr(m,'free_hidden')):
            m.free_hidden()
        if(sum(1 for _ in m.named_children())!=0): 
            free_hidden(m)
    return
            
class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.encoder_info = self.make_encoder_info()
        self.encoder = self.make_encoder()
        
    def make_encoder_info(self):
        encoder_in_info = [        
        [{'input_size':128,'output_size':128,'num_layer':1,'cell':cell_name,'mode':'pass','normalization':'none','activation':'none','raw':True},
        {'input_size':128,'output_size':128,'num_layer':1,'cell':cell_name,'mode':'pass','normalization':'none','activation':'none','raw':True}],                
        [{'input_size':512,'output_size':128,'num_layer':1,'cell':cell_name,'mode':'pass','normalization':'none','activation':'none','raw':True},
        {'input_size':128,'output_size':128,'num_layer':1,'cell':cell_name,'mode':'pass','normalization':'none','activation':'none','raw':True}],
        [{'input_size':512,'output_size':128,'num_layer':1,'cell':cell_name,'mode':'pass','normalization':'none','activation':'none','raw':True},
        {'input_size':128,'output_size':128,'num_layer':1,'cell':cell_name,'mode':'pass','normalization':'none','activation':'none','raw':True}],        
        ]
        encoder_hidden_info = [
        [{'input_size':128,'output_size':128,'num_layer':1,'cell':cell_name,'mode':'pass','normalization':'none','activation':'none','raw':True},
        {'input_size':128,'output_size':128,'num_layer':1,'cell':cell_name,'mode':'pass','normalization':'none','activation':'none','raw':True}],
        [{'input_size':128,'output_size':128,'num_layer':1,'cell':cell_name,'mode':'pass','normalization':'none','activation':'none','raw':True},
        {'input_size':128,'output_size':128,'num_layer':1,'cell':cell_name,'mode':'pass','normalization':'none','activation':'none','raw':True}],
        [{'input_size':128,'output_size':128,'num_layer':1,'cell':cell_name,'mode':'pass','normalization':'none','activation':'none','raw':True},
        {'input_size':128,'output_size':128,'num_layer':1,'cell':cell_name,'mode':'pass','normalization':'none','activation':'none','raw':True}]           
        ]
        encoder_shortcut_info = [
        [{'cell':'none'}],
        [{'input_size':512,'output_size':128,'num_layer':1,'cell':cell_name,'mode':'fc','normalization':'none','activation':'none','raw':True}],
        [{'input_size':512,'output_size':128,'num_layer':1,'cell':cell_name,'mode':'fc','normalization':'none','activation':'none','raw':True}]           
        ]
        encoder_info = [
        {'input_size':3,'output_size':32,'num_layer':1,'cell':cell_name,'mode':'pass','normalization':'none','activation':'tanh','raw':False},
        {'cell': 'ResLSTMCell','num_layer':2,'in': encoder_in_info[0],'hidden':encoder_hidden_info[0],'shortcut':encoder_shortcut_info[0],'activation':'tanh'},
        {'cell': 'ResLSTMCell','num_layer':2,'in': encoder_in_info[1],'hidden':encoder_hidden_info[1],'shortcut':encoder_shortcut_info[1],'activation':'tanh'},
        {'cell': 'ResLSTMCell','num_layer':2,'in': encoder_in_info[2],'hidden':encoder_hidden_info[2],'shortcut':encoder_shortcut_info[2],'activation':'tanh'},
        {'input_size':128,'output_size':code_size,'num_layer':1,'cell':cell_name,'mode':'fc','normalization':'none','activation':'tanh','raw':False}
        ]
        return encoder_info

    def make_encoder(self):
        encoder = nn.ModuleList([])
        for i in range(len(self.encoder_info)):
            encoder.append(Cell(self.encoder_info[i]))
        return encoder
        
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
        
class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.decoder_info = self.make_decoder_info()
        self.decoder = self.make_decoder()
        
    def make_decoder_info(self):
        decoder_in_info = [ 
        [{'input_size':128,'output_size':128,'num_layer':1,'cell':cell_name,'mode':'pass','normalization':'none','activation':'none','raw':True},
        {'input_size':128,'output_size':512,'num_layer':1,'cell':cell_name,'mode':'pass','normalization':'none','activation':'none','raw':True}],
        [{'input_size':128,'output_size':128,'num_layer':1,'cell':cell_name,'mode':'pass','normalization':'none','activation':'none','raw':True},
        {'input_size':128,'output_size':512,'num_layer':1,'cell':cell_name,'mode':'pass','normalization':'none','activation':'none','raw':True}],
        [{'input_size':128,'output_size':128,'num_layer':1,'cell':cell_name,'mode':'pass','normalization':'none','activation':'none','raw':True},
        {'input_size':128,'output_size':128,'num_layer':1,'cell':cell_name,'mode':'pass','normalization':'none','activation':'none','raw':True}],        
        ]
        decoder_hidden_info = [
        [{'input_size':128,'output_size':128,'num_layer':1,'cell':cell_name,'mode':'pass','normalization':'none','activation':'none','raw':True},
        {'input_size':512,'output_size':512,'num_layer':1,'cell':cell_name,'mode':'pass','normalization':'none','activation':'none','raw':True}],
        [{'input_size':128,'output_size':128,'num_layer':1,'cell':cell_name,'mode':'pass','normalization':'none','activation':'none','raw':True},
        {'input_size':512,'output_size':512,'num_layer':1,'cell':cell_name,'mode':'pass','normalization':'none','activation':'none','raw':True}], 
        [{'input_size':128,'output_size':128,'num_layer':1,'cell':cell_name,'mode':'pass','normalization':'none','activation':'none','raw':True},
        {'input_size':128,'output_size':128,'num_layer':1,'cell':cell_name,'mode':'pass','normalization':'none','activation':'none','raw':True}],        
        ]
        decoder_shortcut_info = [
        [{'input_size':128,'output_size':512,'num_layer':1,'cell':cell_name,'mode':'fc','normalization':'none','activation':'none','raw':True}],
        [{'input_size':128,'output_size':512,'num_layer':1,'cell':cell_name,'mode':'fc','normalization':'none','activation':'none','raw':True}], 
        [{'cell':'none'}],        
        ]
        decoder_info = [
        {'input_size':code_size,'output_size':128,'num_layer':1,'cell':cell_name,'mode':'fc','normalization':'none','activation':'tanh','raw':False},    
        {'cell': 'ResLSTMCell','num_layer':2,'in': decoder_in_info[0],'hidden':decoder_hidden_info[0],'shortcut':decoder_shortcut_info[0],'activation':'tanh'},
        {'cell': 'ResLSTMCell','num_layer':2,'in': decoder_in_info[1],'hidden':decoder_hidden_info[1],'shortcut':decoder_shortcut_info[1],'activation':'tanh'},
        {'cell': 'ResLSTMCell','num_layer':2,'in': decoder_in_info[2],'hidden':decoder_hidden_info[2],'shortcut':decoder_shortcut_info[2],'activation':'tanh'},       
        {'input_size':32,'output_size':3,'num_layer':1,'cell':cell_name,'mode':'pass','normalization':'none','activation':'tanh','raw':False},
        ]
        return decoder_info

    def make_decoder(self):
        decoder = nn.ModuleList([])
        for i in range(len(self.decoder_info)):
            decoder.append(Cell(self.decoder_info[i]))
        return decoder
        
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
            loss = loss_fn(output['compression']['img'],input['img'],reduction='sum')
            loss = loss/input['img'].size(0)
            loss = loss.mean()
        else:
            loss = torch.tensor(0,device=device,dtype=torch.float32) 
        return loss
        
class Classifier(nn.Module):
    def __init__(self, classes_size):
        super(Classifier, self).__init__()
        self.classes_size = classes_size
        self.classifier_info = self.make_classifier_info()
        self.classifier = self.make_classifier()
        
    def make_classifier_info(self):
        classifier_in_info = [ 
        [{'input_size':code_size,'output_size':512,'num_layer':1,'cell':cell_name,'mode':'pass','normalization':'none','activation':'none','raw':True}],
        [{'input_size':512,'output_size':self.classes_size,'num_layer':1,'cell':cell_name,'mode':'fc','normalization':'none','activation':'none','raw':True}],
        ]
        classifier_hidden_info = [
        [{'input_size':512,'output_size':512,'num_layer':1,'cell':cell_name,'mode':'pass','normalization':'none','activation':'none','raw':True}],
        [{'input_size':self.classes_size,'output_size':self.classes_size,'num_layer':1,'cell':cell_name,'mode':'fc','normalization':'none','activation':'none','raw':True}],      
        ]
        classifier_shortcut_info = [
        [{'input_size':code_size,'output_size':512,'num_layer':1,'cell':cell_name,'mode':'fc','normalization':'none','activation':'none','raw':True}],
        [{'input_size':512,'output_size':self.classes_size,'num_layer':1,'cell':cell_name,'mode':'fc','normalization':'none','activation':'none','raw':True}],         
        ]
        classifier_info = [
        {'cell': 'ResLSTMCell','num_layer':1,'in': classifier_in_info[0],'hidden':classifier_hidden_info[0],'shortcut':classifier_shortcut_info[0],'activation':'tanh'},
        {'cell': 'LSTMCell','num_layer':1,'in': classifier_in_info[1],'hidden':classifier_hidden_info[1],'activation':'tanh'},
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
            loss = torch.tensor(0,device=device,dtype=torch.float32) 
        return loss
        
    def forward(self, input, protocol):
        x = input
        for i in range(len(self.classifier)):
            x = self.classifier[i](x)        
        x = F.adaptive_avg_pool2d(x,1)
        x = x.view(x.size(0),self.classes_size)
        return x
        
class testnet_6(nn.Module):
    def __init__(self,classes_size):
        super(testnet_6, self).__init__()
        self.classes_size = classes_size
        self.codec = Codec()
        self.classifier = Classifier(classes_size)
            
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
        output['compression']['code'] = torch.stack(output['compression']['code'],dim=1)
        output['loss'] = protocol['tuning_param']['compression']*compression_loss
        free_hidden(self.codec)
        
        classification_loss = 0        
        if(protocol['tuning_param']['classification'] > 0):
            output['classification'] = torch.tensor(0,device=device)
            for i in range(protocol['num_iter']): 
                logit = self.classifier(output['compression']['code'][:,i],protocol)
                output['classification'] = output['classification'] + logit
                classification_loss = classification_loss + self.classifier.classification_loss_fn(input,output,protocol)  
            classification_loss = classification_loss/protocol['num_iter']
            output['loss'] += protocol['tuning_param']['classification']*classification_loss
            free_hidden(self.classifier)
        return output     
    
    