import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import config
from data import extract_patches_2d, reconstruct_from_patches_2d
from utils import RGB_to_L, L_to_RGB, apply_fn
from modules import Cell, Quantizer

device = config.PARAM['device']
code_size = 32
activation = config.PARAM['activation']
            
class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.encoder_info = self.make_encoder_info()
        self.encoder = self.make_encoder()
        
    def make_encoder_info(self):
        encoder_in_info = [        
        [{'input_size':64,'output_size':256,'cell':'Conv2d','kernel_size':3,'stride':2,'padding':1,'normalization':'none','activation':'none','raw':True}],                
        [{'input_size':256,'output_size':512,'cell':'Conv2d','kernel_size':3,'stride':2,'padding':1,'normalization':'none','activation':'none','raw':True}],
        [{'input_size':512,'output_size':512,'cell':'Conv2d','kernel_size':3,'stride':2,'padding':1,'normalization':'none','activation':'none','raw':True}],        
        ]
        encoder_hidden_info = [
        [{'input_size':256,'output_size':256,'num_layer':1,'cell':'Conv2d','mode':'fc','normalization':'none','activation':'none','raw':True}],
        [{'input_size':512,'output_size':512,'num_layer':1,'cell':'Conv2d','mode':'fc','normalization':'none','activation':'none','raw':True}],
        [{'input_size':512,'output_size':512,'num_layer':1,'cell':'Conv2d','mode':'fc','normalization':'none','activation':'none','raw':True}]           
        ]
        encoder_info = [
        {'input_size':3,'output_size':64,'cell':'Conv2d','kernel_size':3,'stride':2,'padding':1,'normalization':'none','activation':activation,'raw':False},
        {'cell': 'LSTMCell','num_layer':1,'in': encoder_in_info[0],'hidden':encoder_hidden_info[0],'activation':activation},
        {'cell': 'LSTMCell','num_layer':1,'in': encoder_in_info[1],'hidden':encoder_hidden_info[1],'activation':activation}, 
        {'cell': 'LSTMCell','num_layer':1,'in': encoder_in_info[2],'hidden':encoder_hidden_info[2],'activation':activation},
        {'input_size':512,'output_size':code_size,'num_layer':1,'cell':'BasicCell','mode':'fc','normalization':'none','activation':'tanh','raw':False}
        ]
        return encoder_info

    def make_encoder(self):
        encoder = nn.ModuleList([])
        for i in range(len(self.encoder_info)):
            encoder.append(Cell(self.encoder_info[i]))
        return encoder
        
    def forward(self, input, protocol):
        x = L_to_RGB(input) if (protocol['img_mode'] == 'L') else input
        for i in range(len(self.encoder)):
            x = self.encoder[i](x)
        return x
        
class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.decoder_info = self.make_decoder_info()
        self.decoder = self.make_decoder()
        
    def make_decoder_info(self):
        decoder_in_info = [ 
        [{'input_size':512,'output_size':512,'num_layer':1,'cell':'BasicCell','mode':'pass','normalization':'none','activation':'none','raw':True}],
        [{'input_size':128,'output_size':512,'num_layer':1,'cell':'BasicCell','mode':'pass','normalization':'none','activation':'none','raw':True}],
        [{'input_size':128,'output_size':256,'num_layer':1,'cell':'BasicCell','mode':'pass','normalization':'none','activation':'none','raw':True}],
        [{'input_size':64,'output_size':128,'num_layer':1,'cell':'BasicCell','mode':'pass','normalization':'none','activation':'none','raw':True}],        
        ]
        decoder_hidden_info = [
        [{'input_size':512,'output_size':512,'num_layer':1,'cell':'BasicCell','mode':'fc','normalization':'none','activation':'none','raw':True}],
        [{'input_size':512,'output_size':512,'num_layer':1,'cell':'BasicCell','mode':'fc','normalization':'none','activation':'none','raw':True}], 
        [{'input_size':256,'output_size':256,'num_layer':1,'cell':'BasicCell','mode':'pass','normalization':'none','activation':'none','raw':True}],
        [{'input_size':128,'output_size':128,'num_layer':1,'cell':'BasicCell','mode':'pass','normalization':'none','activation':'none','raw':True}],        
        ]
        decoder_info = [
        {'input_size':code_size,'output_size':512,'num_layer':1,'cell':'BasicCell','mode':'fc','normalization':'none','activation':activation,'raw':False},      
        {'cell': 'LSTMCell','num_layer':1,'in': decoder_in_info[0],'hidden':decoder_hidden_info[0],'activation':activation},
        {'cell':'ShuffleCell','mode':'up','scale_factor':2},
        {'cell': 'LSTMCell','num_layer':1,'in': decoder_in_info[1],'hidden':decoder_hidden_info[1],'activation':activation},
        {'cell':'ShuffleCell','mode':'up','scale_factor':2},
        {'cell': 'LSTMCell','num_layer':1,'in': decoder_in_info[2],'hidden':decoder_hidden_info[2],'activation':activation},
        {'cell':'ShuffleCell','mode':'up','scale_factor':2},
        {'cell': 'LSTMCell','num_layer':1,'in': decoder_in_info[3],'hidden':decoder_hidden_info[3],'activation':activation},
        {'cell':'ShuffleCell','mode':'up','scale_factor':2},
        {'input_size':32,'output_size':3,'num_layer':1,'cell':'BasicCell','mode':'fc','normalization':'none','activation':'tanh','raw':False},
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
        x = RGB_to_L(x) if (protocol['img_mode'] == 'L') else x
        return x
        
class Codec(nn.Module):
    def __init__(self):
        super(Codec, self).__init__()
        self.encoder = Encoder()
        self.quantizer = Quantizer()
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
        [{'input_size':code_size,'output_size':512,'num_layer':1,'cell':'BasicCell','mode':'pass','normalization':'none','activation':'none','raw':True}],
        [{'input_size':512,'output_size':self.classes_size,'num_layer':1,'cell':'BasicCell','mode':'fc','normalization':'none','activation':'none','raw':True}],
        ]
        classifier_hidden_info = [
        [{'input_size':512,'output_size':512,'num_layer':1,'cell':'BasicCell','mode':'pass','normalization':'none','activation':'none','raw':True}],
        [{'input_size':self.classes_size,'output_size':self.classes_size,'num_layer':1,'cell':'BasicCell','mode':'fc','normalization':'none','activation':'none','raw':True}],      
        ]
        classifier_info = [
        {'cell': 'LSTMCell','num_layer':1,'in': classifier_in_info[0],'hidden':classifier_hidden_info[0],'activation':activation},
        {'cell': 'LSTMCell','num_layer':1,'in': classifier_in_info[1],'hidden':classifier_hidden_info[1],'activation':activation},
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
        
class toderici(nn.Module):
    def __init__(self,classes_size):
        super(toderici, self).__init__()
        self.classes_size = classes_size
        self.codec = Codec()
        self.classifier = Classifier(classes_size)
        
    def forward(self, input, protocol):
        output = {'loss':torch.tensor(0,device=device,dtype=torch.float32),
                'compression':{'img':torch.tensor(0,device=device,dtype=torch.float32),'code':[]},
                'classification':torch.tensor(0,device=device,dtype=torch.float32)}        
        
        compression_loss = torch.tensor(0,device=device,dtype=torch.float32) 
        compression_input = input['img']*2-1
        for i in range(protocol['num_iter']):
            encoded = self.codec.encoder(compression_input,protocol)
            output['compression']['code'].append(self.codec.quantizer(encoded))
            if(protocol['tuning_param']['compression'] > 0):
                decoded = self.codec.decoder(output['compression']['code'][i],protocol)
                decoded = (decoded+1)/2 if(i==0) else decoded
                compression_input = input['img']-decoded if(i==0) else compression_input-decoded
                output['compression']['img'] = output['compression']['img'] + decoded
                compression_loss = compression_loss + self.codec.compression_loss_fn(input,output,protocol)
        compression_loss = compression_loss/protocol['num_iter']
        output['compression']['code'] = torch.stack(output['compression']['code'],dim=1)
        apply_fn(self.codec,'free_hidden')
        
        classification_loss = torch.tensor(0,device=device,dtype=torch.float32)         
        if(protocol['tuning_param']['classification'] > 0):
            output['classification'] = torch.tensor(0,device=device)
            for i in range(protocol['num_iter']): 
                logit = self.classifier(output['compression']['code'][:,i],protocol)
                output['classification'] = output['classification'] + logit
                classification_loss = classification_loss + self.classifier.classification_loss_fn(input,output,protocol)  
            classification_loss = classification_loss/protocol['num_iter']
            apply_fn(self.classifier,'free_hidden')
            
        output['loss'] = protocol['tuning_param']['compression']*compression_loss + protocol['tuning_param']['classification']*classification_loss
        return output
    
    