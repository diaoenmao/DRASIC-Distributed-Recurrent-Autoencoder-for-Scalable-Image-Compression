import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import config
from modules import Conv2dLSTM,Conv2dGRU, Sign
from functions import pixel_unshuffle
from data import extract_patches_2d, reconstruct_from_patches_2d
from utils import RGB_to_L, L_to_RGB, dict_to_device, apply_along_dim

config.init()
device = config.PARAM['device']
code_size = config.PARAM['code_size']
cell_name = config.PARAM['cell_name']

def activate(input, protocol):
    if('activation' not in protocol):
        return torch.tanh(input)
    else:
        if(protocol['activation']=='none'):
            return input
        elif(protocol['activation']=='tanh'):
            return torch.tanh(input)
        else:
            raise ValueError('Activation mode not supported')
    return

class Cell(nn.Module):
    def __init__(self, cell_info):
        super(Cell, self).__init__()
        self.cell_info = cell_info
        self.cell = self.make_cell(cell_info)
        self.hidden = None
        
    def parse_model_info(self, model_info):
        if(model_info['model'] == 'Conv2d'):
            model = nn.Conv2d(self.cell_info['input_size'],self.cell_info['output_size'],self.cell_info['kernel_size'],\
                self.cell_info['stride'],self.cell_info['padding'],self.cell_info['dilation'],1,self.cell_info['bias'])
            self.recurrent = False
        elif(model_info['model'] == 'Conv2dLSTM'):
            model = Conv2dLSTM(self.cell_info['input_size'],self.cell_info['output_size'],self.cell_info['num_layers'],self.cell_info['kernel_size'],self.cell_info['hidden_kernel_size'],\
                self.cell_info['stride'],self.cell_info['hidden_stride'],self.cell_info['padding'],self.cell_info['hidden_padding'],self.cell_info['dilation'],self.cell_info['bias'])
            self.recurrent = True
        elif(model_info['model'] == 'Conv2dGRU'):
            model = Conv2dGRU(self.cell_info['input_size'],self.cell_info['output_size'],self.cell_info['num_layers'],self.cell_info['kernel_size'],self.cell_info['hidden_kernel_size'],\
                self.cell_info['stride'],self.cell_info['hidden_stride'],self.cell_info['padding'],self.cell_info['hidden_padding'],self.cell_info['dilation'],self.cell_info['bias'])
            self.recurrent = True
        else:
            raise ValueError('parse model mode not supported')
        return model
                
    def make_cell(self, cell_info):
        cell = self.parse_model_info(cell_info)
        return cell

    def forward(self, input, protocol):
        if(self.recurrent):
            if(protocol['free_hidden']):
                x, _ = self.cell(input, None)
            else:
                x, self.hidden = self.cell(input, self.hidden)
        else:
            x = activate(apply_along_dim(input, fn=self.cell, dim=1),protocol)
        return x

class EncoderCell(nn.Module):
    def __init__(self, encoder_cell_info):
        super(EncoderCell, self).__init__()
        self.encoder_cell_info = encoder_cell_info
        self.encoder_cell = self.make_encoder_cell()

    def make_encoder_cell(self):
        encoder_cell = nn.ModuleList([])
        for i in range(len(self.encoder_cell_info)):   
            encoder_cell.append(Cell(self.encoder_cell_info[i]))
        return encoder_cell
        
    def free_hidden(self):
        for i in range(len(self.encoder_cell_info)):
            self.encoder_cell[i].hidden = None
        return
        
    def forward(self, input, protocol):
        x = input
        for i in range(len(self.encoder_cell)):          
            x = self.encoder_cell[i](x,protocol)
        return x

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.encoder_info = self.make_encoder_info()
        self.encoder = self.make_encoder()
        
    def make_encoder_info(self):
        encoder_info = [        
        [{'model':'Conv2d','input_size':3,'output_size':32,'num_layers':1,'kernel_size':1,'stride':1,'padding':0,'dilation':1,'bias':False}],
        [{'model':cell_name,'input_size':128,'output_size':128,'num_layers':1,'kernel_size':3,'hidden_kernel_size':3,'stride':1,'hidden_stride':1,'padding':1,'hidden_padding':1,'dilation':1,'bias':False}],
        [{'model':cell_name,'input_size':512,'output_size':128,'num_layers':1,'kernel_size':3,'hidden_kernel_size':3,'stride':1,'hidden_stride':1,'padding':1,'hidden_padding':1,'dilation':1,'bias':False}],
        [{'model':cell_name,'input_size':512,'output_size':128,'num_layers':1,'kernel_size':3,'hidden_kernel_size':3,'stride':1,'hidden_stride':1,'padding':1,'hidden_padding':1,'dilation':1,'bias':False}],
        [{'model':'Conv2d','input_size':128,'output_size':code_size,'num_layers':1,'kernel_size':1,'stride':1,'padding':0,'dilation':1,'bias':False}]
        ]
        return encoder_info

    def make_encoder(self):
        encoder = nn.ModuleList([])
        for i in range(len(self.encoder_info)):
            encoder.append(EncoderCell(self.encoder_info[i]))
        return encoder

    def free_hidden(self):
        for i in range(len(self.encoder)):
            self.encoder[i].free_hidden()
        return
        
    def forward(self, input, protocol):
        x = apply_along_dim(input,fn=L_to_RGB,dim=1) if (protocol['mode'] == 'L') else input
        x = self.encoder[0](x, protocol)
        for i in range(1,protocol['depth']+1):
            x = apply_along_dim(x, protocol['jump_rate'], fn=pixel_unshuffle, dim=1)
            x = self.encoder[i](x, protocol)
        x = self.encoder[-1](x, protocol)
        return x

class Embedding(nn.Module):
    def __init__(self):
        super(Embedding, self).__init__()
        self.sign = Sign()

    def forward(self, x, protocol):
        x = self.sign(x)
        return x
        
class DecoderCell(nn.Module):
    def __init__(self, decoder_cell_info):
        super(DecoderCell, self).__init__()
        self.decoder_cell_info = decoder_cell_info
        self.decoder_cell = self.make_decoder_cell()
        
    def make_decoder_cell(self):
        decoder_cell = nn.ModuleList([])
        for i in range(len(self.decoder_cell_info)):   
            decoder_cell.append(Cell(self.decoder_cell_info[i]))
        return decoder_cell
        
    def free_hidden(self):
        for i in range(len(self.decoder_cell_info)):
            self.decoder_cell[i].hidden = None
        return
        
    def forward(self, input, protocol):
        x = input
        for i in range(len(self.decoder_cell)):          
            x = self.decoder_cell[i](x,protocol)
        return x

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.decoder_info = self.make_decoder_info()
        self.decoder = self.make_decoder()
        
    def make_decoder_info(self):
        decoder_info = [
        [{'model':'Conv2d','input_size':code_size,'output_size':128,'num_layers':1,'kernel_size':1,'stride':1,'padding':0,'dilation':1,'bias':False}],
        [{'model':cell_name,'input_size':128,'output_size':512,'num_layers':1,'kernel_size':3,'hidden_kernel_size':3,'stride':1,'hidden_stride':1,'padding':1,'hidden_padding':1,'dilation':1,'bias':False}],
        [{'model':cell_name,'input_size':128,'output_size':512,'num_layers':1,'kernel_size':3,'hidden_kernel_size':3,'stride':1,'hidden_stride':1,'padding':1,'hidden_padding':1,'dilation':1,'bias':False}],
        [{'model':cell_name,'input_size':128,'output_size':128,'num_layers':1,'kernel_size':3,'hidden_kernel_size':3,'stride':1,'hidden_stride':1,'padding':1,'hidden_padding':1,'dilation':1,'bias':False}],
        [{'model':'Conv2d','input_size':32,'output_size':3,'num_layers':1,'kernel_size':1,'stride':1,'padding':0,'dilation':1,'bias':False}]
        ]
        return decoder_info

    def free_hidden(self):
        for i in range(len(self.decoder)):
            self.decoder[i].free_hidden()
        return
        
    def make_decoder(self):
        decoder = nn.ModuleList([])
        for i in range(len(self.decoder_info)):
            decoder.append(DecoderCell(self.decoder_info[i]))
        return decoder
        
    def forward(self, input, protocol):
        x = self.decoder[0](input, protocol)
        for i in range(1,protocol['depth']+1):
            x = self.decoder[i](x, protocol)
            x = apply_along_dim(x, protocol['jump_rate'], fn=F.pixel_shuffle, dim=1)
        x = self.decoder[-1](x, protocol)
        x = apply_along_dim(x,fn=RGB_to_L,dim=1) if (protocol['mode'] == 'L') else x
        return x
        
class Codec(nn.Module):
    def __init__(self):
        super(Codec, self).__init__()
        self.encoder = Encoder()
        self.embedding = Embedding()
        self.decoder = Decoder()
        
    def free_hidden(self):
        self.encoder.free_hidden()
        self.decoder.free_hidden()           
        return
        
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
            loss = apply_along_dim(output['compression']['img'],input['img'],fn=loss_fn,dim=[1,0],reduction='sum')
            loss = loss/input['img'].size(0)
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
        classifier_cell = nn.ModuleList([])
        for i in range(len(self.classifier_cell_info)):
            classifier_cell.append(Cell(self.classifier_cell_info[i]))
        return classifier_cell

    def free_hidden(self):
        for i in range(len(self.classifier_cell_info)):
            self.classifier_cell[i].hidden = None
        return
        
    def forward(self, input, protocol):
        x = input
        for i in range(len(self.classifier_cell)):          
            x = self.classifier_cell[i](x,protocol)
        return x
        
class Classifier(nn.Module):
    def __init__(self, classes_size):
        super(Classifier, self).__init__()
        self.classes_size = classes_size
        self.classifier_info = self.make_classifier_info()
        self.classifier = self.make_classifier()
        
    def make_classifier_info(self):
        classifier_info = [
                        [{'model':cell_name,'input_size':code_size,'output_size':512,'num_layers':1,'kernel_size':3,'hidden_kernel_size':3,'stride':1,'hidden_stride':1,'padding':1,'hidden_padding':1,'dilation':1,'bias':False}],
                        [{'model':cell_name,'input_size':512,'output_size':self.classes_size,'num_layers':1,'kernel_size':1,'hidden_kernel_size':1,'stride':1,'hidden_stride':1,'padding':0,'hidden_padding':0,'dilation':1,'bias':False}]
                        ]
        return classifier_info

    def make_classifier(self):
        classifier = nn.ModuleList([])
        for i in range(len(self.classifier_info)):
            classifier.append(ClassifierCell(self.classifier_info[i]))
        return classifier

    def free_hidden(self):
        for i in range(len(self.classifier)):
            self.classifier[i].free_hidden()
        return
        
    def classification_loss_fn(self, input, output, protocol):
        if(protocol['loss_mode']['classification'] == 'ce'):
            loss_fn = F.cross_entropy
        else:
            raise ValueError('classification loss mode not supported')
        if(protocol['tuning_param']['classification'] > 0):
            loss = apply_along_dim(output['classification'],input['label'],fn=loss_fn,dim=[1,0],reduction='mean')
            loss = loss.mean()
        else:
            loss = 0
        return loss
        
    def forward(self, input, protocol):
        x = self.classifier[0](input,protocol)
        for i in range(1,len(self.classifier_info)-1):
            x = self.classifier[i](x, protocol)
        x = apply_along_dim(x,1,fn=F.adaptive_avg_pool2d,dim=1)
        x = self.classifier[-1](x,protocol)
        x = x.view(x.size(0),x.size(1),self.classes_size)
        return x
        
class Joint(nn.Module):
    def __init__(self,classes_size):
        super(Joint, self).__init__()
        self.classes_size = classes_size
        self.codec = Codec()
        self.classifier = Classifier(classes_size)
    
    def loss_fn(self, input, output, protocol):
        compression_loss = self.codec.compression_loss_fn(input,output,protocol)
        classification_loss = self.classifier.classification_loss_fn(input,output,protocol)
        loss = protocol['tuning_param']['compression']*compression_loss + protocol['tuning_param']['classification']*classification_loss
        return loss
            
    def forward(self, input, protocol):
        output = {}        
        protocol['free_hidden'] = False
        output['compression'] = {'img':0,'code':[]}
        
        img = input['img'].unsqueeze(dim=1)
        patches = apply_along_dim(img,protocol['patch_shape'],protocol['step'],fn=extract_patches_2d,dim=[1,2])
        patches = patches.view(-1,*patches.size()[2:])
        batch_size, patch_size = patches.size(0), patches.size(1)
        
        compression_loss = 0
        compression_residual = patches
        for i in range(protocol['num_iter']):
            if(i==0):
                compression_residual = patches*2-1
            encoded = self.codec.encoder(compression_residual,protocol)
            output['compression']['code'].append(self.codec.embedding(encoded,protocol))
            decoded = self.codec.decoder(output['compression']['code'][i],protocol)
            if(i==0):
                decoded = (decoded+1)/2
                compression_residual = patches - decoded
            else:
                compression_residual = compression_residual - decoded
            decoded = apply_along_dim(decoded.view(batch_size,patch_size,*decoded.size()[1:]),\
                protocol['img_shape'],protocol['step'],fn=reconstruct_from_patches_2d,dim=[2,1])
            output['compression']['img'] = output['compression']['img'] + decoded
            compression_loss = compression_loss + self.codec.compression_loss_fn(input,output,protocol)
        compression_loss = compression_loss/protocol['num_iter']
        output['compression']['code'] = torch.cat(output['compression']['code'],dim=1)
        output['compression']['code'] = apply_along_dim(output['compression']['code'].view(batch_size,patch_size,*output['compression']['code'].size()[1:]),\
            (protocol['img_shape'][0]//(protocol['jump_rate']**protocol['depth']),protocol['img_shape'][1]//(protocol['jump_rate']**protocol['depth'])),protocol['step'],fn=reconstruct_from_patches_2d,dim=[2,1])
        output['compression']['img'] = output['compression']['img'][:,0]
        self.codec.free_hidden()

        classification_loss = 0        
        if(protocol['tuning_param']['classification'] > 0):
            output['classification'] = torch.tensor(0,device=device)
            for i in range(protocol['num_iter']): 
                output['classification'] = output['classification'] + self.classifier(output['compression']['code'][:,[i]],protocol)
                classification_loss = classification_loss + self.classifier.classification_loss_fn(input,output,protocol)
            classification_loss = classification_loss/protocol['num_iter']
            output['classification'] = output['classification'][:,0]
        output['loss'] = protocol['tuning_param']['compression']*compression_loss + protocol['tuning_param']['classification']*classification_loss
        self.classifier.free_hidden()
        return output     
    
    