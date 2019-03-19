import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import numpy as np
import config
from data import extract_patches_2d, reconstruct_from_patches_2d
from utils import RGB_to_L, L_to_RGB, apply_fn
from modules import Cell, Quantizer

device = config.PARAM['device']
code_size = config.PARAM['code_size']
activation = config.PARAM['activation']
            
class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.encoder_info = self.make_encoder_info()
        self.encoder = self.make_encoder()
        
    def make_encoder_info(self):
        encoder_in_info = [        
        [{'input_size':128,'output_size':128,'num_layer':1,'cell':'BasicCell','mode':'pass','normalization':'none','activation':'none','raw':True}],                
        [{'input_size':512,'output_size':128,'num_layer':1,'cell':'BasicCell','mode':'pass','normalization':'none','activation':'none','raw':True}],
        [{'input_size':512,'output_size':128,'num_layer':1,'cell':'BasicCell','mode':'pass','normalization':'none','activation':'none','raw':True}],        
        ]
        encoder_hidden_info = [
        [{'input_size':128,'output_size':128,'num_layer':1,'cell':'BasicCell','mode':'pass','normalization':'none','activation':'none','raw':True}],
        [{'input_size':128,'output_size':128,'num_layer':1,'cell':'BasicCell','mode':'pass','normalization':'none','activation':'none','raw':True}],
        [{'input_size':128,'output_size':128,'num_layer':1,'cell':'BasicCell','mode':'pass','normalization':'none','activation':'none','raw':True}]           
        ]
        encoder_info = [
        {'input_size':3,'output_size':32,'num_layer':1,'cell':'BasicCell','mode':'fc','normalization':'none','activation':activation,'raw':False},
        {'cell':'ShuffleCell','mode':'down','scale_factor':2}, 
        {'cell': 'LSTMCell','num_layer':1,'in': encoder_in_info[0],'hidden':encoder_hidden_info[0],'activation':activation},
        {'cell':'ShuffleCell','mode':'down','scale_factor':2}, 
        {'cell': 'LSTMCell','num_layer':1,'in': encoder_in_info[1],'hidden':encoder_hidden_info[1],'activation':activation},
        {'cell':'ShuffleCell','mode':'down','scale_factor':2}, 
        {'cell': 'LSTMCell','num_layer':1,'in': encoder_in_info[2],'hidden':encoder_hidden_info[2],'activation':activation},
        {'input_size':128,'output_size':code_size,'num_layer':1,'cell':'BasicCell','mode':'fc','normalization':'none','activation':'tanh','raw':False}
        ]
        return encoder_info

    def make_encoder(self):
        encoder = nn.ModuleDict({})
        if(config.PARAM['num_node']['E']==0):
            encoder['0'] = nn.ModuleList([])
            for j in range(len(self.encoder_info)):
                encoder['0'].append(Cell(self.encoder_info[j]))
        else:
            for i in range(config.PARAM['num_node']['E']):
                encoder[str(i)] = nn.ModuleList([])
                for j in range(len(self.encoder_info)):
                    encoder[str(i)].append(Cell(self.encoder_info[j]))
        return encoder
        
    def forward(self, input, protocol):
        x = [None for i in range(len(protocol['split_map']['E']))]
        output = None
        for i in range(len(protocol['split_map']['E'])):
            cur_split_map = protocol['split_map']['E'][i]
            if(input[cur_split_map].size(0)==0):
                continue
            x[i] = input[cur_split_map]
            x[i] = L_to_RGB(x[i]) if (protocol['img_mode'] == 'L') else x[i]
            cur_node_name = str(i)
            for j in range(len(self.encoder[cur_node_name])):
                x[i] = self.encoder[cur_node_name][j](x[i])
            output = input.new_zeros(input.size(0),*x[i].size()[1:]) if(output is None) else output
            output[cur_split_map] = x[i]
        output = input if(output is None) else output
        return output
        
class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.decoder_info = self.make_decoder_info()
        self.decoder = self.make_decoder()
        
    def make_decoder_info(self):
        decoder_in_info = [ 
        [{'input_size':128,'output_size':512,'num_layer':1,'cell':'BasicCell','mode':'pass','normalization':'none','activation':'none','raw':True}],
        [{'input_size':128,'output_size':512,'num_layer':1,'cell':'BasicCell','mode':'pass','normalization':'none','activation':'none','raw':True}],
        [{'input_size':128,'output_size':128,'num_layer':1,'cell':'BasicCell','mode':'pass','normalization':'none','activation':'none','raw':True}],        
        ]
        decoder_hidden_info = [
        [{'input_size':512,'output_size':512,'num_layer':1,'cell':'BasicCell','mode':'pass','normalization':'none','activation':'none','raw':True}],
        [{'input_size':512,'output_size':512,'num_layer':1,'cell':'BasicCell','mode':'pass','normalization':'none','activation':'none','raw':True}], 
        [{'input_size':128,'output_size':128,'num_layer':1,'cell':'BasicCell','mode':'pass','normalization':'none','activation':'none','raw':True}],        
        ]
        decoder_info = [
        {'input_size':code_size,'output_size':128,'num_layer':1,'cell':'BasicCell','mode':'fc','normalization':'none','activation':activation,'raw':False},      
        {'cell': 'LSTMCell','num_layer':1,'in': decoder_in_info[0],'hidden':decoder_hidden_info[0],'activation':activation},
        {'cell':'ShuffleCell','mode':'up','scale_factor':2},
        {'cell': 'LSTMCell','num_layer':1,'in': decoder_in_info[1],'hidden':decoder_hidden_info[1],'activation':activation},
        {'cell':'ShuffleCell','mode':'up','scale_factor':2},
        {'cell': 'LSTMCell','num_layer':1,'in': decoder_in_info[2],'hidden':decoder_hidden_info[2],'activation':activation},
        {'cell':'ShuffleCell','mode':'up','scale_factor':2},        
        {'input_size':32,'output_size':3,'num_layer':1,'cell':'BasicCell','mode':'fc','normalization':'none','activation':'tanh','raw':False},
        ]
        return decoder_info

    def make_decoder(self):
        decoder = nn.ModuleDict({})
        if(config.PARAM['num_node']['D']==0):
            decoder['0'] = nn.ModuleList([])
            for j in range(len(self.decoder_info)):
                decoder['0'].append(Cell(self.decoder_info[j]))
        else:
            for i in range(config.PARAM['num_node']['D']):
                decoder[str(i)] = nn.ModuleList([])
                for j in range(len(self.decoder_info)):
                    decoder[str(i)].append(Cell(self.decoder_info[j]))
        return decoder
        
    def forward(self, input, protocol):
        x = [None for i in range(len(protocol['split_map']['D']))]
        output = None
        for i in range(len(protocol['split_map']['D'])):
            cur_split_map = protocol['split_map']['D'][i]
            if(input[cur_split_map].size(0)==0):
                continue
            x[i] = input[cur_split_map]
            cur_node_name = str(i)
            for j in range(len(self.decoder[cur_node_name])):
                x[i] = self.decoder[cur_node_name][j](x[i])
            x[i] = RGB_to_L(x[i]) if (protocol['img_mode'] == 'L') else x[i]
            output = input.new_zeros(input.size(0),*x[i].size()[1:]) if(output is None) else output
            output[cur_split_map] = x[i]
        output = input if(output is None) else output
        return output
        
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
            loss = torch.zeros(input['img'].size(0),device=device,dtype=torch.float32)
            for i in range(len(protocol['split_map']['E'])):
                cur_split_map = protocol['split_map']['E'][i]
                if(input['img'][cur_split_map].size(0)==0 or protocol['cur_iter']>=protocol['num_iter'][i]):
                    continue
                loss[cur_split_map] = loss[cur_split_map] +\
                    loss_fn(output['compression']['img'][cur_split_map],input['img'][cur_split_map],reduction='none').view(input['img'][cur_split_map].size(0),-1).mean(dim=1)                        
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
        
class iter_shuffle_codec(nn.Module):
    def __init__(self,classes_size):
        super(iter_shuffle_codec, self).__init__()
        self.classes_size = classes_size
        self.codec = Codec()
        self.classifier = Classifier(classes_size)
        
    def forward(self, input, protocol):
        output = {'loss':torch.tensor(0,device=device,dtype=torch.float32),
                'compression':{'img':torch.tensor(0,device=device,dtype=torch.float32),'code':[]},
                'classification':torch.tensor(0,device=device,dtype=torch.float32)}        
        
        compression_loss = torch.tensor(0,device=device,dtype=torch.float32) 
        compression_input = input['img']*2-1
        indices = torch.arange(input['img'].size(0),device=device)
        protocol['split_map'] = {}
        if('activate_node' in protocol):
            if(len(protocol['node_name']['E'])!=0 or len(protocol['node_name']['D'])!=0):
                if(len(protocol['node_name']['E'])!=0):
                    protocol['split_map']['E'] = [[] for i in range(len(protocol['node_name']['E']))]
                    protocol['split_map']['E'][protocol['activate_node']] = indices
                else:
                    protocol['split_map']['E'] = [indices]
                if(len(protocol['node_name']['D'])!=0):
                    protocol['split_map']['D'] = [[] for i in range(len(protocol['node_name']['D']))]
                    protocol['split_map']['D'][protocol['activate_node']] = indices
                else:
                    protocol['split_map']['D'] = [indices]
            else:
                protocol['split_map']['E'] = [indices]
                protocol['split_map']['D'] = [indices]
        else:
            if(protocol['num_class']>0):
                if(len(protocol['node_name']['E'])!=0 or len(protocol['node_name']['D'])!=0):
                    protocol['split_map']['E'] = [indices[input['label']==i] for i in range(len(protocol['node_name']['E']))] if(len(protocol['node_name']['E'])!=0) else [indices]
                    protocol['split_map']['D'] = [indices[input['label']==i] for i in range(len(protocol['node_name']['D']))] if(len(protocol['node_name']['D'])!=0) else [indices]
                else:
                    protocol['split_map']['E'] = [indices[input['label']<protocol['num_class']]]
                    protocol['split_map']['D'] = [indices[input['label']<protocol['num_class']]]
            else:
                protocol['split_map'] = {}
                protocol['split_map']['E'] = list(indices.chunk(len(protocol['node_name']['E']),dim=0)) if(len(protocol['node_name']['E'])!=0) else [indices]
                protocol['split_map']['D'] = list(indices.chunk(len(protocol['node_name']['D']),dim=0)) if(len(protocol['node_name']['D'])!=0) else [indices]
        if(isinstance(protocol['num_iter'],int)):
            protocol['max_num_iter'] = protocol['num_iter']
            protocol['num_iter'] = [protocol['num_iter'] for i in range(len(protocol['split_map']['E']))]
        else:
            protocol['max_num_iter'] = max(protocol['num_iter']) 
        if(self.training):
            for i in range(protocol['max_num_iter']):
                protocol['cur_iter'] = i
                encoded = self.codec.encoder(compression_input,protocol)
                output['compression']['code'].append(self.codec.quantizer(encoded))
                if(protocol['tuning_param']['compression'] > 0):
                    decoded = self.codec.decoder(output['compression']['code'][i],protocol)
                    decoded = (decoded+1)/2 if(i==0) else decoded
                    compression_input = input['img']-decoded if(i==0) else compression_input-decoded
                    output['compression']['img'] = output['compression']['img'] + decoded
                    compression_loss = compression_loss + self.codec.compression_loss_fn(input,output,protocol)
            compression_loss = (compression_loss.sum()/input['img'].size(0)).mean()
            compression_loss = compression_loss/protocol['max_num_iter']
            output['compression']['code'] = torch.stack(output['compression']['code'],dim=1)
            apply_fn(self.codec,'free_hidden')   
            output['loss'] = protocol['tuning_param']['compression']*compression_loss           
        else:
            output = [copy.deepcopy(output) for _ in range(protocol['max_num_iter'])]
            for i in range(protocol['max_num_iter']):
                protocol['cur_iter'] = i
                encoded = self.codec.encoder(compression_input,protocol)
                for j in range(i,protocol['max_num_iter']):
                    output[j]['compression']['code'].append(self.codec.quantizer(encoded))
                if(protocol['tuning_param']['compression'] > 0):
                    decoded = self.codec.decoder(output[i]['compression']['code'][i],protocol)
                    decoded = (decoded+1)/2 if(i==0) else decoded
                    compression_input = input['img']-decoded if(i==0) else compression_input-decoded
                    output[i]['compression']['img'] = output[i-1]['compression']['img'] + decoded if(i>0) else decoded
                    output[i]['loss'] = output[i-1]['loss'] + self.codec.compression_loss_fn(input,output[i],protocol) if(i>0) else self.codec.compression_loss_fn(input,output[i],protocol)
                    output[i]['loss'] = (output[i]['loss'].sum()/input['img'].size(0)).mean()
            for i in range(protocol['max_num_iter']):
                output[i]['loss'] = output[i]['loss']/protocol['max_num_iter']
                output[i]['compression']['code'] = torch.stack(output[i]['compression']['code'],dim=1)
            apply_fn(self.codec,'free_hidden')
        return output
    
    