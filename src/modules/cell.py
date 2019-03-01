import torch.nn as nn
import torch.nn.functional as F
import torch
import config
from collections import OrderedDict 
from utils import _ntuple,apply_along_dim

config.init()
device = config.PARAM['device']

def Normalization(normalization,output_size):
    if(normalization=='none'):
        return nn.Sequential()
    elif(normalization=='bn'):
        return nn.BatchNorm2d(output_size)
    elif(normalization=='in'):
        return nn.InstanceNorm2d(output_size)
    else:
        raise ValueError('Normalization mode not supported')
    return
    
def Activation(activation,inplace=False):
    if(activation=='none'):
        return nn.Sequential()
    elif(activation=='tanh'):
        return nn.Tanh()
    elif(activation=='relu'):
        return nn.ReLU()
    elif(activation=='prelu'):
        return nn.PReLU()
    elif(activation=='elu'):
        return nn.ELU()
    elif(activation=='selu'):
        return nn.SELU()
    elif(activation=='celu'):
        return nn.CELU()
    elif(activation=='logsoftmax'):
        return nn.SoftMax(dim=-1)
    else:
        raise ValueError('Activation mode not supported')
    return

class BasicCell(nn.Module):
    def __init__(self, cell_info):
        super(BasicCell, self).__init__()
        self.cell_info = cell_info
        self.cell = self.make_cell()
        
    def make_cell(self):
        cell = nn.ModuleList([])
        if(self.cell_info['mode']=='downsample'):
            cell_info = {'cell':'Conv2d','normalization':self.cell_info['normalization'],'activation':self.cell_info['activation'],
                        'input_size':self.cell_info['input_size'],'output_size':self.cell_info['output_size'],
                        'kernel_size':2,'stride':2,'padding':0,'dilation':1,'group':1,'bias':False}
        elif(self.cell_info['mode']=='pass'):
            cell_info = {'cell':'Conv2d','normalization':self.cell_info['normalization'],'activation':self.cell_info['activation'],
                        'input_size':self.cell_info['input_size'],'output_size':self.cell_info['output_size'],
                        'kernel_size':3,'stride':1,'padding':1,'dilation':1,'group':1,'bias':False}
        elif(self.cell_info['mode']=='upsample'):
            cell_info = {'cell':'ConvTranspose2d','normalization':self.cell_info['normalization'],'activation':self.cell_info['activation'],
                        'input_size':self.cell_info['input_size'],'output_size':self.cell_info['output_size'],
                        'kernel_size':2,'stride':2,'padding':0,'output_padding':0,'dilation':1,'group':1,'bias':False}
        elif(self.cell_info['mode']=='fc'):
            cell_info = {'cell':'Conv2d','normalization':self.cell_info['normalization'],'activation':self.cell_info['activation'],
                        'input_size':self.cell_info['input_size'],'output_size':self.cell_info['output_size'],
                        'kernel_size':1,'stride':1,'padding':0,'dilation':1,'group':1,'bias':False}
        else:
            raise ValueError('Sample mode not supported')
        for i in range(self.cell_info['num_layer']):
            if(i>0):
                cell_info = {**cell_info,'cell':'Conv2d','input_size':self.cell_info['output_size'],'kernel_size':3,'stride':1,'padding':1}
            if(i==self.cell_info['num_layer']-1 and self.cell_info['raw']):
                cell_info = {**cell_info,'normalization':'none','activation':'none'}
            cell.append(Cell(cell_info))
        return cell
        
    def forward(self, input):
        x = input
        if(x.dim()==4):
            for i in range(self.cell_info['num_layer']):
                x = self.cell[i](x)
        elif(x.dim()==5):
            for i in range(self.cell_info['num_layer']):
                x = apply_along_dim(x, fn=self.cell[i], dim=1) 
        return x

class ResCell(nn.Module):
    def __init__(self, cell_info):
        super(ResCell, self).__init__()
        self.cell_info = cell_info
        self.cell = self.make_cell()
        
    def make_cell(self):
        cell = nn.ModuleDict({})
        cell['in'] = nn.Sequential(*[Cell(self.cell_info['in'][i]) for i in range(len(self.cell_info['in']))])
        cell['shortcut'] = Cell(self.cell_info['shortcut'])
        cell['activation'] = Activation(self.cell_info['activation']) if(not self.cell_info['raw']) else Activation('none')
        if('normalization' in self.cell_info):
            cell['normalization'] = Normalization(self.cell_info['normalization'],self.cell_info['in'][-1]['output_size'])
        return cell
        
    def forward(self, input):
        x = input
        x = self.cell['in'](x)
        x += self.cell['shortcut'](input)
        x = self.cell['activation'](x)
        return x
        
class LSTMCell(nn.Module):
    def __init__(self, cell_info):
        super(LSTMCell, self).__init__()
        self.cell_info = cell_info
        self.cell = self.make_cell()
        self.hidden = None
        
    def make_cell(self):
        _tuple = _ntuple(2)
        self.cell_info['activation'] = _tuple(self.cell_info['activation'])
        cell = nn.ModuleList([nn.ModuleDict({}) for _ in range(self.cell_info['num_layer'])])
        for i in range(self.cell_info['num_layer']):
            cell_in_info = {**self.cell_info['in'][i],'output_size':4*self.cell_info['in'][i]['output_size']}
            cell_hidden_info = {**self.cell_info['hidden'][i],'output_size':4*self.cell_info['hidden'][i]['output_size']}
            cell[i]['in'] = Cell(cell_in_info)
            cell[i]['hidden'] = Cell(cell_hidden_info)
            cell[i]['activation'] = nn.ModuleList([Activation(self.cell_info['activation'][0]),Activation(self.cell_info['activation'][1])])
        return cell
        
    def init_hidden(self, hidden_size):
        hidden = [[torch.zeros(hidden_size,device=device)],[torch.zeros(hidden_size,device=device)]]
        return hidden
    
    def free_hidden(self):
        self.hidden = None
        return
        
    def forward(self, input, hidden=None):
        x = input
        x = x.unsqueeze(1) if(input.dim()==4) else x
        hx,cx = [None for _ in range(len(self.cell))],[None for _ in range(len(self.cell))]
        for i in range(len(self.cell)):
            y = [None for _ in range(x.size(1))]
            for j in range(x.size(1)):
                gates = self.cell[i]['in'](x[:,j])
                if(hidden is None):
                    if(self.hidden is None):
                        self.hidden = self.init_hidden((gates.size(0),self.cell_info['hidden'][i]['output_size'],*gates.size()[2:]))
                    else:
                        if(i==len(self.hidden[0])):
                            tmp_hidden = self.init_hidden((gates.size(0),self.cell_info['hidden'][i]['output_size'],*gates.size()[2:]))
                            self.hidden[0].extend(tmp_hidden[0])
                            self.hidden[1].extend(tmp_hidden[1])
                        else:
                            pass
                if(j==0):
                    hx[i],cx[i] = self.hidden[0][i],self.hidden[1][i]
                gates += self.cell[i]['hidden'](hx[i])
                ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)
                ingate = torch.sigmoid(ingate)
                forgetgate = torch.sigmoid(forgetgate)
                cellgate = self.cell[i]['activation'][0](cellgate)
                outgate = torch.sigmoid(outgate)
                cx[i] = (forgetgate * cx[i]) + (ingate * cellgate)
                hx[i] = outgate * self.cell[i]['activation'][1](cx[i])                
                y[j] = hx[i]
            x = torch.stack(y,dim=1)
        self.hidden = [hx,cx]
        x = x.squeeze(1) if(input.dim()==4) else x
        return x

class ResLSTMCell(nn.Module):
    def __init__(self, cell_info):
        super(ResLSTMCell, self).__init__()
        self.cell_info = cell_info
        self.cell = self.make_cell()
        self.hidden = None
        
    def make_cell(self):
        _tuple = _ntuple(2)
        self.cell_info['activation'] = _tuple(self.cell_info['activation'])
        cell = nn.ModuleList([nn.ModuleDict({}) for _ in range(self.cell_info['num_layer'])])
        for i in range(self.cell_info['num_layer']):
            if(i==0):
                cell_shortcut_info = self.cell_info['shortcut'][i]
                cell[i]['shortcut'] = Cell(cell_shortcut_info)
            cell_in_info = {**self.cell_info['in'][i],'output_size':4*self.cell_info['in'][i]['output_size']}
            cell_hidden_info = {**self.cell_info['hidden'][i],'output_size':4*self.cell_info['hidden'][i]['output_size']}
            cell[i]['in'] = Cell(cell_in_info)
            cell[i]['hidden'] = Cell(cell_hidden_info)            
            cell[i]['activation'] = nn.ModuleList([Activation(self.cell_info['activation'][0]),Activation(self.cell_info['activation'][1])])         
        return cell
        
    def init_hidden(self, hidden_size):
        hidden = [[torch.zeros(hidden_size,device=device)],[torch.zeros(hidden_size,device=device)]]
        return hidden
    
    def free_hidden(self):
        self.hidden = None
        return

    def forward(self, input, hidden=None):
        x = input
        x = x.unsqueeze(1) if(input.dim()==4) else x
        hx,cx = [None for _ in range(len(self.cell))],[None for _ in range(len(self.cell))]
        shortcut = [None for _ in range(x.size(1))]
        for i in range(len(self.cell)):
            y = [None for _ in range(x.size(1))]
            for j in range(x.size(1)):
                if(i==0):
                    shortcut[j] = self.cell[i]['shortcut'](x[:,j])
                gates = self.cell[i]['in'](x[:,j])
                if(hidden is None):
                    if(self.hidden is None):
                        self.hidden = self.init_hidden((gates.size(0),self.cell_info['hidden'][i]['output_size'],*gates.size()[2:]))
                    else:
                        if(i==len(self.hidden[0])):
                            tmp_hidden = self.init_hidden((gates.size(0),self.cell_info['hidden'][i]['output_size'],*gates.size()[2:]))
                            self.hidden[0].extend(tmp_hidden[0])
                            self.hidden[1].extend(tmp_hidden[1])
                        else:
                            pass
                else:
                    self.hidden = hidden
                if(j==0):
                    hx[i],cx[i] = self.hidden[0][i],self.hidden[1][i]
                gates += self.cell[i]['hidden'](hx[i])
                ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)
                ingate = torch.sigmoid(ingate)
                forgetgate = torch.sigmoid(forgetgate)
                cellgate = self.cell[i]['activation'][0](cellgate)
                outgate = torch.sigmoid(outgate)
                cx[i] = (forgetgate * cx[i]) + (ingate * cellgate)  
                hx[i] = outgate * self.cell[i]['activation'][1](cx[i]) if(i<len(self.cell)-1) else outgate * (shortcut[j] + self.cell[i]['activation'][1](cx[i]))
                y[j] = hx[i]
            x = torch.stack(y,dim=1)
        self.hidden = [hx,cx]
        x = x.squeeze(1) if(input.dim()==4) else x
        return x
        
class Cell(nn.Module):
    def __init__(self, cell_info):
        super(Cell, self).__init__()
        self.cell_info = cell_info
        self.cell = self.make_cell()
        
    def make_cell(self):
        if(self.cell_info['cell'] == 'none'):
            cell = nn.Sequential()
        elif(self.cell_info['cell'] == 'Conv2d'):
            default_cell_info = {'kernel_size':3,'stride':1,'padding':1,'dilation':1,'group':1,'bias':False,'normalization':'none','activation':'relu'}
            self.cell_info = {**default_cell_info,**self.cell_info}
            module = nn.Conv2d(self.cell_info['input_size'],self.cell_info['output_size'],self.cell_info['kernel_size'],\
                self.cell_info['stride'],self.cell_info['padding'],self.cell_info['dilation'],self.cell_info['group'],self.cell_info['bias'])
            normalization = Normalization(self.cell_info['normalization'],self.cell_info['output_size'])
            activation = Activation(self.cell_info['activation'])
            cell = nn.Sequential(OrderedDict([
                                  ('module', module),
                                  ('normalization', normalization),
                                  ('activation', activation),
                                ]))
        elif(self.cell_info['cell'] == 'ConvTranspose2d'):
            default_cell_info = {'kernel_size':3,'stride':1,'padding':1,'output_padding':0,'dilation':1,'group':1,'bias':False,'normalization':'none','activation':'relu'}
            self.cell_info = {**default_cell_info,**self.cell_info}
            module = nn.ConvTranspose2d(self.cell_info['input_size'],self.cell_info['output_size'],self.cell_info['kernel_size'],\
                self.cell_info['stride'],self.cell_info['padding'],self.cell_info['output_padding'],self.cell_info['group'],self.cell_info['bias'],self.cell_info['dilation'])
            normalization = Normalization(self.cell_info['normalization'],self.cell_info['output_size'])
            activation = Activation(self.cell_info['activation'])
            cell = nn.Sequential(OrderedDict([
                                  ('module', module),
                                  ('normalization', normalization),
                                  ('activation', activation),
                                ]))
        elif(self.cell_info['cell'] == 'BasicCell'):
            default_cell_info = {'mode':'pass','normalization':'none','activation':'relu','raw':False}
            self.cell_info = {**default_cell_info,**self.cell_info}
            cell = BasicCell(self.cell_info)
        elif(self.cell_info['cell'] == 'ResCell'):
            default_cell_info = {'normalization':'none','activation':'relu','raw':False}
            self.cell_info = {**default_cell_info,**self.cell_info}
            cell = ResCell(self.cell_info)
        elif(self.cell_info['cell'] == 'LSTMCell'):
            default_cell_info = {'activation':'tanh'}
            self.cell_info = {**default_cell_info,**self.cell_info}
            cell = LSTMCell(self.cell_info)
        elif(self.cell_info['cell'] == 'ResLSTMCell'):
            default_cell_info = {'activation':'tanh'}
            self.cell_info = {**default_cell_info,**self.cell_info}
            cell = ResLSTMCell(self.cell_info)
        else:
            raise ValueError('parse model mode not supported')
        return cell
        
    def forward(self, *input):
        x = self.cell(*input)
        return x
