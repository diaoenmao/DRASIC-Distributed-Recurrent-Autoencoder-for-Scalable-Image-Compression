import torch.nn as nn
import torch.nn.functional as F
import torch
import config

config.init()
device = config.PARAM['device']

class Conv2dLSTMCell(nn.Module):
    def __init__(self, input_size, output_size, kernel_size=3, hidden_kernel_size=3, stride=1, hidden_stride=1, padding=1, hidden_padding=1, dilation=1, bias=True):
        super(Conv2dLSTMCell, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.kernel_size = kernel_size
        self.hidden_kernel_size = hidden_kernel_size
        self.stride = stride
        self.hidden_stride = hidden_stride
        self.padding = padding
        self.hidden_padding = hidden_padding
        self.dilation = dilation
        self.bias = bias
        self.conv_ih = nn.Conv2d(self.input_size,4*self.output_size,self.kernel_size,self.stride,self.padding,self.dilation,1,self.bias)
        self.conv_hh = nn.Conv2d(self.output_size,4*self.output_size,self.hidden_kernel_size,self.hidden_stride,self.hidden_padding,self.dilation,1,self.bias)
        self.reset_parameters()
        
    def reset_parameters(self):
        self.conv_ih.reset_parameters()
        self.conv_hh.reset_parameters()
        return
    
    def init_hidden(self, input_size):
        hidden_size = [input_size[0],-1,input_size[3]//self.stride,input_size[4]//self.stride]
        hidden_size[1] = self.output_size
        hidden = (torch.zeros(hidden_size,device=device),torch.zeros(hidden_size,device=device))
        return hidden
        
    def forward(self, input, hidden=None):
        if(hidden is None):
            hidden = self.init_hidden(input.size())
        hx, cx = hidden
        gates = self.conv_ih(input) + self.conv_hh(hx)
        ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)
        ingate = torch.sigmoid(ingate)
        forgetgate = torch.sigmoid(forgetgate)
        cellgate = torch.tanh(cellgate)
        outgate = torch.sigmoid(outgate)
        cy = (forgetgate * cx) + (ingate * cellgate)
        hy = outgate * torch.tanh(cy)
        return hy, cy

class Conv2dLSTM(nn.Module):
    def __init__(self, input_size, output_size, num_layers, kernel_size=3, hidden_kernel_size=3, stride=1, hidden_stride=1, padding=1, hidden_padding=1, dilation=1, bias=True):
        super(Conv2dLSTM, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.kernel_size = kernel_size
        self.hidden_kernel_size = hidden_kernel_size
        self.stride = stride
        self.hidden_stride = hidden_stride
        self.padding = padding
        self.hidden_padding = hidden_padding
        self.dilation = dilation
        self.bias = bias
        self.convlstm = self.make_convlstm(input_size,output_size,num_layers,kernel_size,hidden_kernel_size,stride,hidden_stride,padding,hidden_padding,dilation,bias)
   
    def make_convlstm(self, input_size, output_size, num_layers, kernel_size, hidden_kernel_size, stride, hidden_stride, padding, hidden_padding, dilation, bias):
        convlstm = nn.ModuleList([])
        for i in range(num_layers):
            convlstm.append(Conv2dLSTMCell(input_size,output_size,kernel_size,hidden_kernel_size,stride,hidden_stride,padding,hidden_padding,dilation,bias))
            input_size = output_size
        return convlstm

    def init_hidden(self, input_size):
        hidden_size = [input_size[0],-1,input_size[3]//self.stride,input_size[4]//self.stride]
        hidden_size[1] = self.output_size
        hidden = []
        for i in range(self.num_layers):
            hidden.append((torch.zeros(hidden_size,device=device),torch.zeros(hidden_size,device=device)))
        return hidden
        
    def forward(self, input, hidden=None):
        output = []
        if(hidden is None):
            hidden = self.init_hidden(input.size())
        for i in range(len(self.convlstm)):
            for j in range(input.size(1)):
                hidden[i] = self.convlstm[i](input[:,j],hidden[i])
                if(i == len(self.convlstm)-1):     
                    output.append(hidden[i][0])
        output = torch.stack(output,1)
        return output,hidden

class Conv2dGRUCell(nn.Module):
    def __init__(self, input_size, output_size, kernel_size=3, hidden_kernel_size=3, stride=1, hidden_stride=1, padding=1, hidden_padding=1, dilation=1, bias=True):
        super(Conv2dGRUCell, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.kernel_size = kernel_size
        self.hidden_kernel_size = hidden_kernel_size
        self.stride = stride
        self.hidden_stride = hidden_stride
        self.padding = padding
        self.hidden_padding = hidden_padding
        self.dilation = dilation
        self.bias = bias
        self.conv_ih = nn.Conv2d(self.input_size,3*self.output_size,self.kernel_size,self.stride,self.padding,self.dilation,1,self.bias)
        self.conv_hh = nn.Conv2d(self.output_size,3*self.output_size,self.hidden_kernel_size,self.hidden_stride,self.hidden_padding,self.dilation,1,self.bias)
        self.reset_parameters()
        
    def reset_parameters(self):
        self.conv_ih.reset_parameters()
        self.conv_hh.reset_parameters()
        return
        
    def init_hidden(self, input_size):
        hidden_size = [input_size[0],-1,input_size[3]//self.stride,input_size[4]//self.stride]
        hidden_size[1] = self.output_size
        hidden = torch.zeros(hidden_size,device=device)
        return hidden
    
    def forward(self, input, hidden=None):
        if(hidden is None):
            hidden = self.init_hidden(input.size())
        hx = hidden
        ih, hh = self.conv_ih(input).chunk(3, 1), self.conv_hh(hx).chunk(3, 1)
        forgetgate = torch.sigmoid(ih[0]+hh[0])
        outgate = torch.sigmoid(ih[0]+hh[1])
        ingate = torch.tanh(ih[2] + forgetgate*hh[2])
        hy = (1-outgate) * ingate + outgate * hx
        return hy

class Conv2dGRU(nn.Module):
    def __init__(self, input_size, output_size, num_layers, kernel_size=3, hidden_kernel_size=3, stride=1, hidden_stride=1, padding=1, hidden_padding=1, dilation=1, bias=True):
        super(Conv2dGRU, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.kernel_size = kernel_size
        self.hidden_kernel_size = hidden_kernel_size
        self.stride = stride
        self.hidden_stride = hidden_stride
        self.padding = padding
        self.hidden_padding = hidden_padding
        self.dilation = dilation
        self.bias = bias
        self.convgru = self.make_convgru(input_size,output_size,num_layers,kernel_size,hidden_kernel_size,stride,hidden_stride,padding,hidden_padding,dilation,bias)
   
    def make_convgru(self, input_size, output_size, num_layers, kernel_size, hidden_kernel_size, stride, hidden_stride, padding, hidden_padding, dilation, bias):
        convgru = nn.ModuleList([])
        for i in range(num_layers):
            convgru.append(Conv2dGRUCell(input_size,output_size,kernel_size,hidden_kernel_size,stride,hidden_stride,padding,hidden_padding,dilation,bias))
            input_size = output_size
        return convgru
    
    def init_hidden(self, input_size):
        hidden_size = [input_size[0],-1,input_size[3]//self.stride,input_size[4]//self.stride]
        hidden_size[1] = self.output_size
        hidden = []
        for i in range(self.num_layers):
            hidden.append(torch.zeros(hidden_size,device=device))
        return hidden
        
    def forward(self, input, hidden=None):
        output = []
        if(hidden is None):
            hidden = self.init_hidden(input.size())
        for i in range(len(self.convgru)):
            for j in range(input.size(1)):
                hidden[i] = self.convgru[i](input[:,j],hidden[i])
                if(i == len(self.convgru)-1):     
                    output.append(hidden[i])
        output = torch.stack(output,1)
        return output,hidden
