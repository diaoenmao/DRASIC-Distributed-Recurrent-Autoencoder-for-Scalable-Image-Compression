import config
import copy
import torch
import torch.nn as nn
from modules.shuffle import PixelShuffle, PixelUnShuffle
from modules.quantization import Quantization


def Normalization(cell_info):
    if cell_info['mode'] == 'none':
        return nn.Sequential()
    elif cell_info['mode'] == 'bn':
        return nn.BatchNorm1d(cell_info['input_size'])
    elif cell_info['mode'] == 'in':
        return nn.InstanceNorm1d(cell_info['input_size'])
    elif cell_info['mode'] == 'ln':
        return nn.LayerNorm(cell_info['input_size'])
    else:
        raise ValueError('Not valid normalization')
    return


def Activation(cell_info):
    if cell_info['mode'] == 'none':
        return nn.Sequential()
    elif cell_info['mode'] == 'tanh':
        return nn.Tanh()
    elif cell_info['mode'] == 'hardtanh':
        return nn.Hardtanh()
    elif cell_info['mode'] == 'relu':
        return nn.ReLU(inplace=True)
    elif cell_info['mode'] == 'prelu':
        return nn.PReLU()
    elif cell_info['mode'] == 'elu':
        return nn.ELU(inplace=True)
    elif cell_info['mode'] == 'selu':
        return nn.SELU(inplace=True)
    elif cell_info['mode'] == 'celu':
        return nn.CELU(inplace=True)
    elif cell_info['mode'] == 'sigmoid':
        return nn.Sigmoid()
    elif cell_info['mode'] == 'softmax':
        return nn.SoftMax()
    else:
        raise ValueError('Not valid activation')
    return


class ConvCell(nn.Module):
    def __init__(self, cell_info):
        super(ConvCell, self).__init__()
        self.cell_info = cell_info
        self.cell = self.make_cell()

    def make_cell(self):
        cell_info = copy.deepcopy(self.cell_info)
        cell = nn.ModuleDict({})
        cell_main_info = "nn.Conv2d(cell_info['input_size'], cell_info['output_size'], " \
                         "kernel_size=cell_info['kernel_size'], stride=cell_info['stride'], " \
                         "padding=cell_info['padding'], bias=cell_info['bias'])"
        cell_normalization_info = {'cell': 'Normalization', 'input_size': cell_info['output_size'],
                                   'mode': cell_info['normalization']}
        cell_activation_info = {'cell': 'Activation', 'mode': cell_info['activation']}
        cell['main'] = eval(cell_main_info)
        cell['activation'] = Cell(cell_normalization_info)
        cell['normalization'] = Cell(cell_activation_info)
        return cell

    def forward(self, input):
        x = input
        x = self.cell['activation'](self.cell['normalization'](self.cell['main'](x)))
        return x


class ConvLSTMCell(nn.Module):
    def __init__(self, cell_info):
        super(ConvLSTMCell, self).__init__()
        self.cell_info = cell_info
        self.cell = self.make_cell()
        self.hidden = None

    def make_cell(self):
        cell_info = copy.deepcopy(self.cell_info)
        cell = nn.ModuleList([nn.ModuleDict({}) for _ in range(cell_info['num_layers'])])
        for i in range(cell_info['num_layers']):
            cell_in_info = {'cell': 'ConvCell', 'input_size': cell_info['input_size'],
                            'output_size': 4 * cell_info['output_size'],
                            'kernel_size': cell_info['kernel_size'], 'stride': cell_info['stride'],
                            'padding': cell_info['padding'], 'bias': cell_info['bias'], 'normalization': 'none',
                            'activation': 'none'}
            cell_hidden_info = {'cell': 'ConvCell', 'input_size': cell_info['output_size'],
                                'output_size': 4 * cell_info['output_size'],
                                'kernel_size': cell_info['kernel_size'], 'stride': cell_info['stride'],
                                'padding': cell_info['padding'], 'bias': cell_info['bias'], 'normalization': 'none',
                                'activation': 'none'}
            cell_activation_info = {'cell': 'Activation', 'mode': cell_info['activation']}
            cell[i]['in'] = Cell(cell_in_info)
            cell[i]['hidden'] = Cell(cell_hidden_info)
            cell[i]['activation'] = nn.ModuleList([Cell(cell_activation_info), Cell(cell_activation_info)])
        return cell

    def init_hidden(self, hidden_size):
        hidden = [[torch.zeros(hidden_size, device=config.PARAM['device'])], [torch.zeros(hidden_size, device=config.PARAM['device'])]]
        return hidden

    def free_hidden(self):
        self.hidden = None
        return

    def forward(self, input, hidden=None):
        x = input
        x = x.unsqueeze(1) if (input.dim() == 4) else x
        hx, cx = [None for _ in range(len(self.cell))], [None for _ in range(len(self.cell))]
        for i in range(len(self.cell)):
            y = [None for _ in range(x.size(1))]
            for j in range(x.size(1)):
                gates = self.cell[i]['in'](x[:, j])
                if hidden is None:
                    if self.hidden is None:
                        self.hidden = self.init_hidden(
                            (gates.size(0), self.cell_info['output_size'], *gates.size()[2:]))
                    else:
                        if i == len(self.hidden[0]):
                            new_hidden = self.init_hidden(
                                (gates.size(0), self.cell_info['output_size'], *gates.size()[2:]))
                            self.hidden[0].extend(new_hidden[0])
                            self.hidden[1].extend(new_hidden[1])
                        else:
                            pass
                if j == 0:
                    hx[i], cx[i] = self.hidden[0][i], self.hidden[1][i]
                gates += self.cell[i]['hidden'](hx[i])
                ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)
                ingate = torch.sigmoid(ingate)
                forgetgate = torch.sigmoid(forgetgate)
                cellgate = self.cell[i]['activation'][0](cellgate)
                outgate = torch.sigmoid(outgate)
                cx[i] = (forgetgate * cx[i]) + (ingate * cellgate)
                hx[i] = outgate * self.cell[i]['activation'][1](cx[i])
                y[j] = hx[i]
            x = torch.stack(y, dim=1)
        self.hidden = [hx, cx]
        x = x.squeeze(1) if (input.dim() == 4) else x
        return x


class PixelShuffleCell(nn.Module):
    def __init__(self, cell_info):
        super(PixelShuffleCell, self).__init__()
        self.cell_info = cell_info
        self.cell = self.make_cell()

    def make_cell(self):
        cell_info = copy.deepcopy(self.cell_info)
        if self.cell_info['mode'] == 'down':
            cell = PixelUnShuffle(cell_info['scale_factor'])
        elif self.cell_info['mode'] == 'up':
            cell = PixelShuffle(cell_info['scale_factor'])
        else:
            raise ValueError('Not valid model mode')
        return cell

    def forward(self, input):
        x = self.cell(input)
        return x


class QuantizationCell(nn.Module):
    def __init__(self, cell_info):
        super(QuantizationCell, self).__init__()
        self.cell_info = cell_info
        self.cell = self.make_cell()

    def make_cell(self):
        cell_info = copy.deepcopy(self.cell_info)
        cell = Quantization()
        return cell

    def forward(self, input):
        x = input
        x = self.cell(x)
        return x


class Cell(nn.Module):
    def __init__(self, cell_info):
        super(Cell, self).__init__()
        self.cell_info = cell_info
        self.cell = self.make_cell()

    def make_cell(self):
        if self.cell_info['cell'] == 'none':
            cell = nn.Sequential()
        elif self.cell_info['cell'] == 'Normalization':
            cell = Normalization(self.cell_info)
        elif self.cell_info['cell'] == 'Activation':
            cell = Activation(self.cell_info)
        elif self.cell_info['cell'] == 'ConvCell':
            cell = ConvCell(self.cell_info)
        elif self.cell_info['cell'] == 'ConvLSTMCell':
            cell = ConvLSTMCell(self.cell_info)
        elif self.cell_info['cell'] == 'PixelShuffleCell':
            cell = PixelShuffleCell(self.cell_info)
        elif self.cell_info['cell'] == 'QuantizationCell':
            cell = QuantizationCell(self.cell_info)
        else:
            raise ValueError('Not valid {} model'.format(self.cell_info['cell']))
        return cell

    def forward(self, *input):
        x = self.cell(*input)
        return x