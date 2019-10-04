import config
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import makedir_exist_ok, apply_fn
from .utils import make_model


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.model = make_model(config.PARAM['model'])
        makedir_exist_ok('./output/tmp')

    def pack(self, code):
        code = np.packbits(code.detach().cpu().numpy().astype(np.int8)).reshape(-1, 1)
        return code

    def unpack(self, code):
        code = torch.from_numpy(code.astype(np.float32)).to(config.PARAM['device'])
        return code

    def loss_fn(self, output, target):
        if config.PARAM['loss_fn'] == 'bce':
            loss_fn = F.binary_cross_entropy
        elif config.PARAM['loss_fn'] == 'mse':
            loss_fn = F.mse_loss
        elif config.PARAM['loss_fn'] == 'mae':
            loss_fn = F.l1_loss
        else:
            raise ValueError('Not valid loss function')
        loss = loss_fn(output, target)
        return loss

    def forward(self, input):
        output = {'loss': torch.tensor(0, device=config.PARAM['device'], dtype=torch.float32), 'img': [], 'code': []}
        x = input['img'] * 2 - 1
        reconstructed = x.new_zeros(x.size())
        for i in range(config.PARAM['num_iter']):
            encoded = self.model['encoder'](x)
            code = self.model['quantizer'](encoded)
            output['code'] += [self.pack(code)] if i == 0 else [
                np.concatenate((output['code'][-1], self.pack(code)), axis=1)]
            decoded = self.model['decoder'](code)
            decoded = (decoded + 1) / 2 if i == 0 else decoded
            reconstructed = reconstructed + decoded
            output['loss'] = output['loss'] + self.loss_fn(reconstructed, input['img'])
            x = input['img'] - reconstructed.detach()
            output['img'].append(reconstructed.detach())
        output['loss'] = output['loss'] / config.PARAM['num_iter']
        apply_fn(self, 'free_hidden')
        return output


def codec():
    config.PARAM['model'] = {}
    config.PARAM['model']['encoder'] = (
        {'cell': 'ConvCell', 'input_size': config.PARAM['num_channel'], 'output_size': 32, 'kernel_size': 1,
         'stride': 1, 'padding': 0, 'bias': True, 'activation': config.PARAM['activation'],
         'normalization': config.PARAM['normalization']},
        {'cell': 'PixelShuffleCell', 'mode': 'down', 'scale_factor': 2},
        {'cell': 'ConvLSTMCell', 'input_size': 128, 'output_size': 128, 'kernel_size': 3, 'stride': 1,
         'padding': 1, 'bias': True, 'num_layers': 1, 'activation': config.PARAM['activation']},
        {'cell': 'PixelShuffleCell', 'mode': 'down', 'scale_factor': 2},
        {'cell': 'ConvLSTMCell', 'input_size': 512, 'output_size': 128, 'kernel_size': 3, 'stride': 1,
         'padding': 1, 'bias': True, 'num_layers': 1, 'activation': config.PARAM['activation']},
        {'cell': 'PixelShuffleCell', 'mode': 'down', 'scale_factor': 2},
        {'cell': 'ConvLSTMCell', 'input_size': 512, 'output_size': 128, 'kernel_size': 3, 'stride': 1,
         'padding': 1, 'bias': True, 'num_layers': 1, 'activation': config.PARAM['activation']},
        {'cell': 'ConvCell', 'input_size': 128, 'output_size': config.PARAM['code_size'], 'kernel_size': 1,
         'stride': 1, 'padding': 0, 'bias': True, 'activation': config.PARAM['activation'],
         'normalization': config.PARAM['normalization']}
    )
    config.PARAM['model']['quantizer'] = {'cell': 'QuantizationCell'}
    config.PARAM['model']['decoder'] = (
        {'cell': 'ConvCell', 'input_size': config.PARAM['code_size'], 'output_size': 128, 'kernel_size': 1,
         'stride': 1, 'padding': 0, 'bias': True, 'activation': config.PARAM['activation'],
         'normalization': config.PARAM['normalization']},
        {'cell': 'ConvLSTMCell', 'input_size': 128, 'output_size': 512, 'kernel_size': 3, 'stride': 1,
         'padding': 1, 'bias': True, 'num_layers': 1, 'activation': config.PARAM['activation']},
        {'cell': 'PixelShuffleCell', 'mode': 'up', 'scale_factor': 2},
        {'cell': 'ConvLSTMCell', 'input_size': 128, 'output_size': 512, 'kernel_size': 3, 'stride': 1,
         'padding': 1, 'bias': True, 'num_layers': 1, 'activation': config.PARAM['activation']},
        {'cell': 'PixelShuffleCell', 'mode': 'up', 'scale_factor': 2},
        {'cell': 'ConvLSTMCell', 'input_size': 128, 'output_size': 128, 'kernel_size': 3, 'stride': 1,
         'padding': 1, 'bias': True, 'num_layers': 1, 'activation': config.PARAM['activation']},
        {'cell': 'PixelShuffleCell', 'mode': 'up', 'scale_factor': 2},
        {'cell': 'ConvCell', 'input_size': 32, 'output_size': config.PARAM['num_channel'], 'kernel_size': 1,
         'stride': 1, 'padding': 0, 'bias': True, 'activation': config.PARAM['activation'],
         'normalization': config.PARAM['normalization']}
    )
    model = Model()
    return model