import config
import numpy as np
import os

config.init()
import time
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image

from utils import save_img

device = config.PARAM['device']


class magick():
    def __init__(self):
        super(magick, self).__init__()
        self.supported_format = ['jpg', 'jp2', 'bpg', 'webp', 'png']

    def encode(self, input):
        if config.PARAM['format'] not in self.supported_format:
            raise ValueError('Not valid format')
        filename = 'tmp'
        for i in range(len(input)):
            save_img(input[i], './output/tmp/{filename}_{idx}.png'.format(filename=filename, idx=i))
        head = 'magick mogrify '
        tail = './output/tmp/*.png'
        quality = config.PARAM['quality']
        sampling_factor = None
        option = '-format {} -depth 8 '.format(format)
        if quality is not None:
            option += '-quality {quality} '.format(quality=quality)
        if sampling_factor is not None and (format in ['jpg', 'webp']):
            option += '-sampling-factor {sampling_factor} '.format(sampling_factor=sampling_factor)
        command = '{head}{option}{tail}'.format(head=head, option=option, tail=tail)
        os.system(command)
        code = []
        for i in range(len(input)):
            f = open('./output/tmp/{filename}_{idx}.{format}'.format(filename=filename, idx=i, format=format), 'rb')
            buffer = f.read()
            f.close()
            code.append(torch.from_numpy(np.frombuffer(buffer, dtype=np.uint8)))
        return code

    def decode(self, code):
        if config.PARAM['format'] not in self.supported_format:
            raise ValueError('Not valid format')
        filename = 'tmp'
        for i in range(len(code)):
            try:
                f = open('./output/tmp/{filename}_{idx}.{format}'.format(filename=filename, idx=i, format=format), 'wb')
                f.write(code[i])
                f.close()
            except OSError:
                time.sleep(0.1)
                f = open('./output/tmp/{filename}_{idx}.{format}'.format(filename=filename, idx=i, format=format), 'wb')
                f.write(code[i])
                f.close()
        command = 'magick mogrify -format png -depth 8 ./output/tmp/*.{format}'.format(format=format)
        os.system(command)
        output = []
        for i in range(len(code)):
            output.append(transforms.ToTensor()(
                Image.open('./output/tmp/{filename}_{idx}.png'.format(filename=filename, idx=i))).to(device))
        return output

    def loss_fn(self, input, output):
        if config.PARAM['loss_fn'] == 'bce':
            loss_fn = F.binary_cross_entropy
        elif config.PARAM['loss_fn'] == 'mse':
            loss_fn = F.mse_loss
        elif config.PARAM['loss_fn'] == 'mae':
            loss_fn = F.l1_loss
        else:
            raise ValueError('Not valid loss function')
        loss = loss_fn(output['img'], input['img'])
        return loss

    def forward(self, input):
        output = {}
        output['code'] = self.encode(input['img'])
        output['img'] = self.decode(output['code'])
        output['loss'] = self.loss_fn(input, output)
        return output