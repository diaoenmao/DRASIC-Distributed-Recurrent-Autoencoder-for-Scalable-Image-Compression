import collections.abc as container_abcs
import config
import errno
import numpy as np
import os
import torch
from itertools import repeat
from torchvision.utils import save_image


def makedir_exist_ok(dirpath):
    try:
        os.makedirs(dirpath)
    except OSError as e:
        if e.errno == errno.EEXIST:
            pass
        else:
            raise
    return


def save(input, path, protocol=2, mode='torch'):
    dirname = os.path.dirname(path)
    makedir_exist_ok(dirname)
    if mode == 'torch':
        torch.save(input, path, pickle_protocol=protocol)
    elif mode == 'numpy':
        np.save(path, input)
    else:
        raise ValueError('Not valid save mode')
    return


def load(path, mode='torch'):
    if mode == 'torch':
        return torch.load(path, map_location=lambda storage, loc: storage)
    elif mode == 'numpy':
        return np.load(path)
    else:
        raise ValueError('Not valid save mode')
    return


def save_img(img, path):
    makedir_exist_ok(os.path.dirname(path))
    save_image(img, path, padding=0)
    return


def to_device(input, device):
    output = recur(lambda x, y: x.to(y), input, device)
    return output


def ntuple(n):
    def parse(x):
        if isinstance(x, container_abcs.Iterable) and not isinstance(x, str):
            return x
        return tuple(repeat(x, n))

    return parse


def apply_fn(module, fn):
    for n, m in module.named_children():
        if hasattr(m, fn):
            exec('m.{0}()'.format(fn))
        if sum(1 for _ in m.named_children()) != 0:
            exec('apply_fn(m,\'{0}\')'.format(fn))
    return


def normalize(input):
    with torch.no_grad():
        broadcast_size = [1] * input.dim()
        broadcast_size[1] = input.size(1)
        m, s = config.PARAM['stats']['img'].mean.view(broadcast_size).to(input.device), config.PARAM['stats'][
            'img'].std.view(broadcast_size).to(input.device)
        input = input.sub(m).div(s).detach()
    return input


def recur(fn, input, *args):
    if isinstance(input, torch.Tensor) or isinstance(input, np.ndarray):
        output = fn(input, *args)
    elif isinstance(input, list):
        output = []
        for i in range(len(input)):
            output.append(recur(fn, input[i], *args))
    elif isinstance(input, dict):
        output = {}
        for key in input:
            output[key] = recur(fn, input[key], *args)
    else:
        raise ValueError('Not valid input type')
    return output


def denormalize(input):
    with torch.no_grad():
        broadcast_size = [1] * input.dim()
        broadcast_size[1] = input.size(1)
        m, s = config.PARAM['stats']['img'].mean.view(broadcast_size).to(input.device), \
               config.PARAM['stats']['img'].std.view(broadcast_size).to(input.device)
        input = input.mul(s).add(m).detach()
    return input


def process_control_name():
    control_name = config.PARAM['control_name'].split('_')
    config.PARAM['num_iter'] = int(control_name[0])
    config.PARAM['code_size'] = int(control_name[1])
    config.PARAM['num_channel'] = 1 if config.PARAM['data_name']['train'] == 'MNIST' else 3
    print(config.PARAM)
    return


def process_evaluation(evaluation):
    processed_evaluation = {}
    for k in evaluation:
        if isinstance(evaluation[k], list):
            processed_evaluation[k] = evaluation[k][-1]
        else:
            processed_evaluation[k] = evaluation[k]
    return processed_evaluation