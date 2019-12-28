import collections.abc as container_abcs
import config
import errno
import numpy as np
import os
import torch
from itertools import repeat
from torchvision.utils import save_image


def check_exists(path):
    return os.path.exists(path)


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
    config.PARAM['normalization'] = config.PARAM['control']['normalization']
    config.PARAM['activation'] = config.PARAM['control']['activation']
    config.PARAM['num_iter'] = int(config.PARAM['control']['num_iter'])
    config.PARAM['code_size'] = int(config.PARAM['control']['code_size'])
    config.PARAM['num_node'] = int(config.PARAM['control']['num_node'])
    config.PARAM['num_channel'] = 1 if config.PARAM['data_name'] == 'MNIST' else 3
    return

def make_stats(dataset):
    if os.path.exists('./data/stats/{}.pt'.format(dataset.data_name)):
        stats = load('./data/stats/{}.pt'.format(dataset.data_name))
    elif dataset is not None:
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=100, shuffle=False, num_workers=0)
        stats = Stats(dim=1)
        with torch.no_grad():
            for input in data_loader:
                stats.update(input['img'])
        save(stats, './data/stats/{}.pt'.format(dataset.data_name))
    return stats


class Stats(object):
    def __init__(self, dim):
        self.dim = dim
        self.n_samples = 0
        self.n_features = None
        self.mean = None
        self.std = None

    def update(self, data):
        data = data.transpose(self.dim, -1).reshape(-1, data.size(self.dim))
        if self.n_samples == 0:
            self.n_samples = data.size(0)
            self.n_features = data.size(1)
            self.mean = data.mean(dim=0)
            self.std = data.std(dim=0)
        else:
            m = float(self.n_samples)
            n = data.size(0)
            new_mean = data.mean(dim=0)
            new_std = 0 if n == 1 else data.std(dim=0)
            old_mean = self.mean
            old_std = self.std
            self.mean = m / (m + n) * old_mean + n / (m + n) * new_mean
            self.std = torch.sqrt(m / (m + n) * old_std ** 2 + n / (m + n) * new_std ** 2 + m * n / (m + n) ** 2 * (
                    old_mean - new_mean) ** 2)
            self.n_samples += n
        return


def process_dataset(dataset):
    config.PARAM['classes_size'] = dataset.classes_size
    return


def resume(model, model_tag, optimizer=None, scheduler=None, load_tag='checkpoint'):
    if os.path.exists('./output/model/{}_{}.pt'.format(model_tag, load_tag)):
        checkpoint = load('./output/model/{}_{}.pt'.format(model_tag, load_tag))
        last_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['model_dict'])
        if optimizer is not None:
            optimizer.load_state_dict(checkpoint['optimizer_dict'])
        if scheduler is not None:
            scheduler.load_state_dict(checkpoint['scheduler_dict'])
        logger = checkpoint['logger']
        print('Resume from {}'.format(last_epoch))
        return last_epoch, model, optimizer, scheduler, logger
    else:
        raise ValueError('Not exists model tag')
    return


def collate(input):
    for k in input:
        input[k] = torch.stack(input[k], 0)
    return input
