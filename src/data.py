import config
import numpy as np
import os
import torch
import datasets
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
from utils import save, load


def fetch_dataset(data_name):
    dataset = {}
    print('fetching data {}...'.format(data_name))
    data_name_head = data_name.split('_')[0]
    root = './data/{}'.format(data_name_head)
    if data_name_head in ['MNIST', 'FashionMNIST', 'SVHN']:
        dataset['train'] = eval('datasets.{}(root=root, split=\'train\', download=True, transform=datasets.Compose(['
                                'transforms.ToTensor()]))'.format(data_name_head))
        dataset['test'] = eval('datasets.{}(root=root, split=\'test\', download=True, transform=datasets.Compose(['
                               'transforms.ToTensor()]))'.format(data_name_head))
        config.PARAM['stats'] = make_stats(dataset['train'], batch_size=config.PARAM['batch_size']['test'])
        config.PARAM['transform'] = {
            'train': datasets.Compose([transforms.Resize((32, 32)), transforms.ToTensor()]),
            'test': datasets.Compose([transforms.Resize((32, 32)), transforms.ToTensor()])
        }
    elif data_name_head == 'EMNIST':
        subset = data_name.split('_')[1]
        dataset['train'] = datasets.EMNIST(root=root, split='train_{}'.format(subset), download=True,
                                           transform=datasets.Compose([transforms.ToTensor()]))
        dataset['test'] = datasets.EMNIST(root=root, split='test_{}'.format(subset), download=True,
                                          transform=datasets.Compose([transforms.ToTensor()]))
        config.PARAM['stats'] = make_stats(dataset['train'], batch_size=config.PARAM['batch_size']['test'])
        config.PARAM['transform'] = {
            'train': datasets.Compose([transforms.Resize((32, 32)), transforms.ToTensor()]),
            'test': datasets.Compose([transforms.Resize((32, 32)), transforms.ToTensor()])
        }
    elif data_name_head in ['CIFAR10', 'CIFAR100']:
        dataset['train'] = eval('datasets.{}(root=root, split=\'train\', download=True, transform=datasets.Compose(['
                                'transforms.ToTensor()]))'.format(data_name_head))
        dataset['test'] = eval('datasets.{}(root=root, split=\'test\', download=True, transform=datasets.Compose(['
                               'transforms.ToTensor()]))'.format(data_name_head))
        config.PARAM['stats'] = make_stats(dataset['train'], batch_size=config.PARAM['batch_size']['test'])
        config.PARAM['transform'] = {
            'train': datasets.Compose([transforms.ToTensor()]),
            'test': datasets.Compose([transforms.ToTensor()])
        }
    elif data_name_head == 'ImageNet':
        dataset['train'] = datasets.ImageNet(root, split='train', download=True,
                                             transform=datasets.Compose([transforms.ToTensor()]))
        dataset['test'] = datasets.ImageNet(root, split='test', download=True,
                                            transform=datasets.Compose([transforms.ToTensor()]))
        config.PARAM['stats'] = make_stats(dataset['train'], batch_size=config.PARAM['batch_size']['test'])
        config.PARAM['transform'] = {
            'train': datasets.Compose([transforms.Resize((224, 224)), transforms.ToTensor()]),
            'test': datasets.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
        }
    elif data_name_head == 'Kodak':
        dataset['train'] = datasets.ImageFolder(root, transform=datasets.Compose([transforms.ToTensor()]))
        dataset['test'] = datasets.ImageFolder(root, transform=datasets.Compose([transforms.ToTensor()]))
        config.PARAM['stats'] = make_stats(dataset['train'], batch_size=config.PARAM['batch_size']['test'])
        config.PARAM['transform'] = {
            'train': datasets.Compose([transforms.ToTensor()]), 'test': datasets.Compose([transforms.ToTensor()])
        }
    else:
        raise ValueError('Not valid dataset name')
    dataset['train'].transform = config.PARAM['transform']['train']
    dataset['test'].transform = config.PARAM['transform']['test']
    print('data ready')
    return dataset


def input_collate(batch):
    if isinstance(batch[0], dict):
        output = {key: [] for key in batch[0].keys()}
        for b in batch:
            for key in b:
                output[key].append(b[key])
        return output
    else:
        return default_collate(batch)


def split_dataset(dataset, data_size, batch_size, radomGen=np.random.RandomState(1234), collate_fn=input_collate):
    data_loader = {}
    for k in dataset:
        if k in data_size:
            data_size[k] = len(dataset[k]) if (data_size[k] > len(dataset[k])) else data_size[k]
            data_size[k] = len(dataset[k]) if (data_size[k] == 0) else data_size[k]
            batch_size[k] = data_size[k] if (batch_size[k] == 0) else batch_size[k]
            data_idx_k = radomGen.choice(list(range(len(dataset[k]))), size=data_size[k], replace=False)
            dataset_k = torch.utils.data.Subset(dataset[k], data_idx_k)
            data_loader[k] = torch.utils.data.DataLoader(dataset=dataset_k, shuffle=config.PARAM['shuffle'][k],
                                                         batch_size=batch_size[k], pin_memory=True,
                                                         num_workers=config.PARAM['num_workers'], collate_fn=collate_fn)
    return data_loader


def make_stats(dataset, batch_size, reuse=True):
    if reuse and os.path.exists('./data/stats/{}.pkl'.format(dataset.data_name)):
        stats = load('./data/stats/{}.pkl'.format(dataset.data_name))
    elif dataset is not None:
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
        stats = {}
        for k in dataset.feature_dim:
            stats[k] = Stats(dataset.feature_dim[k])
        print('Computing mean and std...')
        with torch.no_grad():
            for input in data_loader:
                for k in dataset.feature_dim:
                    stats[k].update(input[k])
        save(stats, './data/stats/{}.pkl'.format(dataset.data_name))
    else:
        raise ValueError('Not valid dataset')
    for k in dataset.feature_dim:
        print('[{}] mean: {}, std: {}'.format(k, stats[k].mean, stats[k].std))
    return stats


class Stats(object):
    def __init__(self, feature_dim):
        self.feature_dim = feature_dim
        self.n_samples = 0
        self.n_features = None
        self.mean = None
        self.std = None

    def update(self, data):
        collapse_data = data.transpose(self.feature_dim, -1).reshape(-1, data.size(self.feature_dim))
        if self.n_samples == 0:
            self.n_samples = collapse_data.size(0)
            self.n_features = collapse_data.size(1)
            self.mean = collapse_data.mean(dim=0)
            self.std = collapse_data.std(dim=0)
        else:
            if collapse_data.size(1) != self.n_features:
                raise ValueError("Not valid feature dimension")
            m = float(self.n_samples)
            n = collapse_data.size(0)
            new_mean = collapse_data.mean(dim=0)
            new_std = new_mean.new_zeros(new_mean.size()) if (n == 1) else collapse_data.std(dim=0)
            old_mean = self.mean
            old_std = self.std
            self.mean = m / (m + n) * old_mean + n / (m + n) * new_mean
            self.std = torch.sqrt(m / (m + n) * old_std ** 2 + n / (m + n) * new_std ** 2 + m * n / (m + n) ** 2 * (
                    old_mean - new_mean) ** 2)
            self.n_samples += n