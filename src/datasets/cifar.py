import numpy as np
import os
import pickle
import torch
from PIL import Image
from torch.utils.data import Dataset
from utils import makedir_exist_ok, save, load
from .utils import download_url, extract_file, make_classes_counts


class CIFAR10(Dataset):
    data_name = 'CIFAR10'
    url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    md5 = 'c58f30108f718f92721af3b95e74349a'
    base_folder = 'cifar-10-batches-py'
    meta = {'filename': 'batches.meta', 'key': 'label_names'}
    train_list = ['data_batch_1', 'data_batch_2', 'data_batch_3', 'data_batch_4', 'data_batch_5']
    test_list = ['test_batch']
    feature_dim = {'img': 1}

    def __init__(self, root, split, transform=None, download=False):
        self.root = os.path.expanduser(root)
        self.split = split
        self.transform = transform
        if download:
            self.download()
        if not self._check_exists():
            raise RuntimeError('Dataset not found. You can use download=True to download it')
        self.img, self.label = load(os.path.join(self.processed_folder, '{}.pt'.format(split)))
        self.classes_to_labels = load(os.path.join(self.processed_folder, 'meta.pt'))
        self.classes = self.classes_to_labels.keys()
        self.classes_size = len(self.classes)
        self.classes_counts = make_classes_counts(self.label)

    def __getitem__(self, index):
        input = {'img': Image.fromarray(self.img[index]), 'label': torch.tensor(self.label[index])}
        if self.transform is not None:
            input = self.transform(input)
        return input

    def __len__(self):
        return len(self.img)

    @property
    def processed_folder(self):
        return os.path.join(self.root, 'processed')

    @property
    def raw_folder(self):
        return os.path.join(self.root, 'raw')

    def _check_exists(self):
        return os.path.exists(self.processed_folder)

    def download(self):
        if self._check_exists():
            return
        makedir_exist_ok(self.raw_folder)
        filename = os.path.basename(self.url)
        file_path = os.path.join(self.raw_folder, filename)
        extracted_file_path = os.path.join(self.raw_folder, self.base_folder)
        download_url(self.url, root=self.raw_folder, filename=filename, md5=self.md5)
        extract_file(file_path)
        train_set = read_pickle_file(extracted_file_path, self.train_list)
        test_set = read_pickle_file(extracted_file_path, self.test_list)
        save(train_set, os.path.join(self.processed_folder, 'train.pt'))
        save(test_set, os.path.join(self.processed_folder, 'test.pt'))
        classes_to_labels = parse_meta(os.path.join(self.raw_folder, self.base_folder, self.meta['filename']), self.meta['key'])
        save(classes_to_labels, os.path.join(self.processed_folder, 'meta.pt'))
        return

    def __repr__(self):
        fmt_str = 'Dataset {}\n'.format(self.__class__.__name__)
        fmt_str += '    Number of data points: {}\n'.format(self.__len__())
        fmt_str += '    Split: {}\n'.format(self.split)
        fmt_str += '    Root: {}\n'.format(self.root)
        fmt_str += '    Transforms: {}\n'.format(self.transform.__repr__())
        return fmt_str


class CIFAR100(CIFAR10):
    data_name = 'CIFAR100'
    url = "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
    md5 = 'eb9058c3a382ffc7106e4002c42a8d85'
    meta = {'filename': 'meta', 'key': 'fine_label_names'}
    base_folder = 'cifar-100-python'
    train_list = ['train']
    test_list = ['test']

    def __init__(self, root, split, **kwargs):
        super(CIFAR100, self).__init__(root, split, **kwargs)

    @property
    def processed_folder(self):
        return os.path.join(self.root, 'processed')

    def download(self):
        if self._check_exists():
            return
        filename = os.path.basename(self.url)
        file_path = os.path.join(self.raw_folder, filename)
        extracted_file_path = os.path.join(self.raw_folder, self.base_folder)
        download_url(self.url, root=self.raw_folder, filename=filename, md5=self.md5)
        extract_file(file_path)
        train_set = read_pickle_file(extracted_file_path, self.train_list)
        test_set = read_pickle_file(extracted_file_path, self.test_list)
        save(train_set, os.path.join(self.processed_folder, 'train.pt'))
        save(test_set, os.path.join(self.processed_folder, 'test.pt'))
        classes_to_labels = parse_meta(os.path.join(self.raw_folder, self.base_folder, self.meta['filename']), self.meta['key'])
        save(classes_to_labels, os.path.join(self.processed_folder, 'meta.pt'))
        return


def read_pickle_file(path, filenames):
    img, label = [], []
    for filename in filenames:
        file_path = os.path.join(path, filename)
        with open(file_path, 'rb') as f:
            entry = pickle.load(f, encoding='latin1')
            img.append(entry['data'])
            label.extend(entry['labels']) if 'labels' in entry else label.extend(entry['fine_labels'])
    img = np.vstack(img).reshape(-1, 3, 32, 32)
    img = img.transpose((0, 2, 3, 1))
    return img, label


def parse_meta(path, key):
    with open(path, 'rb') as f:
        data = pickle.load(f, encoding='latin1')
        classes = data[key]
    classes_to_labels = {classes[i]: i for i in range(len(classes))}
    return classes_to_labels


_classes = {
    'aquatic mammals': ['beaver', 'dolphin', 'otter', 'seal', 'whale'],
    'fish': ['aquarium_fish', 'flatfish', 'ray', 'shark', 'trout'],
    'flowers': ['orchid', 'poppy', 'rose', 'sunflower', 'tulip'],
    'food containers': ['bottle', 'bowl', 'can', 'cup', 'plate'],
    'fruit and vegetables': ['apple', 'mushroom', 'orange', 'pear', 'sweet_pepper'],
    'household electrical devices': ['clock', 'keyboard', 'lamp', 'telephone', 'television'],
    'household furniture': ['bed', 'chair', 'couch', 'table', 'wardrobe'],
    'insects': ['bee', 'beetle', 'butterfly', 'caterpillar', 'cockroach'],
    'large carnivores': ['bear', 'leopard', 'lion', 'tiger', 'wolf'],
    'large man-made outdoor things': ['bridge', 'castle', 'house', 'road', 'skyscraper'],
    'large natural outdoor scenes': ['cloud', 'forest', 'mountain', 'plain', 'sea'],
    'large omnivores and herbivores': ['camel', 'cattle', 'chimpanzee', 'elephant', 'kangaroo'],
    'medium-sized mammals': ['fox', 'porcupine', 'possum', 'raccoon', 'skunk'],
    'non-insect invertebrates': ['crab', 'lobster', 'snail', 'spider', 'worm'],
    'people': ['baby', 'boy', 'girl', 'man', 'woman'],
    'reptiles': ['crocodile', 'dinosaur', 'lizard', 'snake', 'turtle'],
    'small mammals': ['hamster', 'mouse', 'rabbit', 'shrew', 'squirrel'],
    'trees': ['maple_tree', 'oak_tree', 'palm_tree', 'pine_tree', 'willow_tree'],
    'vehicles 1': ['bicycle', 'bus', 'motorcycle', 'pickup_truck', 'train'],
    'vehicles 2': ['lawn_mower', 'rocket', 'streetcar', 'tank', 'tractor']
}