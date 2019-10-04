import codecs
import numpy as np
import os
import torch
from PIL import Image
from torch.utils.data import Dataset
from utils import makedir_exist_ok, save, load
from .utils import download_url, extract_file, make_classes_counts


class MNIST(Dataset):
    data_name = 'MNIST'
    urls = ['http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz',
            'http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz',
            'http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz',
            'http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz']
    classes = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
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
        self.classes_to_labels = {self.classes[i]: i for i in range(len(self.classes))}
        self.classes_size = len(self.classes_to_labels)
        self.classes_counts = make_classes_counts(self.label)

    def __getitem__(self, index):
        img, label = Image.fromarray(self.img[index], mode='L'), torch.tensor(self.label[index])
        input = {'img': img, 'label': label}
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
        for url in self.urls:
            filename = os.path.basename(url)
            file_path = os.path.join(self.raw_folder, filename)
            download_url(url, root=self.raw_folder, filename=filename, md5=None)
            extract_file(file_path)
        train_set = (read_image_file(os.path.join(self.raw_folder, 'train-images-idx3-ubyte')),
                     read_label_file(os.path.join(self.raw_folder, 'train-labels-idx1-ubyte')))
        test_set = (read_image_file(os.path.join(self.raw_folder, 't10k-images-idx3-ubyte')),
                    read_label_file(os.path.join(self.raw_folder, 't10k-labels-idx1-ubyte')))
        save(train_set, os.path.join(self.processed_folder, 'train.pt'))
        save(test_set, os.path.join(self.processed_folder, 'test.pt'))
        return

    def __repr__(self):
        fmt_str = 'Dataset {}\n'.format(self.__class__.__name__)
        fmt_str += '    Number of data points: {}\n'.format(self.__len__())
        fmt_str += '    Split: {}\n'.format(self.split)
        fmt_str += '    Root: {}\n'.format(self.root)
        fmt_str += '    Transforms: {}\n'.format(self.transform.__repr__())
        return fmt_str


class EMNIST(MNIST):
    data_name = 'EMNIST'
    url = 'http://www.itl.nist.gov/iaui/vip/cs_links/EMNIST/gzip.zip'
    subsets = ['byclass', 'bymerge', 'balanced', 'letters', 'digits', 'mnist']
    digits_classes = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    upper_letters_classes = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R',
                             'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
    lower_letters_classes = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r',
                             's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
    merged_classes = ['c', 'i', 'j', 'k', 'l', 'm', 'o', 'p', 's', 'u', 'v', 'w', 'x', 'y', 'z']
    unmerged_classes = list(set(lower_letters_classes) - set(merged_classes))
    feature_dim = {'img': 1}

    def __init__(self, root, split, **kwargs):
        split_list = split.split('_')
        if split_list[1] == 'digits' or split_list[1] == 'mnist':
            self.classes = self.digits_classes
        elif split_list[1] == 'letters':
            self.classes = self.upper_letters_classes + self.unmerged_classes
        elif split_list[1] == 'balanced' or split_list[1] == 'bymerge':
            self.classes = self.digits_classes + self.upper_letters_classes + self.unmerged_classes
        elif split_list[1] == 'byclass':
            self.classes = self.digits_classes + self.upper_letters_classes + self.lower_letters_classes
        super(EMNIST, self).__init__(root, split, **kwargs)

    def download(self):
        if self._check_exists():
            return
        makedir_exist_ok(self.raw_folder)
        filename = os.path.basename(self.url)
        file_path = os.path.join(self.raw_folder, filename)
        download_url(self.url, root=self.raw_folder, filename=filename, md5=None)
        extract_file(file_path)
        gzip_folder = os.path.join(self.raw_folder, 'gzip')
        for gzip_file in os.listdir(gzip_folder):
            if gzip_file.endswith('.gz'):
                extract_file(os.path.join(gzip_folder, gzip_file))
        for subset in self.subsets:
            train_set = (read_image_file(os.path.join(gzip_folder, 'emnist-{}-train-images-idx3-ubyte'.format(subset))),
                         read_label_file(os.path.join(gzip_folder, 'emnist-{}-train-labels-idx1-ubyte'.format(subset))))
            test_set = (read_image_file(os.path.join(gzip_folder, 'emnist-{}-test-images-idx3-ubyte'.format(subset))),
                        read_label_file(os.path.join(gzip_folder, 'emnist-{}-test-labels-idx1-ubyte'.format(subset))))
            save(train_set, os.path.join(self.processed_folder, 'train_{}.pt'.format(subset)))
            save(test_set, os.path.join(self.processed_folder, 'test_{}.pt'.format(subset)))
        return


class FashionMNIST(MNIST):
    data_name = 'FashionMNIST'
    urls = ['http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-images-idx3-ubyte.gz',
            'http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-labels-idx1-ubyte.gz',
            'http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-images-idx3-ubyte.gz',
            'http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-labels-idx1-ubyte.gz', ]
    classes = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    feature_dim = {'img': 1}

    @property
    def data_path(self):
        return os.path.join(self.root, 'fashionmnist')


def get_int(b):
    return int(codecs.encode(b, 'hex'), 16)


def read_image_file(path):
    with open(path, 'rb') as f:
        data = f.read()
        assert get_int(data[:4]) == 2051
        length = get_int(data[4:8])
        num_rows = get_int(data[8:12])
        num_cols = get_int(data[12:16])
        parsed = np.frombuffer(data, dtype=np.uint8, offset=16).reshape((length, num_rows, num_cols))
        return parsed


def read_label_file(path):
    with open(path, 'rb') as f:
        data = f.read()
        assert get_int(data[:4]) == 2049
        length = get_int(data[4:8])
        parsed = np.frombuffer(data, dtype=np.uint8, offset=8).reshape(length).astype(np.int64)
        return parsed