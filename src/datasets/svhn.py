import numpy as np
import os
import torch
from PIL import Image
from torch.utils.data import Dataset
from utils import makedir_exist_ok, save, load
from .utils import download_url, make_classes_counts


class SVHN(Dataset):
    data_name = 'SVHN'
    urls = ['http://ufldl.stanford.edu/housenumbers/train_32x32.mat',
            'http://ufldl.stanford.edu/housenumbers/test_32x32.mat',
            'http://ufldl.stanford.edu/housenumbers/extra_32x32.mat']
    md5s = ['e26dedcc434d2e4c54c9b2d4a06d8373', 'eb5a983be6a315427106f1b164d9cef3', 'a93ce644f1a588dc4d68dda5feec44a7']
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
        img, label = Image.fromarray(self.img[index]), torch.tensor(self.label[index])
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
        makedir_exist_ok(self.processed_folder)
        for i in range(len(self.urls)):
            url, md5 = self.urls[i], self.md5s[i]
            filename = os.path.basename(url)
            download_url(url, root=self.raw_folder, filename=filename, md5=md5)
        train_set = read_mat_file(os.path.join(self.raw_folder, 'train_32x32.mat'))
        test_set = read_mat_file(os.path.join(self.raw_folder, 'test_32x32.mat'))
        extra_set = read_mat_file(os.path.join(self.raw_folder, 'extra_32x32.mat'))
        save(train_set, os.path.join(self.processed_folder, 'train.pt'))
        save(test_set, os.path.join(self.processed_folder, 'test.pt'))
        save(extra_set, os.path.join(self.processed_folder, 'extra.pt'))
        return

    def __repr__(self):
        fmt_str = 'Dataset {}\n'.format(self.__class__.__name__)
        fmt_str += '    Number of data points: {}\n'.format(self.__len__())
        fmt_str += '    Split: {}\n'.format(self.split)
        fmt_str += '    Root: {}\n'.format(self.root)
        fmt_str += '    Transforms: {}\n'.format(self.transform.__repr__())
        return fmt_str


def read_mat_file(path):
    import scipy.io as sio
    mat = sio.loadmat(path)
    img = mat['X']
    label = mat['y'].astype(np.int64).squeeze()
    np.place(label, label == 10, 0)
    label = label.tolist()
    img = np.transpose(img, (3, 0, 1, 2))
    return img, label