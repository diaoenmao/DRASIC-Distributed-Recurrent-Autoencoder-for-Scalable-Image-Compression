import anytree
import numpy as np
import os
import torch
from PIL import Image
from torch.utils.data import Dataset
from utils import check_exists, makedir_exist_ok, save, load
from .utils import download_url, extract_file, make_classes_counts, make_tree, make_flat_index


class SVHN(Dataset):
    data_name = 'SVHN'
    file = [('http://ufldl.stanford.edu/housenumbers/train_32x32.mat', 'e26dedcc434d2e4c54c9b2d4a06d8373'),
            ('http://ufldl.stanford.edu/housenumbers/test_32x32.mat', 'eb5a983be6a315427106f1b164d9cef3'),
            ('http://ufldl.stanford.edu/housenumbers/extra_32x32.mat', 'a93ce644f1a588dc4d68dda5feec44a7')]

    def __init__(self, root, split, subset, transform=None):
        self.root = os.path.expanduser(root)
        self.split = split
        self.subset = subset
        self.transform = transform
        if not check_exists(self.processed_folder):
            self.process()
        self.img, self.target = load(os.path.join(self.processed_folder, '{}.pt'.format(self.split)))
        self.classes_to_labels, self.classes_size = load(os.path.join(self.processed_folder, 'meta.pt'))
        self.classes_to_labels, self.classes_size = self.classes_to_labels[self.subset], self.classes_size[self.subset]
        self.classes_counts = make_classes_counts(self.target[self.subset])

    def __getitem__(self, index):
        img, target = Image.fromarray(self.img[index]), {s: torch.tensor(self.target[s][index]) for s in self.target}
        input = {'img': img, **target}
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

    def process(self):
        if not check_exists(self.raw_folder):
            self.download()
        train_set, test_set, extra_set, meta = self.make_data()
        save(train_set, os.path.join(self.processed_folder, 'train.pt'))
        save(test_set, os.path.join(self.processed_folder, 'test.pt'))
        save(extra_set, os.path.join(self.processed_folder, 'extra.pt'))
        save(meta, os.path.join(self.processed_folder, 'meta.pt'))
        return

    def download(self):
        makedir_exist_ok(self.raw_folder)
        for (url, md5) in self.file:
            filename = os.path.basename(url)
            download_url(url, self.raw_folder, filename, md5)
            extract_file(os.path.join(self.raw_folder, filename))
        return

    def __repr__(self):
        fmt_str = 'Dataset {}\nSize: {}\nRoot: {}\nSplit: {}\nSubset: {}\nTransforms: {}'.format(
            self.__class__.__name__, self.__len__(), self.root, self.split, self.subset, self.transform.__repr__())
        return fmt_str

    def make_data(self):
        train_img, train_label = read_mat_file(os.path.join(self.raw_folder, 'train_32x32.mat'))
        test_img, test_label = read_mat_file(os.path.join(self.raw_folder, 'test_32x32.mat'))
        extra_img, extra_label = read_mat_file(os.path.join(self.raw_folder, 'extra_32x32.mat'))
        train_target, test_target, extra_target = {'label': train_label}, {'label': test_label}, {'label': extra_label}
        classes_to_labels = {'label': anytree.Node('U', index=[])}
        classes = list(map(str, list(range(10))))
        for c in classes:
            make_tree(classes_to_labels['label'], [c])
        classes_size = {'label': make_flat_index(classes_to_labels['label'])}
        return (train_img, train_target), (test_img, test_target), (extra_img, extra_target), \
               (classes_to_labels, classes_size)


def read_mat_file(path):
    import scipy.io as sio
    mat = sio.loadmat(path)
    img = mat['X']
    label = mat['y'].astype(np.int64).squeeze()
    np.place(label, label == 10, 0)
    img = np.transpose(img, (3, 0, 1, 2))
    return img, label