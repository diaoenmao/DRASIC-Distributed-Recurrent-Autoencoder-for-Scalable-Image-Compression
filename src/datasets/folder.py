import os
import shutil
import torch
from torch.utils.data import Dataset
from utils import makedir_exist_ok, save, load
from .utils import find_classes, make_img_dataset, make_classes_counts, default_loader, IMG_EXTENSIONS


class ImageFolder(Dataset):
    feature_dim = {'img': 1}

    def __init__(self, root, transform=None):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.data_name = os.path.basename(self.root)
        if not self._check_exists():
            makedir_exist_ok(self.raw_folder)
            files = os.listdir(self.root)
            for f in files:
                print(f)
                shutil.move(os.path.join(root, f), self.raw_folder)
            classes_to_labels = find_classes(self.raw_folder)
            dataset = make_img_dataset(self.raw_folder, IMG_EXTENSIONS, classes_to_labels)
            save(dataset, os.path.join(self.processed_folder, 'data.pt'))
            save(classes_to_labels, os.path.join(self.processed_folder, 'meta.pt'))
        self.img, self.label = load(os.path.join(self.processed_folder, 'data.pt'))
        self.classes_to_labels = load(os.path.join(self.processed_folder, 'meta.pt'))
        self.classes = self.classes_to_labels.keys()
        self.classes_size = len(self.classes)
        self.classes_counts = make_classes_counts(self.label)

    def __getitem__(self, index):
        input = {'img': default_loader(self.img[index])} if not self.label else {
            'img': default_loader(self.img[index]), 'label': torch.tensor(self.label[index])
        }
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

    def __repr__(self):
        fmt_str = 'Dataset {}\n'.format(self.__class__.__name__)
        fmt_str += '    Number of data points: {}\n'.format(self.__len__())
        fmt_str += '    Root: {}\n'.format(self.root)
        fmt_str += '    Transforms: {}\n'.format(self.transform.__repr__())
        return fmt_str