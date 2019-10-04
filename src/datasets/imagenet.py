import os
import shutil
import torch
from torch.utils.data import Dataset

from utils import makedir_exist_ok, save, load
from .utils import download_url, extract_file, make_classes_counts, make_img_dataset, default_loader, IMG_EXTENSIONS


class ImageNet(Dataset):
    data_name = 'ImageNet'
    urls = ['http://www.image-net.org/challenges/LSVRC/2012/nnoupb/ILSVRC2012_img_train.tar',
            'http://www.image-net.org/challenges/LSVRC/2012/nnoupb/ILSVRC2012_img_val.tar',
            'http://www.image-net.org/challenges/LSVRC/2012/nnoupb/ILSVRC2012_devkit_t12.tar.gz']
    md5s = ['1d675b47d978889d74fa0da5fadfb00e', '29b22e2961454d5413ddabcf34fc5622', 'fa75699e90414af021442c21a62c3abf']
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
        self.classes_to_labels, self.classes_to_names = load(os.path.join(self.processed_folder, 'meta.pt'))
        self.classes = self.classes_to_labels.keys()
        self.classes_size = len(self.classes)
        self.classes_counts = make_classes_counts(self.label)

    def __getitem__(self, index):
        input = {'img': default_loader(self.img[index])} if 'label' is None else {
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

    def download(self):
        if self._check_exists():
            return
        makedir_exist_ok(self.raw_folder)
        for i in range(len(self.urls)):
            url, md5 = self.urls[i], self.md5s[i]
            filename = os.path.basename(url)
            file_path = os.path.join(self.raw_folder, filename)
            download_url(url, root=self.raw_folder, filename=filename, md5=None)
            if i == len(self.urls) - 1:
                extract_file(file_path, dest=os.path.join(self.raw_folder))
            else:
                extract_file(file_path, dest=os.path.join(self.raw_folder, filename.split(os.extsep)[0]))
        meta = parse_devkit(os.path.join(self.raw_folder, 'ILSVRC2012_devkit_t12'))
        prepare_train_folder(os.path.join(self.raw_folder, 'ILSVRC2012_img_train'))
        prepare_val_folder(os.path.join(self.raw_folder, 'ILSVRC2012_img_val'), meta[2])
        train_set = make_img_dataset(os.path.join(self.raw_folder, 'ILSVRC2012_img_train'), IMG_EXTENSIONS, meta[0])
        test_set = make_img_dataset(os.path.join(self.raw_folder, 'ILSVRC2012_img_val'), IMG_EXTENSIONS, meta[0])
        save(train_set, os.path.join(self.processed_folder, 'train.pt'))
        save(test_set, os.path.join(self.processed_folder, 'test.pt'))
        save(meta[:2], os.path.join(self.processed_folder, 'meta.pt'))
        return

    def __repr__(self):
        fmt_str = 'Dataset {}\n'.format(self.__class__.__name__)
        fmt_str += '    Number of data points: {}\n'.format(self.__len__())
        fmt_str += '    Split: {}\n'.format(self.split)
        fmt_str += '    Root: {}\n'.format(self.root)
        fmt_str += '    Transforms: {}\n'.format(self.transform.__repr__())
        return fmt_str


def parse_devkit(path):
    classes_to_labels, classes_to_names = parse_meta(path)
    val_idcs = parse_val_groundtruth(path)
    labels_to_classes = {v: k for k, v in classes_to_labels.items()}
    val_classes = [labels_to_classes[idx] for idx in val_idcs]
    return classes_to_labels, classes_to_names, val_classes


def parse_meta(path):
    import scipy.io as sio
    metafile = os.path.join(path, 'data', 'meta.mat')
    meta = sio.loadmat(metafile, squeeze_me=True)['synsets']
    nums_children = list(zip(*meta))[4]
    meta = [meta[idx] for idx, num_children in enumerate(nums_children) if num_children == 0]
    idcs, wnids, classes = list(zip(*meta))[:3]
    classes = [tuple(clss.split(', ')) for clss in classes]
    classes_to_labels = {wnid: idx for idx, wnid in zip(idcs, wnids)}
    classes_to_names = {wnid: clss for wnid, clss in zip(wnids, classes)}
    return classes_to_labels, classes_to_names


def parse_val_groundtruth(path):
    with open(os.path.join(path, 'data', 'ILSVRC2012_validation_ground_truth.txt'), 'r') as txtfh:
        val_idcs = txtfh.readlines()
    return [int(val_idx) for val_idx in val_idcs]


def prepare_train_folder(folder):
    for archive in [os.path.join(folder, archive) for archive in os.listdir(folder)]:
        extract_file(archive, os.path.splitext(archive)[0], delete=True)


def prepare_val_folder(folder, wnids):
    img_files = sorted([os.path.join(folder, file) for file in os.listdir(folder)])
    for wnid in set(wnids):
        os.mkdir(os.path.join(folder, wnid))
    for wnid, img_file in zip(wnids, img_files):
        shutil.move(img_file, os.path.join(folder, wnid, os.path.basename(img_file)))