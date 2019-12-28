import os
from PIL import Image
from torch.utils.data import Dataset
from utils import check_exists, save, load
from .utils import IMG_EXTENSIONS
from .utils import make_data


class ImageFolder(Dataset):

    def __init__(self, root, transform=None):
        self.root = os.path.expanduser(root)
        self.transform = transform
        if not check_exists(self.processed_folder):
            self.process()
        self.img = load(os.path.join(self.processed_folder, 'data.pt'))

    def __getitem__(self, index):
        img = Image.open(self.img[index], mode='r').convert('RGB')
        input = {'img': img}
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
            raise RuntimeError('Dataset not found')
        data_set = self.make_data()
        save(data_set, os.path.join(self.processed_folder, 'data.pt'))
        return

    def __repr__(self):
        fmt_str = 'Dataset {}\nSize: {}\nRoot: {}\nTransforms: {}'.format(
            self.__class__.__name__, self.__len__(), self.root, self.transform.__repr__())
        return fmt_str

    def make_data(self):
        img = make_data(self.raw_folder, IMG_EXTENSIONS)
        return img