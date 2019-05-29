from .mnist import MNIST, EMNIST, FashionMNIST
from .cifar import CIFAR10, CIFAR100
from .svhn import SVHN
from .folder import ImageFolder, DatasetFolder
from .voc import VOCDetection, VOCSegmentation
from .coco import CocoDetection, CocoCaptions
from .cub import CUB2011
from .wheatimage import WheatImage
from .mosi import MOSI
from .transforms import *


__all__ = ('MNIST','EMNIST', 'FashionMNIST',
           'CIFAR10', 'CIFAR100', 'SVHN',
           'ImageFolder', 'DatasetFolder',
           'VOCDetection', 'VOCSegmentation',         
           'CocoCaptions', 'CocoDetection',
           'CUB2011',
           'WheatImage','MOSI')