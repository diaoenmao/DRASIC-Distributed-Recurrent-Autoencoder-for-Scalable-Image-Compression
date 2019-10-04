from .mnist import MNIST, EMNIST, FashionMNIST
from .cifar import CIFAR10, CIFAR100
from .svhn import SVHN
from .imagenet import ImageNet
from .folder import ImageFolder
from .utils import *

__all__ = ('MNIST','EMNIST', 'FashionMNIST',
           'CIFAR10', 'CIFAR100', 'SVHN',
           'ImageNet',
           'ImageFolder')
