import config
config.init()
import torch
from torch.utils.data import Dataset

class BITS(Dataset):
    data_name = 'BIT'
    output_names = ['block']
    
    def __init__(self, train=True, transform=None):
        self.transform = transform
        self.train = train
        if(train):
            self.bits, _ = fetch_dataset_bits(config.PARAM['block_size'])
        else:
           _, self.bits = fetch_dataset_bits(config.PARAM['block_size'])
        
    def __getitem__(self, index):
        bits = self.bits[index]      
        input = {'bits': bits}
        if self.transform is not None:
            input = self.transform(input)            
        return input
        
    def __len__(self):
        return len(self.bits)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        tmp = 'train' if self.train is True else 'test'
        fmt_str += '    Split: {}\n'.format(tmp)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str
        
def fetch_dataset_bits(block_size):
    train_size = 200000 if (config.PARAM['data_size']['train']==0) else config.PARAM['data_size']['train']
    test_size = 200000 if (config.PARAM['data_size']['test']==0) else config.PARAM['data_size']['test']
    train_dataset = torch.randint(low=0,high=2,size=(train_size,block_size),dtype=torch.float32)
    test_dataset = torch.randint(low=0,high=2,size=(test_size,block_size),dtype=torch.float32)
    return train_dataset,test_dataset