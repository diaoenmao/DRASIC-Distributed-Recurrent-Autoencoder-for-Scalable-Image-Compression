import os
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from pycocotools.coco import COCO

class CocoDetection(Dataset):
    def __init__(self, root, annFile, transform=None):
        
        self.root = root
        self.coco = COCO(annFile)
        self.ids = list(self.coco.imgs.keys())
        self.transform = transform

    def __getitem__(self, index):
        coco = self.coco
        img_id = self.ids[index]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        anns = coco.loadAnns(ann_ids)
        bbox = []
        label = []
        segmentation = np.zeros(coco.annToMask(anns[0]).shape,dtype=np.int64)
        for ann in anns:
            bbox.append(torch.tensor([ann['bbox'][1],ann['bbox'][1]+ann['bbox'][3],ann['bbox'][0],ann['bbox'][0]+ann['bbox'][2]]))
            label.append(torch.tensor(ann['category_id']))
            mask = coco.annToMask(ann).astype(np.bool)
            segmentation[mask] = ann['category_id']
        bbox = torch.stack(bbox,0)
        label = torch.stack(label,0)
        path = coco.loadImgs(img_id)[0]['file_name']
        img = Image.open(os.path.join(self.root, path)).convert('RGB')
        segmentation = torch.from_numpy(segmentation)
        input = {'img': img, 'bbox': bbox, 'label': label, 'segmentation':segmentation}
        if self.transform is not None:
            input = self.transform(input)
        return input
        
    def __len__(self):
        return len(self.ids)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str
        
class CocoCaptions(Dataset):
    def __init__(self, root, annFile, transform=None, target_transform=None):
        self.root = os.path.expanduser(root)
        self.coco = COCO(annFile)
        self.ids = list(self.coco.imgs.keys())
        self.transform = transform

    def __getitem__(self, index):
        coco = self.coco
        img_id = self.ids[index]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        anns = coco.loadAnns(ann_ids)
        caption = [ann['caption'] for ann in anns]
        path = coco.loadImgs(img_id)[0]['file_name']
        img = Image.open(os.path.join(self.root, path)).convert('RGB')
        input = {'img': img, 'caption': caption}
        if self.transform is not None:
            input = self.transform(input)
        return input

    def __len__(self):
        return len(self.ids)

