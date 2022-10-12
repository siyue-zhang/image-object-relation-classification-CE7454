import io
import json
import logging
import os
from random import random
from syslog import LOG_WARNING

import torch
import torchvision.transforms as trn
from PIL import Image, ImageFile
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision.transforms import AutoAugment, AutoAugmentPolicy
import numpy as np
import matplotlib.pyplot as plt


class Convert:
    def __init__(self, mode='RGB'):
        self.mode = mode

    def __call__(self, image):
        return image.convert(self.mode)


def get_transforms(stage: str):
    # mean, std = [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]
    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    # train set
    # mean, std = [0.495, 0.493, 0.491], [0.320, 0.319, 0.320]
    H = 256
    W = 256

    if stage == 'train':
        return trn.Compose([
            Convert('RGB'),
            # trn.Resize((H, W)),
            trn.Resize((int(1.4*H), int(1.4*W))),
            trn.RandomCrop((H, W)),
            trn.RandomHorizontalFlip(),
            trn.ColorJitter(brightness=0.5, contrast=0.5, hue=0.5),
            trn.ToTensor(),
            trn.Normalize(mean, std),
        ])

    elif stage in ['val', 'test']:
        return trn.Compose([
            Convert('RGB'),
            trn.Resize((H, W)),
            trn.ToTensor(),
            trn.Normalize(mean, std),
        ])


class balancePSGClsDataset(Dataset):
    def __init__(
        self,
        stage,
        root='./data/coco/',
        num_classes=56,
        low=[]
    ):
        super(balancePSGClsDataset, self).__init__()
        with open('./data/psg/psg_cls_basic.json') as f:
            dataset = json.load(f)
        self.imglist = [
            d for d in dataset['data']
            if d['image_id'] in dataset[f'{stage}_image_ids']
        ]
        self.low=low
        self.root = root
        self.imglist_new = []
        for image in self.imglist:
            if any([ relation in self.low for relation in image['relations']]):
                for _ in range(1):
                    self.imglist_new.append(image)
            elif random()<0.01:
                self.imglist_new.append(image)
        self.transform_image = get_transforms(stage)
        self.num_classes = num_classes

    def __len__(self):
        return len(self.imglist_new)

    def __getitem__(self, index):
        sample = self.imglist_new[index]
        path = os.path.join(self.root, sample['file_name'])
        try:
            with open(path, 'rb') as f:
                content = f.read()
                filebytes = content
                buff = io.BytesIO(filebytes)
                image = Image.open(buff).convert('RGB')
                data = self.transform_image(image)
                # sample['data'] = data
        except Exception as e:
            logging.error('Error, cannot read [{}]'.format(path))
            raise e
        # Generate Soft Label
        soft_label = torch.Tensor(self.num_classes)
        soft_label.fill_(0)
        soft_label[sample['relations']] = 1
        sample['soft_label'] = soft_label
        del sample
        return data, soft_label



class balancePredictDataset(Dataset):
    def __init__(
        self,
        root='./data/coco/',
        num_classes=56,
        low=[]
    ):
        super(balancePredictDataset, self).__init__()
        with open('./data/psg/psg_cls_basic.json') as f:
            dataset = json.load(f)
        self.imglist = [
            d for d in dataset['data']
            if (d['image_id'] in dataset[f'train_image_ids']) or (d['image_id'] in dataset[f'val_image_ids'])
        ]
        self.low=low
        self.root = root
        self.imglist_new = []
        for image in self.imglist:
            if any([ relation in self.low for relation in image['relations']]):
                for _ in range(1):
                    self.imglist_new.append(image)
            elif random()<0.01:
                self.imglist_new.append(image)
        self.transform_image = get_transforms('train')
        self.num_classes = num_classes

    def __len__(self):
        return len(self.imglist_new)

    def __getitem__(self, index):
        sample = self.imglist_new[index]
        path = os.path.join(self.root, sample['file_name'])
        try:
            with open(path, 'rb') as f:
                content = f.read()
                filebytes = content
                buff = io.BytesIO(filebytes)
                image = Image.open(buff).convert('RGB')
                data = self.transform_image(image)
        except Exception as e:
            logging.error('Error, cannot read [{}]'.format(path))
            raise e
        # Generate Soft Label
        soft_label = torch.Tensor(self.num_classes)
        soft_label.fill_(0)
        soft_label[sample['relations']] = 1
        sample['soft_label'] = soft_label
        del sample
        return data, soft_label





if __name__ == '__main__':
    # low = [6,7,8,9,10,12,13,15,17,18,19,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,50,51,52,53,54,55]
    low=[6,7,8,9,10,13,15,17,18,19,24,25,27,28,29,30,31,32,33,34,35,38,39,40,41,43,44,50,52,53]
    # train_dataset = balancePredictDataset(low=low)
    train_dataset = balancePSGClsDataset(stage='train',low=low)
    print(len(train_dataset.imglist))      
    print(len(train_dataset.imglist_new))
    x = []
    for i in train_dataset.imglist_new:
        for y in i['relations']:
            x.append(y)
    plt.hist(x, bins=56)
    plt.savefig('tmp.png')


