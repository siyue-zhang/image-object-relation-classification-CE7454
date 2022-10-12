import io
import json
import logging
import os
from syslog import LOG_WARNING

import torch
import torchvision.transforms as trn
from PIL import Image, ImageFile
from torch.utils.data import Dataset
from torchvision.transforms import AutoAugment, AutoAugmentPolicy
import numpy as np

# to fix "OSError: image file is truncated"
ImageFile.LOAD_TRUNCATED_IMAGES = True


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
            trn.Resize((int(1.4*H), int(1.4*W))),
            # trn.Resize((H,W)),
            trn.RandomCrop((H, W)),
            trn.RandomHorizontalFlip(),
            # trn.RandomCrop((H, W), padding=4),
            trn.ColorJitter(brightness=0.5, contrast=0.5, hue=0.5),
            trn.RandomPerspective(distortion_scale=0.6, p=0.5),
            trn.AugMix(),
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


class PSGClsDataset(Dataset):
    def __init__(
        self,
        stage,
        root='./data/coco/',
        num_classes=56,
    ):
        super(PSGClsDataset, self).__init__()
        with open('./data/psg/psg_cls_basic.json') as f:
            dataset = json.load(f)
        self.imglist = [
            d for d in dataset['data']
            if d['image_id'] in dataset[f'{stage}_image_ids']
        ]
        self.root = root
        self.transform_image = get_transforms(stage)
        self.num_classes = num_classes

    def __len__(self):
        return len(self.imglist)

    def __getitem__(self, index):
        sample = self.imglist[index]
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

class PredictDataset(Dataset):
    def __init__(
        self,
        root='./data/coco/',
        num_classes=56,
    ):
        super(PredictDataset, self).__init__()
        with open('./data/psg/psg_cls_basic.json') as f:
            dataset = json.load(f)
        self.imglist = [
            d for d in dataset['data']
            if (d['image_id'] in dataset[f'train_image_ids']) or (d['image_id'] in dataset[f'val_image_ids'])
        ]
        self.root = root
        self.transform_image = get_transforms('train')
        self.num_classes = num_classes

    def __len__(self):
        return len(self.imglist)

    def __getitem__(self, index):
        sample = self.imglist[index]
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
