from cProfile import label
import torch
import clip
from PIL import Image
import numpy as np
import torch.nn as nn
from torch.utils.data import Dataset
import json
import os
import io
import logging
from torch.utils.data import DataLoader


batch_size=32

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.cuda.set_device(2)

model, preprocess = clip.load("ViT-B/32", device=device)
# image = preprocess(Image.open("input.jpg")).unsqueeze(0).to(device)

PREDICATES=[
    # 'over',
    # 'in front of',
    # 'beside',
    # 'on',
    # 'in',
    # 'attached to',
    'hanging from',
    'on back of',
    'falling off',
    'going down',
    'painted on',
    'walking on',
    'running on',
    'crossing',
    'standing on',
    'lying on',
    'sitting on',
    'flying over',
    'jumping over',
    'jumping from',
    'wearing',
    'holding',
    'carrying',
    'looking at',
    'guiding',
    'kissing',
    'eating',
    'drinking',
    'feeding',
    'biting',
    'catching',
    'picking',
    'playing with',
    'chasing',
    'climbing',
    'cleaning',
    'playing',
    'touching',
    'pushing',
    'pulling',
    'opening',
    'cooking',
    'talking to',
    'throwing',
    'slicing',
    'driving',
    'riding',
    'parked on',
    'driving on',
    'about to hit',
    'kicking',
    'swinging',
    'entering',
    'exiting',
    'enclosing',
    'leaning on',
]

PREDICATES_mod=[
    # 'over',
    # 'in front of',
    # 'beside',
    # 'on',
    # 'in',
    # 'attached to',
    'hanging',
    'on back',
    'falling off',
    'going down',
    'painted',
    'walking',
    'running',
    'crossing',
    'standing',
    'lying',
    'sitting',
    'flying',
    'jumping over',
    'jumping',
    'wearing',
    'holding',
    'carrying',
    'looking',
    'guiding',
    'kissing',
    'eating',
    'drinking',
    'feeding',
    'biting',
    'catching',
    'picking',
    'playing',
    'chasing',
    'climbing',
    'cleaning',
    'playing',
    'touching',
    'pushing',
    'pulling',
    'opening',
    'cooking',
    'talking',
    'throwing',
    'slicing',
    'driving',
    'riding',
    'parked',
    'driving',
    'hitting',
    'kicking',
    'swinging',
    'entering',
    'exiting',
    'enclosing',
    'leaning',
]

PREDICATES_mod2=[
    # 'over',
    # 'in front of',
    # 'beside',
    # 'on',
    # 'in',
    # 'attached to',
    'hanging',
    'being on back',
    'falling off',
    'going down',
    'painting',
    'walking',
    'running',
    'crossing',
    'standing',
    'lying',
    'sitting',
    'flying',
    'jumping over',
    'jumping',
    'wearing',
    'holding',
    'carrying',
    'looking',
    'guiding',
    'kissing',
    'eating',
    'drinking',
    'feeding',
    'biting',
    'catching',
    'picking',
    'playing',
    'chasing',
    'climbing',
    'cleaning',
    'playing',
    'touching',
    'pushing',
    'pulling',
    'opening',
    'cooking',
    'talking',
    'throwing',
    'slicing',
    'driving',
    'riding',
    'parked',
    'driving',
    'hitting',
    'kicking',
    'swinging',
    'entering',
    'exiting',
    'enclosing',
    'leaning',
]

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
        self.stage=stage
        self.num_classes = num_classes

    def __len__(self):
        return len(self.imglist)

    def __getitem__(self, index):
        sample = self.imglist[index]
        path = os.path.join(self.root, sample['file_name'])
        image = Image.open(path)
        # image.save('tmp.png')
        image = preprocess(image).unsqueeze(0).to(device)
        if self.stage!='test':
            relations = sample['relations']
            return image, relations
        else:
            return image, None

stage = 'val'
val = PSGClsDataset(stage=stage)
k = 3
pred_list = []
gt_list = []
for n in range(len(val)):
    image, relations = val.__getitem__(n)
    text_inputs =  clip.tokenize([f'relation of {i}' for i in PREDICATES]).to(device)
    # text_inputs =  clip.tokenize([f'action of {i}' for i in PREDICATES_mod2]).to(device)

    # Calculate features
    with torch.no_grad():
        image_features = model.encode_image(image)
        text_features = model.encode_text(text_inputs)

    # Pick the top 5 most similar labels for the image
    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)
    similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
    values, indices = similarity[0].topk(k)

    idx = indices+6
    pred_list.append(list(idx.cpu().numpy()))
    gt_list.append(relations)


if stage!='test':
# compute mean recall
    score_list = np.zeros([56, 2], dtype=int)
    for gt, pred in zip(gt_list, pred_list):
        for gt_id in gt:
            # pos 0 for counting all existing relations
            score_list[gt_id][0] += 1
            if gt_id in pred:
                # pos 1 for counting relations that is recalled
                score_list[gt_id][1] += 1
    score_list = score_list[6:]
    # to avoid nan
    score_list[:, 0][score_list[:, 0] == 0] = 1
    meanrecall = np.mean(score_list[:, 1] / score_list[:, 0])
    print('val',meanrecall)
else:
    # save into the file
    with open(f'result.txt', 'w') as writer:
        for label_list in pred_list:
            a = [str(x) for x in label_list]
            save_str = ' '.join(a)
            writer.writelines(save_str + '\n')
    print('Result Saved!', flush=True)
