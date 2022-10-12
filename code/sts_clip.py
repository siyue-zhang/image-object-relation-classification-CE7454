from cProfile import label
import torch
import clip
from PIL import Image
from tqdm import tqdm
import numpy as np
import torch.nn as nn
from torch.utils.data import Dataset
import json
import os
import io
import logging
from torch.utils.data import DataLoader

from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
import os, json, cv2, random

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog


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
        image = preprocess(image).unsqueeze(0).to(device)
        if self.stage!='test':
            relations = sample['relations']
            return image, relations, path
        else:
            return image, None, path

stage = 'val'
val = PSGClsDataset(stage=stage)

cfg = get_cfg()
# add project-specific config (e.g., TensorMask) here if you're not running a model in detectron2's core library
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
# Find a model from detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
predictor = DefaultPredictor(cfg)
anno = MetadataCatalog.get(cfg.DATASETS.TRAIN[0]).thing_classes

pred_list = []
gt_list = []

for n in tqdm(range(1,len(val)+1)):
# for n in range(5):
    n=n-1
    image, relations, p = val.__getitem__(n)

    im = cv2.imread(p)
    outputs = predictor(im)

    # v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
    # out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    # cv2.imwrite(f'{n}tmp.jpg', out.get_image()[:, :, ::-1]) 

    instances = outputs['instances'].pred_classes
    ins = torch.unique(instances)
    N = 5
    if len(ins)>N:
        ins = ins[:N]

    texts = []
    # for id, r in enumerate(PREDICATES_mod2):
    #     texts.append((id+6,f'action of {r}'))
    # for i in ins:
    #     for id, r in enumerate(PREDICATES_mod2):
    #         texts.append((id+6,f'a {anno[i]} is {r}'))
    for i in ins:
        for id, r in enumerate(PREDICATES):
            for j in ins:
                texts.append((id+6,f'{anno[i]} {r} {anno[j]}'))

    text_inputs =  clip.tokenize([x[1] for x in texts]).to(device)

    # Calculate features
    with torch.no_grad():
        image_features = model.encode_image(image)
        text_features = model.encode_text(text_inputs)

    # Pick the top k most similar labels for the image
    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)
    similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
    values, indices = similarity[0].topk(10)

    ind=[]
    for x in [texts[x][0] for x in indices]:
        if x not in ind:
            if len(ind)<3:
                ind.append(x)

    backup = [14,21,11]
    if len(ind)<3:
        miss = 3-len(ind)
        for i in range(miss):
            ind.append(backup[i])


    pred_list.append(ind)
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
