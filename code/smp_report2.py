import io
import json
import logging
import os
import torch
import random
import numpy as np
import torchvision.transforms as trn
from torchvision.transforms import InterpolationMode
from torch.utils.data import DataLoader
import torchvision.transforms.functional as F
from PIL import Image, ImageFile
from torch.utils.data import Dataset
from os.path import exists
import matplotlib.pyplot as plt
import argparse

# to fix "OSError: image file is truncated"
ImageFile.LOAD_TRUNCATED_IMAGES = True

#1e-4 # 2e-5,4e-6
parser = argparse.ArgumentParser()
parser.add_argument('--seg_lambda', type=float, default=1e-4)
parser.add_argument('--final', type=float, default=0)
args = parser.parse_args()

CLASSES=[
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train',
    'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag',
    'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite',
    'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
    'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon',
    'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
    'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant',
    'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
    'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
    'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
    'hair drier', 'toothbrush', 'banner', 'blanket', 'bridge', 'cardboard',
    'counter', 'curtain', 'door-stuff', 'floor-wood', 'flower', 'fruit',
    'gravel', 'house', 'light', 'mirror-stuff', 'net', 'pillow', 'platform',
    'playingfield', 'railroad', 'river', 'road', 'roof', 'sand', 'sea',
    'shelf', 'snow', 'stairs', 'tent', 'towel', 'wall-brick', 'wall-stone',
    'wall-tile', 'wall-wood', 'water-other', 'window-blind', 'window-other',
    'tree-merged', 'fence-merged', 'ceiling-merged', 'sky-other-merged',
    'cabinet-merged', 'table-merged', 'floor-other-merged', 'pavement-merged',
    'mountain-merged', 'grass-merged', 'dirt-merged', 'paper-merged',
    'food-other-merged', 'building-other-merged', 'rock-merged',
    'wall-other-merged', 'rug-merged', 'background'
]

class Convert:
    def __init__(self, mode='RGB'):
        self.mode = mode

    def __call__(self, image):
        return image.convert(self.mode)

def rgb2id(color):
    if isinstance(color, np.ndarray) and len(color.shape) == 3:
        if color.dtype == np.uint8:
            color = color.astype(np.int32)
        return color[:, :, 0] + 256 * color[:, :, 1] + 256 * 256 * color[:, :, 2]
    return int(color[0] + 256 * color[1] + 256 * 256 * color[2])

class PSGClsDataset(Dataset):
    def __init__(
        self,
        stage,
        root='./data/coco/',
        H=512,
        W=512,
    ):
        super(PSGClsDataset, self).__init__()
        with open('./data/psg/psg_cls_basic.json') as f:
            dataset = json.load(f)
        with open('./data/psg/psg_cls_advanced.json') as f:
            dataset_pan = json.load(f)
        
        self.stage = stage
        if args.final==1 and self.stage=='train':
            self.imglist = [
                d for d in dataset['data']
                if (d['image_id'] in dataset[f'train_image_ids']) or (d['image_id'] in dataset[f'val_image_ids'])
            ]
        else:
            self.imglist = [
                d for d in dataset['data']
                if d['image_id'] in dataset[f'{stage}_image_ids']
            ]

        self.root = root
        self.low = [6,7,8,9,10,12,13,15,17,18,19,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,50,51,52,53,54,55]
        self.N_THINGS = 133
        self.N_PREDICATES = 56
        self.H = H
        self.W = W
        self.data_pan = dataset_pan["data"]
        self.imglist = [ g for g in self.imglist if exists('./data/coco/'+g['file_name'])]
        if self.stage!='test':
            self.imglist = [ p for p in self.imglist if exists('./data/coco/panoptic_'+p['file_name'].replace('jpg','png'))]
        
        self.imglist_new = []
        for image in self.imglist:
            if any([ relation in self.low for relation in image['relations']]):
                for _ in range(2):
                    self.imglist_new.append(image)
            elif random.random()<0.1:
                self.imglist_new.append(image)

    def __len__(self):
        if self.stage=='train':
            return len(self.imglist_new)
        else:
            return len(self.imglist)

    def __getitem__(self, index):

        if self.stage=='train':
            sample = self.imglist_new[index]
        else:
            sample = self.imglist[index]
        path = os.path.join(self.root, sample['file_name'])
        path_pan = os.path.join(self.root,'panoptic_'+sample['file_name']).replace('jpg','png')
        hflip = random.random()

        try:
            with open(path, 'rb') as f:
                content = f.read()
                filebytes = content
                buff = io.BytesIO(filebytes)
                image = Image.open(buff).convert('RGB')
        except Exception as e:
            logging.error('Error, cannot read [{}]'.format(path))
            raise e

        if self.stage=='train':
            # panoptic+augmentation
            try:
                with open(path_pan, 'rb') as f:
                    content = f.read()
                    filebytes = content
                    buff = io.BytesIO(filebytes)
                    panoptic = Image.open(buff).convert('RGB')
            except Exception as e:
                logging.error('Error, cannot read [{}]'.format(path_pan))
                raise e
            tfs = trn.Compose([
                    Convert('RGB'),
                    trn.Resize((int(1.4*self.H), int(1.4*self.W)),
                    interpolation=InterpolationMode.NEAREST)])
            image = tfs(image)
            panoptic = tfs(panoptic)
            i, j, h, w = trn.RandomCrop.get_params(image, (self.H, self.W))
            image = F.crop(image, i, j, h, w)
            panoptic = F.crop(panoptic, i, j, h, w)         
            if hflip>0.5:
                image = F.hflip(image)
                panoptic = F.hflip(panoptic)
            if random.random()<0.4:
                tf_color = trn.ColorJitter(brightness=.5, hue=.3)
                image = tf_color(image)
                inv = trn.RandomInvert()
                image = inv(image)
        
        elif self.stage=='val':
            # panoptic
            try:
                with open(path_pan, 'rb') as f:
                    content = f.read()
                    filebytes = content
                    buff = io.BytesIO(filebytes)
                    panoptic = Image.open(buff).convert('RGB')
            except Exception as e:
                logging.error('Error, cannot read [{}]'.format(path_pan))
                raise e
            tfs = trn.Compose([
                    Convert('RGB'), 
                    trn.Resize((int(self.H), int(self.W)),
                    interpolation=InterpolationMode.NEAREST)])
            image = tfs(image)
            panoptic = tfs(panoptic)

        else:
            tfs = trn.Compose([
                    Convert('RGB'), 
                    trn.Resize((int(self.H), int(self.W)),
                    interpolation=InterpolationMode.NEAREST)])
            image = tfs(image)
            panoptic = None 

        soft_label = torch.Tensor(self.N_PREDICATES)
        soft_label.fill_(0)
        soft_label[sample['relations']] = 1

        if panoptic:
            image_pan_arr = np.asarray(panoptic)
            ids = rgb2id(image_pan_arr)
            sample_pan = [d for d in self.data_pan if d["file_name"]==sample["file_name"]][0]
            seg_info = sample_pan["segments_info"]
            category_ids = ids.copy()

            memory = {}
            for i in range(category_ids.shape[0]):
                for j in range(category_ids.shape[1]):
                    tmp = category_ids[i,j]
                    if tmp in memory.keys():
                        category_ids[i,j] = memory[tmp]
                    else:
                        pxl = [x for x in seg_info if x['id']==tmp]
                        if len(pxl)>0:
                            pxl = pxl[0]
                            category_ids[i,j] = pxl['category_id']
                        else:
                            # the pixel id is 0, no category found, set to a new category 133
                            category_ids[i,j] = self.N_THINGS
                        memory[tmp] = category_ids[i,j]
            panoptic = torch.from_numpy(category_ids)
        else:
            panoptic = torch.zeros(1)

        mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
        tfs_last = trn.Compose([trn.ToTensor(), trn.Normalize(mean, std)])
        image = tfs_last(image)

        return image, panoptic, soft_label

# test dataset
# train_dataset = PSGClsDataset(stage='val')
# image, panoptic, soft_label = train_dataset.__getitem__(80)
# print(image, panoptic, soft_label)
# print(panoptic.shape, 'relations:', soft_label.shape)
# print('N:', len(CLASSES))

# for x in torch.unique(panoptic):
#     print(x.data.numpy(), CLASSES[x])

# plt.imshow(image)
# plt.savefig('tmp1.png')
# plt.imshow(panoptic)
# plt.savefig('tmp2.png')

#################################################################

torch.cuda.set_device(2)

# hyperparameter
learning_rate = 0.00003
# momentum = 0.8
weight_decay = 0.005
epoch = 12
k = 3

size_H=512
size_W=512
batch_size=8

seg_lambda=args.seg_lambda
model_name='dp'
savename = f'SMP2_{model_name}_H{size_H}W{size_W}_e{epoch}_bs{batch_size}_lr{learning_rate}_seglambda{seg_lambda}'

# loading dataset
train_dataset = PSGClsDataset(H=size_H, W=size_W,stage='train')
train_dataloader = DataLoader(train_dataset,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=8)

val_dataset = PSGClsDataset(H=size_H, W=size_W,stage='val')
val_dataloader = DataLoader(val_dataset,
                            batch_size=batch_size,
                            shuffle=False,
                            num_workers=8)

test_dataset = PSGClsDataset(H=size_H, W=size_W,stage='test')
test_dataloader = DataLoader(test_dataset,
                             batch_size=batch_size,
                             shuffle=False,
                             num_workers=8)
print('Data Loaded...', flush=True)



from torchvision import models
from torchvision.models.segmentation.deeplabv3 import DeepLabHead
from torchvision.models.segmentation import DeepLabV3_ResNet50_Weights
import segmentation_models_pytorch as smp
import time
from tqdm import tqdm

aux_params = {'classes': 56, 'pooling':'avg'}

if model_name=='deeplab':
    def createDeepLabv3(outputchannels):

        model = models.segmentation.deeplabv3_resnet50(weights = DeepLabV3_ResNet50_Weights.DEFAULT,
                                                        progress=True)
        model.classifier = DeepLabHead(2048, outputchannels)
        # Set the model in training mode
        model.train()
        return model
    model = createDeepLabv3(134)
elif model_name=='FPN':
    model = smp.FPN(
        encoder_name='resnet50',
        decoder_segmentation_channels=256,
        classes=134,
        activation='softmax2d',
        aux_params=aux_params)
elif model_name=='dp':
    model = smp.DeepLabV3(
        encoder_name='resnet50',
        encoder_depth=5,
        # encoder_weights='imagenet',
        encoder_weights='ssl',
        decoder_channels=512,
        in_channels=3,
        classes=134, 
        activation='logsoftmax',
        upsampling=8,
        aux_params=aux_params)
    # model=smp.DeepLabV3Plus(
    #     encoder_name='resnet50', 
    #     encoder_depth=5, 
    #     encoder_weights='imagenet', 
    #     encoder_output_stride=16, 
    #     decoder_channels=512, 
    #     decoder_atrous_rates=(12, 24, 36), 
    #     in_channels=3, 
    #     classes=134, 
    #     activation='logsoftmax', 
    #     upsampling=4, 
    #     aux_params=aux_params)
model.cuda()

print('Model Loaded...', flush=True)

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
# criterion = torch.nn.CrossEntropyLoss(reduction='sum')
criterion_dice = smp.losses.DiceLoss(mode='multiclass', from_logits=False)
criterion_ce = torch.nn.CrossEntropyLoss(reduction='sum')

def cosine_annealing(step, total_steps, lr_max, lr_min):
    return lr_min + (lr_max - lr_min) * 0.5 * (1 + np.cos(np.pi*step/total_steps))

scheduler = torch.optim.lr_scheduler.LambdaLR(
    optimizer,
    lr_lambda=lambda step: cosine_annealing(
        step,
        epoch * len(train_dataloader),
        1,
        1e-7 / learning_rate,
    ))

# scheduler = torch.optim.lr_scheduler.StepLR(
#     optimizer,
#     step_size=15*len(train_dataloader),
#     gamma=0.5)

best_val_recall = 0.0
begin_epoch = time.time()
log = {'Epoch':[], 'Train_loss':[], 'Val_loss':[], 'mR':[], 'loss_val_seg':[], 'loss_val_cls':[], 'IOU':[], 'acc':[]}
metrics = {}
t=0
for ep in range(0, epoch):
# for ep in range(0, 5):

    loss_train = []
    train_dataiter = iter(train_dataloader)

    for train_step in tqdm(range(1, len(train_dataiter) + 1)):
    # for train_step in tqdm(range(1, 5)):

        model.train()
        optimizer.zero_grad()

        image, panoptic, soft_label = next(train_dataiter)
        data = image.cuda()
        panoptic_gt = panoptic.type(torch.LongTensor).cuda()
        predicate_gt = soft_label.cuda()
        if model_name == 'deeplab':
            panoptic_pred = model(data)['out']
        else:
            panoptic_pred, predicate = model(data)
        c1 = (criterion_ce(panoptic_pred, panoptic_gt)+criterion_dice(panoptic_pred, panoptic_gt))*seg_lambda
        weights_bce = torch.ones(predicate.shape).cuda()
        for x in train_dataset.low:
            weights_bce[:,x]=2.5
        c2 = torch.nn.functional.binary_cross_entropy_with_logits(predicate, predicate_gt, reduction='sum',weight=weights_bce)
        loss = c1+c2

        l1_norm = sum(p.abs().sum() for p in model.parameters())
        # print('L1',l1_norm)
        # loss += 0.001*l1_norm
        
        loss.backward()
        optimizer.step()
        scheduler.step()
        loss_train.append(loss.detach().cpu().numpy()) 

    metrics['train_loss'] = np.mean(loss_train)

    model.eval()
    loss_val = []
    loss_val_seg = []
    loss_val_cls = []
    mean_IOU = []
    mean_acc = []
    mean_recall_seg = []
    pred_list, gt_list = [], []
    t+=1
    tt=0

    with torch.no_grad():
        for image, panoptic, soft_label in iter(val_dataloader):

            data = image.cuda()
            panoptic_gt = panoptic.type(torch.LongTensor).cuda()
            predicate_gt = soft_label.cuda()
            if model_name == 'deeplab':
                panoptic_pred = model(data)['out']
            else:
                panoptic_pred, predicate = model(data)

            loss1 = (criterion_ce(panoptic_pred, panoptic_gt)+criterion_dice(panoptic_pred, panoptic_gt))*seg_lambda
            weights_bce = torch.ones(predicate.shape).cuda()
            for x in train_dataset.low:
                weights_bce[:,x]=2.5
            loss2 = torch.nn.functional.binary_cross_entropy_with_logits(predicate, predicate_gt, reduction='sum',weight=weights_bce)
            tmp = loss2+loss1
            loss_val.append(tmp.detach().cpu().numpy())
            loss_val_seg.append(loss1.detach().cpu().numpy())
            loss_val_cls.append(loss2.detach().cpu().numpy())

            panoptic_pre = torch.argmax(panoptic_pred,dim=1)
            panoptic_pre = panoptic_pre.type(torch.LongTensor).cuda()
            panoptic_pre = torch.nn.functional.one_hot(panoptic_pre, 134)
            panoptic_pre = torch.permute(panoptic_pre, (0,3,1,2)).cuda()            

            panoptic_gt_onehot = torch.nn.functional.one_hot(panoptic_gt, 134)
            panoptic_gt_onehot = torch.permute(panoptic_gt_onehot, (0,3,1,2)).cuda()
            panoptic_gt_onehot = panoptic_gt_onehot

            I = torch.sum(torch.multiply(panoptic_gt_onehot, panoptic_pre))
            U = panoptic_gt_onehot+panoptic_pre
            U = torch.sum(U>0)
            IOU = I/U
            mean_IOU.append(IOU.detach().cpu().numpy())

            TPN = (panoptic_gt_onehot == panoptic_pre)
            acc = torch.sum(TPN)/(panoptic_pre.shape[0]*panoptic_pre.shape[1]*panoptic_pre.shape[2]*panoptic_pre.shape[3])
            mean_acc.append(acc.detach().cpu().numpy())

            TP = torch.sum(panoptic_pre[panoptic_gt_onehot==1])
            P = torch.sum(panoptic_gt_onehot)
            recall = TP/P
            mean_recall_seg.append(recall.detach().cpu().numpy())

            if t%10==1:
                if tt==0:
                    n = 5
                    print('file',val_dataset.imglist[n])
                    class_labels = np.argmax(panoptic_pre[n,:,:,:].cpu().numpy(), axis=0)
                    # print(class_labels.shape,'size')
                    # things = np.unique(class_labels)
                    # print([ CLASSES[t] for t in things])
                    plt.imshow(class_labels)
                    plt.savefig(f'tmp{t}_{args.seg_lambda}.png')
                    if t==1:
                        gt = np.argmax(panoptic_gt_onehot[n,:,:,:].cpu().numpy(), axis=0)
                        gt_things = np.unique(gt)
                        print('True Things:',[ CLASSES[t] for t in gt_things])
                        plt.imshow(gt)
                        plt.savefig(f'tmp0.png')                    
            tt+=1

            # gather prediction and gt
            prob = torch.sigmoid(predicate)
            for x in train_dataset.low:
                prob[:,x ]=20*prob[:,x]
            pred = torch.topk(prob.data, k)[1]
            pred = pred.cpu().detach().tolist()
            pred_list.extend(pred)
            for soft_label in soft_label:
                gt_label = (soft_label == 1).nonzero(as_tuple=True)[0]\
                            .cpu().detach().tolist()
                gt_list.append(gt_label)

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

    metrics['mean_recall'] = meanrecall
    metrics['val_loss'] = np.mean(loss_val)
    metrics['mean_IOU'] = np.mean(mean_IOU)
    metrics['mean_acc'] = np.mean(mean_acc)
    metrics['mean_recall_seg'] = np.mean(mean_recall_seg)
    metrics['loss_val_seg'] = np.mean(loss_val_seg)
    metrics['loss_val_cls'] = np.mean(loss_val_cls)

    print(
    '{} | Epoch {:3d} | Time {:5d}s - Train Loss {:.1f} | Val Loss {:.0f} | acc {:.3f} | IOU {:.3f} | recall_seg {:.3f} | mR {:.2f} | seg:{:.1f}/cls:{:.1f} r{:.2f} |'
    .format(savename, (ep + 1), int(time.time() - begin_epoch),
            metrics['train_loss'], metrics['val_loss'], metrics['mean_acc'], 
            metrics['mean_IOU'], metrics['mean_recall_seg'], 100*metrics['mean_recall'],
            metrics['loss_val_seg'],metrics['loss_val_cls'],metrics['loss_val_seg']/metrics['loss_val_cls']), 
            flush=True)

    log['Epoch'].append((ep + 1))
    log['Train_loss'].append(float(metrics['train_loss']))
    log['Val_loss'].append(float(metrics['val_loss']))
    log['mR'].append(float((100.0 * metrics['mean_recall'])))
    log['IOU'].append(float(metrics['mean_IOU']))
    log['acc'].append(float(metrics['mean_acc']))
    log['loss_val_seg'].append(float(metrics['loss_val_seg']))
    log['loss_val_cls'].append(float(metrics['loss_val_cls']))

    # save model
    if metrics['mean_recall'] >= best_val_recall:
        torch.save(model.state_dict(), f'./checkpoints/{savename}_best.ckpt')
        best_val_recall = metrics['mean_recall']

print('Training Completed...', flush=True)

def save_json(d, file):
    with open(file, 'w') as f:
        json.dump(d, f)
save_json(log, f'./logs/{savename}.json')

# saving result!
print('Loading Best Ckpt...', flush=True)
checkpoint = torch.load(f'checkpoints/{savename}_best.ckpt')
model.load_state_dict(checkpoint)
model.eval()

pred_list = []
with torch.no_grad():
    for image, _, soft_label in test_dataloader:
        data = image.cuda()
        predicate_gt = soft_label.cuda()
        if model_name == 'deeplab':
            panoptic_pred = model(data)['out']
        else:
            panoptic_pred, predicate = model(data)

        prob = torch.sigmoid(predicate)
        for x in train_dataset.low:
            prob[:,x ]=20*prob[:,x]
        pred = torch.topk(prob.data, k)[1]
        pred = pred.cpu().detach().tolist()
        pred_list.extend(pred)
result = pred_list

with open(f'results/{savename}_{best_val_recall}.txt', 'w') as writer:
    for label_list in result:
        a = [str(x) for x in label_list]
        save_str = ' '.join(a)
        writer.writelines(save_str + '\n')
print('Result Saved!', flush=True)

