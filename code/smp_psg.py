import os
import torch
import time
import segmentation_models_pytorch as smp
from dataset_new import PSGClsDataset_new
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.nn.functional as F
import numpy as np
import json

os.makedirs('./segmentation', exist_ok=True)
torch.cuda.set_device(2)

# hyperparameter
learning_rate = 0.00005
momentum = 0.8
weight_decay = 0.0001
epoch = 100
k = 3

size_H=224
size_W=224
batch_size=16
seg_weight=5

savename = f'smp_deeplab_H{size_H}W{size_W}_e{epoch}_bs{batch_size}_lr{learning_rate}_seg{seg_weight}_wd{weight_decay}'

# loading dataset
train_dataset = PSGClsDataset_new(H=size_H, W=size_W,stage='train')
train_dataloader = DataLoader(train_dataset,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=8)

val_dataset = PSGClsDataset_new(H=size_H, W=size_W,stage='val')
val_dataloader = DataLoader(val_dataset,
                            batch_size=batch_size,
                            shuffle=False,
                            num_workers=8)

test_dataset = PSGClsDataset_new(H=size_H, W=size_W,stage='test')
test_dataloader = DataLoader(test_dataset,
                             batch_size=batch_size,
                             shuffle=False,
                             num_workers=8)
print('Data Loaded...', flush=True)

# loading model
# aux_params = {'classes': 56, 'dropout':0.2, 'activation':'sigmoid'}
aux_params = {'classes': 56, 'dropout':0.2, 'pooling':'max'}
# model = smp.Unet(encoder_name='resnet50', encoder_depth=3, decoder_channels=(512, 512, 256), classes=134, aux_params=aux_params)
model = smp.FPN(encoder_name='resnet50', decoder_segmentation_channels=256, classes=134, aux_params=aux_params, activation='softmax2d')
model.cuda()
# print(model)
print('Model Loaded...', flush=True)


optimizer = torch.optim.SGD(
    model.parameters(),
    learning_rate,
    momentum=momentum,
    weight_decay=weight_decay,
    nesterov=True,
)

# optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20*len(train_dataloader), gamma=0.1)
# scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=0.0001, last_epoch=-1)

criterion = smp.losses.DiceLoss(mode='multiclass', from_logits=False, ignore_index=133)
# criterion = smp.losses.FocalLoss(mode='multiclass', ignore_index=133)

## criterion = smp.losses.SoftCrossEntropyLoss(reduction='mean', smooth_factor=None, ignore_index=133, dim=1)
# https://smp.readthedocs.io/en/latest/_modules/segmentation_models_pytorch/losses/soft_ce.html#SoftCrossEntropyLoss

# train
print('Start Training...', flush=True)
best_val_recall = 0.0
begin_epoch = time.time()
log = {'Epoch':[], 'Train_loss':[], 'Test_loss':[], 'mR':[], 'test_loss_seg':[], 'test_loss_cls':[], 'IOU':[], 'acc':[]}

for epoch in range(0, epoch):

    loss_train = 0.0
    train_dataiter = iter(train_dataloader)

    for train_step in tqdm(range(1, len(train_dataiter) + 1)):

        model.train()
        optimizer.zero_grad()

        batch = next(train_dataiter)
        data = batch['data'].cuda()
        panoptic_gt = batch['class_label'].type(torch.LongTensor).cuda()
        predicate_gt = batch['soft_label'].cuda()
        # forward
        panoptic, predicate = model(data)
        loss = seg_weight*criterion(panoptic, panoptic_gt)+F.binary_cross_entropy_with_logits(predicate, predicate_gt)

        loss.backward()
        optimizer.step()
        scheduler.step()
        loss_train += loss
    
        # break

    metrics = {}
    metrics['train_loss'] = loss_train

    model.eval()
    loss1_val = 0.0
    loss2_val = 0.0
    loss_val = 0.0
    iou = 0.0
    acc = 0.0
    pred_list, gt_list = [], []

    with torch.no_grad():
        for batch in iter(val_dataloader):

            data = batch['data'].cuda()
            panoptic_gt = batch['class_label'].type(torch.LongTensor).cuda()
            predicate_gt = batch['soft_label'].cuda()

            panoptic, predicate = model(data)
            loss1 = seg_weight*criterion(panoptic, panoptic_gt)
            loss2 = F.binary_cross_entropy_with_logits(predicate, predicate_gt)
            loss1_val += loss1
            loss2_val += loss2
            loss_val += loss1+loss2

            panoptic_pre = panoptic>0.5
            panoptic_pre = panoptic_pre.type(torch.LongTensor).cuda()
            panoptic_gt_onehot = F.one_hot(panoptic_gt, 134)
            panoptic_gt_onehot = torch.permute(panoptic_gt_onehot, (0,3,1,2)).cuda()
            panoptic_gt_onehot = panoptic_gt_onehot[:,:133,:,:]
            panoptic_pre = panoptic_pre[:,:133,:,:]

            I = torch.sum(torch.multiply(panoptic_gt_onehot, panoptic_pre))
            U = panoptic_gt_onehot+panoptic_pre
            U = torch.sum(U>0)
            IOU = I.item()/U.item()
            flag = (panoptic_gt_onehot == panoptic_pre)
            acc = torch.sum(flag)/(panoptic_pre.shape[0]*panoptic_pre.shape[1]*panoptic_pre.shape[2]*panoptic_pre.shape[3])
            iou += IOU
            acc += acc
        
            # gather prediction and gt
            prob = torch.sigmoid(predicate)
            pred = torch.topk(prob.data, k)[1]
            pred = pred.cpu().detach().tolist()
            pred_list.extend(pred)
            for soft_label in batch['soft_label']:
                gt_label = (soft_label == 1).nonzero(as_tuple=True)[0]\
                            .cpu().detach().tolist()
                gt_list.append(gt_label)

        # compute mean recall
        score_list = np.zeros([56, 2], dtype=int)

        # print('gt:', gt_list)
        # print('pred:', pred_list)

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
        metrics['IOU'] = iou / len(val_dataloader)
        metrics['acc'] = acc / len(val_dataloader)
        metrics['test_loss'] = loss_val / len(val_dataloader)
        metrics['test_loss_seg'] = loss1_val / len(val_dataloader)
        metrics['test_loss_cls'] = loss2_val / len(val_dataloader)

        print(
        '{} | Epoch {:3d} | Time {:5d}s \n - Train Loss {:.4f} | mR {:.2f} | Test Loss {:.3f} | seg:{:.3f}/cls:{:.3f} | acc {:.3f} | IOU {:.3f} '
        .format(savename, (epoch + 1), int(time.time() - begin_epoch),
                metrics['train_loss'], 100*metrics['mean_recall'], metrics['test_loss'], metrics['test_loss_seg'], 
                metrics['test_loss_cls'], metrics['acc'], 
                metrics['IOU']), 
                flush=True)

    log['Epoch'].append((epoch + 1))
    log['Train_loss'].append(metrics['train_loss'])
    log['Test_loss'].append(metrics['test_loss'])
    log['mR'].append(100.0 * metrics['mean_recall'])
    log['test_loss_seg'].append(metrics['test_loss_seg'])
    log['test_loss_cls'].append(metrics['test_loss_cls'])
    log['IOU'].append(metrics['IOU'])
    log['acc'].append(metrics['acc'])

    # break
    if metrics['mean_recall'] >= best_val_recall:
        torch.save(model.state_dict(), f'./checkpoints/{savename}_best.ckpt')
        best_val_recall = metrics['mean_recall']

def save_json(d, file):
    with open(file, 'w') as f:
        json.dump(d, f)

save_json(log, f'./logs/{savename}.json')
print('Training log saved!')


model.eval()
print('Loading Best Ckpt...', flush=True)
checkpoint = torch.load(f'checkpoints/{savename}_best.ckpt')
model.load_state_dict(checkpoint)

pred_list = []
with torch.no_grad():
    for batch in test_dataloader:
        data = batch['data'].cuda()
        _, predicate = model(data)
        prob = torch.sigmoid(predicate)
        pred = torch.topk(prob.data, k)[1]
        pred = pred.cpu().detach().tolist()
        pred_list.extend(pred)

result =  pred_list
# save into the file
with open(f'results/{savename}_{np.round(best_val_recall,2)}.txt', 'w') as writer:
    for label_list in result:
        a = [str(x) for x in label_list]
        save_str = ' '.join(a)
        writer.writelines(save_str + '\n')
print('Result Saved!', flush=True)
