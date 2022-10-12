import argparse
import os
from sched import scheduler
import time
import json

import torch
import torch.nn as nn
from dataset import PSGClsDataset, PredictDataset
from balanced_dataset2 import balancePSGClsDataset,balancePredictDataset
from evaluator import Evaluator
from torch.utils.data import DataLoader
from torchvision.models import resnet50, ResNet50_Weights
from torchvision.models import resnet101, ResNet101_Weights
from torchvision.models import swin_b, Swin_B_Weights
from swin_v2 import swin_v2_b, Swin_V2_B_Weights
from torchvision.models import efficientnet_b6, EfficientNet_B6_Weights
import numpy as np
from trainer import BaseTrainer

parser = argparse.ArgumentParser()
parser.add_argument('--model_name', type=str, default='res50')
parser.add_argument('--epoch', type=int, default=36)
parser.add_argument('--scheduler', type=str, default='cosine')
parser.add_argument('--optimizer', type=str, default='SDG')
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--momentum', type=float, default=0.9)
parser.add_argument('--weight_decay', type=float, default=0.0005)
parser.add_argument('--final', type=int, default=0)
parser.add_argument('--note', type=str, default='')


args = parser.parse_args()

torch.cuda.set_device(2)

savename = f'B_{args.model_name}_e{args.epoch}_{args.optimizer}_{args.scheduler}_lr{args.lr}_bs{args.batch_size}_wd{args.weight_decay}'
if args.optimizer=='SGD':
    savename +=f'_m{args.momentum}'
savename += f'{args.note}'
if args.final==1:
    savename += 'FINAL'
os.makedirs('./checkpoints', exist_ok=True)
os.makedirs('./results', exist_ok=True)
os.makedirs('./logs', exist_ok=True)

# low = [6,7,8,9,10,12,13,15,17,18,19,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,50,51,52,53,54,55]
low=[6,7,8,9,10,13,15,17,18,19,24,25,27,28,29,30,31,32,33,34,35,38,39,40,41,43,44,50,52,53]
# high=[11,14,16,21,23]
# low = [ i for i in np.arange(6,56) if i not in high]
high = [14,21]

# loading dataset
train_dataset = balancePSGClsDataset(stage='train',low=low)
train_dataloader = DataLoader(train_dataset,
                              batch_size=args.batch_size,
                              shuffle=True,
                              num_workers=8)

val_dataset = PSGClsDataset(stage='val')
val_dataloader = DataLoader(val_dataset,
                            batch_size=32,
                            shuffle=False,
                            num_workers=8)

test_dataset = PSGClsDataset(stage='test')
test_dataloader = DataLoader(test_dataset,
                             batch_size=32,
                             shuffle=False,
                             num_workers=8)

predict_dataset = balancePredictDataset(low=low)
predict_dataloader = DataLoader(predict_dataset,
                              batch_size=args.batch_size,
                              shuffle=True,
                              num_workers=8)
                            
print('Data Loaded...', flush=True)

# def set_parameter_requires_grad(model, feature_extracting):
#     if feature_extracting:
#         for param in model.parameters():
#             param.requires_grad = False

# loading model
if args.model_name == 'res50':
    model = resnet50(weights=ResNet50_Weights.DEFAULT)
    # set_parameter_requires_grad(model, True)
    model.fc = torch.nn.Sequential(
        nn.Dropout(0.2),
        nn.Linear(2048, 56))
elif args.model_name == 'res101':
    model = resnet101(weights=ResNet101_Weights.DEFAULT)
    model.fc = torch.nn.Sequential(
        nn.Dropout(0.1),
        nn.Linear(2048, 56))
elif args.model_name == 'swin':
    model = swin_b(weights=Swin_B_Weights.DEFAULT)
    model.head = torch.nn.Sequential(
        # nn.Dropout(0.1),
        nn.BatchNorm1d(1024),
        nn.Linear(1024, 56))
elif args.model_name == 'swin2':
    model = swin_v2_b(weights=Swin_V2_B_Weights.DEFAULT)
    model.head = torch.nn.Sequential(
        nn.BatchNorm1d(1024),
        nn.Linear(1024, 56))
elif args.model_name == 'eff':
    model = efficientnet_b6(weights=EfficientNet_B6_Weights.DEFAULT)
    model.classifier = torch.nn.Sequential(
        nn.Dropout(0.2),
        nn.Linear(2304, 56))
else:
    print('Model not found!')

model.cuda()

# def initialize_weights(m):
#   if isinstance(m, nn.Conv2d):
#       nn.init.kaiming_uniform_(m.weight.data,nonlinearity='relu')
#       if m.bias is not None:
#           nn.init.constant_(m.bias.data, 0)
#   elif isinstance(m, nn.BatchNorm2d):
#       nn.init.constant_(m.weight.data, 1)
#       nn.init.constant_(m.bias.data, 0)
#   elif isinstance(m, nn.Linear):
#       nn.init.kaiming_uniform_(m.weight.data)
#       nn.init.constant_(m.bias.data, 0)
# model.apply(initialize_weights)

print('Model Loaded...', flush=True)

# loading trainer
if args.final==1:
    trainer = BaseTrainer(model,
                        predict_dataloader,
                        learning_rate=args.lr,
                        momentum=args.momentum,
                        weight_decay=args.weight_decay,
                        epochs=args.epoch,
                        scheduler=args.scheduler,
                        optimizer=args.optimizer,
                        low=low,
                        high=high)
else:
    trainer = BaseTrainer(model,
                        train_dataloader,
                        learning_rate=args.lr,
                        momentum=args.momentum,
                        weight_decay=args.weight_decay,
                        epochs=args.epoch,
                        scheduler=args.scheduler,
                        optimizer=args.optimizer,
                        low=low,
                        high=high)
evaluator = Evaluator(model, k=3, low=low)

# train!
print('Start Training...', flush=True)
begin_epoch = time.time()
best_val_recall = 0.0
log = {'Epoch':[], 'Train_loss':[], 'Test_loss':[], 'mR':[]}

for epoch in range(0, args.epoch):
# for epoch in range(0, 25):
    train_metrics = trainer.train_epoch()
    val_metrics = evaluator.eval_recall(val_dataloader)

    # show log
    print(
        '{} | Epoch {:3d} | Time {:5d}s | Train Loss {:.4f} | Test Loss {:.3f} | mR {:.2f}'
        .format(savename, (epoch + 1), int(time.time() - begin_epoch),
                train_metrics['train_loss'], val_metrics['test_loss'],
                100.0 * val_metrics['mean_recall']),
        flush=True)

    log['Epoch'].append((epoch + 1))
    log['Train_loss'].append(train_metrics['train_loss'])
    log['Test_loss'].append(val_metrics['test_loss'])
    log['mR'].append(100.0 * val_metrics['mean_recall'])

    # save model
    if val_metrics['mean_recall'] >= best_val_recall:
        torch.save(model.state_dict(), f'./checkpoints/{savename}_best.ckpt')
        best_val_recall = val_metrics['mean_recall']

print('Training Completed...', flush=True)

def save_json(d, file):
    with open(file, 'w') as f:
        json.dump(d, f)
save_json(log, f'./logs/{savename}.json')

# saving result!
print('Loading Best Ckpt...', flush=True)
checkpoint = torch.load(f'checkpoints/{savename}_best.ckpt')
model.load_state_dict(checkpoint)
test_evaluator = Evaluator(model, k=3,low=low)
check_metrics = test_evaluator.eval_recall(val_dataloader)
if best_val_recall == check_metrics['mean_recall']:
    print('Successfully load best checkpoint with acc {:.2f}'.format(
        100 * best_val_recall),
          flush=True)
else:
    print('Fail to load best checkpoint')
result = test_evaluator.submit(test_dataloader)

# save into the file
with open(f'results/{savename}_{best_val_recall}.txt', 'w') as writer:
    for label_list in result:
        a = [str(x) for x in label_list]
        save_str = ' '.join(a)
        writer.writelines(save_str + '\n')
print('Result Saved!', flush=True)
