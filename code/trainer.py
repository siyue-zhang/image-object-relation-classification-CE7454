from sys import maxsize
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from smooth import SmoothBCEwLogits


def cosine_annealing(step, total_steps, lr_max, lr_min):
    return lr_min + (lr_max -
                     lr_min) * 0.5 * (1 + np.cos(step / total_steps * np.pi))

def warm_cosine_annealing(step, total_steps, lr_max, lr_min):
    if step<0.1*total_steps:
        return step/(0.1*total_steps)*(lr_max-lr_min)+lr_min
    else:
        return lr_min + (lr_max -
                        lr_min) * 0.5 * (1 + np.cos((step-0.1*total_steps) / 0.9*total_steps * np.pi))

class BaseTrainer:
    def __init__(self,
                 net: nn.Module,
                 train_loader: DataLoader,
                 learning_rate: float = 0.1,
                 momentum: float = 0.9,
                 weight_decay: float = 0.0005,
                 epochs: int = 100,
                 scheduler: str = 'cosine',
                 optimizer: str = 'SDG',
                 low:list =[],
                 high:list =[]) -> None:
        self.net = net
        self.train_loader = train_loader
        self.scheduler = scheduler
        self.optimizer = optimizer
        self.low=low
        self.high=high

        if self.optimizer=='SDG':
            self.optimizer = torch.optim.SGD(
                net.parameters(),
                learning_rate,
                momentum=momentum,
                weight_decay=weight_decay,
                nesterov=True,
            )
        elif self.optimizer == 'Adam':
            self.optimizer = torch.optim.Adam(
                net.parameters(),
                lr=learning_rate,
                weight_decay=weight_decay,
            )
        elif self.optimizer == 'AdamW':
            self.optimizer = torch.optim.AdamW(
                net.parameters(),
                lr=learning_rate,
                weight_decay=weight_decay,
                amsgrad=True
            )

        if self.scheduler=='cosine':
            self.scheduler = torch.optim.lr_scheduler.LambdaLR(
                self.optimizer,
                lr_lambda=lambda step: cosine_annealing(
                    step,
                    epochs * len(train_loader),
                    1,  # since lr_lambda computes multiplicative factor
                    # 1e-6 / learning_rate,
                    1e-7 / learning_rate,
                ),
            )
        elif self.scheduler=='warmcosine':
            self.scheduler = torch.optim.lr_scheduler.LambdaLR(
                self.optimizer,
                lr_lambda=lambda step: warm_cosine_annealing(
                    step,
                    epochs * len(train_loader),
                    1,  # since lr_lambda computes multiplicative factor
                    # 1e-6 / learning_rate,
                    1e-8 / learning_rate,
                ),
            )
        elif self.scheduler=='step':
            self.scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=10*len(self.train_loader),
                # step_size=30*len(self.train_loader),
                gamma=0.5)
        elif self.scheduler=='wcosine':
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                self.optimizer,
                T_0=5, 
                T_mult=2, 
                eta_min=learning_rate/100, 
                last_epoch=-1)
        elif self.scheduler=='cyclic':
            self.scheduler = torch.optim.lr_scheduler.CyclicLR(
                self.optimizer,
                base_lr=learning_rate/10, 
                max_lr=learning_rate,
                step_size_up=10,
                mode="triangular2",
                cycle_momentum=False)


    def train_epoch(self):
        self.net.train()  # enter train mode

        loss_avg = 0.0
        train_dataiter = iter(self.train_loader)

        for train_step in tqdm(range(1, len(train_dataiter) + 1)):
            # for train_step in tqdm(range(1, 5)):

            data, target = next(train_dataiter)
            data = data.cuda()
            target = target.cuda()
            # forward
            logits = self.net(data)
            weights = torch.ones(logits.shape).cuda()
            for x in self.low:
                weights[:,x]=1.2
            for y in self.high:
                weights[:,y]=1
            # l = SmoothBCEwLogits(weight=weights, reduction='sum', smoothing=0.2)
            # loss = l(logits,target)
            loss = F.binary_cross_entropy_with_logits(logits,
                                                      target,
                                                      weight=weights,
                                                      reduction='sum'
                                                      )
            # l1_norm = sum(p.abs().sum() for p in self.net.parameters())
            # l2_norm = sum(p.pow(2).sum() for p in self.net.parameters())
            # l1_lambda = 0.001 #res101            
            # l1_lambda = 0.0001 #res50
            # l2_lambda = 0.001 #res50
            # # print('l1', l1_norm)
            # # print('l2', l2_norm)
            # loss += l1_norm*l1_lambda
            #  + l2_norm*l2_lambda

            # backward
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()

            # exponential moving average, show smooth values
            with torch.no_grad():
                loss_avg = loss_avg * 0.8 + float(loss) * 0.2

        metrics = {}
        metrics['train_loss'] = loss_avg

        return metrics
