import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


class Evaluator:
    def __init__(
        self,
        net: nn.Module,
        k: int,
        low: list
    ):
        self.net = net
        self.k = k
        # self.low = low
        self.low = [6,7,8,9,10,12,13,15,17,18,19,24,25,26,27,28,29,30,31,32,33,34,35,38,39,40,41,42,43,44,45,50,51,52,53,54,55]
        self.w = 1.1
    def eval_recall(
        self,
        data_loader: DataLoader,
    ):
        self.net.eval()
        loss_avg = 0.0
        pred_list, gt_list = [], []
        with torch.no_grad():
            for data, target in data_loader:
                data = data.cuda()
                logits = self.net(data)
                prob = torch.sigmoid(logits)
                target = target.cuda()
                loss = F.binary_cross_entropy(prob, target, reduction='sum')
                loss_avg += float(loss.data)
                # gather prediction and gt
                for x in self.low:
                    prob[:,x ]=self.w*prob[:,x]
                pred = torch.topk(prob.data, self.k)[1]
                pred = pred.cpu().detach().tolist()
                pred_list.extend(pred)
                for soft_label in target:
                    gt_label = (soft_label == 1).nonzero(as_tuple=True)[0]\
                                .cpu().detach().tolist()
                    gt_list.append(gt_label)
        
        import matplotlib.pyplot as plt
        # print(pred_list[:18])
        # print(gt_list[:18])
        t1 = np.zeros(56)
        t = [item for sublist in pred_list for item in sublist]
        for tt in t:
            t1[tt]+=1
        plt.figure()
        plt.bar(x=np.arange(56),height=t1)
        plt.savefig('tmp_pred.png')
        plt.close()
        t2 = np.zeros(56)
        t = [item for sublist in gt_list for item in sublist]
        for tt in t:
            t2[tt]+=1
        plt.figure()
        plt.bar(x=np.arange(56),height=t2)
        plt.savefig('tmp_gt.png')
        plt.close()

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

        scores = score_list[:, 1] / score_list[:, 0]
        plt.figure()
        plt.bar(x=np.arange(len(scores))+6,height=scores)
        plt.savefig('tmp_score.png')
        plt.close()

        metrics = {}
        metrics['test_loss'] = loss_avg / len(data_loader)
        metrics['mean_recall'] = meanrecall

        return metrics

    def submit(
        self,
        data_loader: DataLoader,
    ):
        self.net.eval()

        pred_list = []
        with torch.no_grad():
            for data, target in data_loader:
                data = data.cuda()
                logits = self.net(data)
                prob = torch.sigmoid(logits)
                for x in self.low:
                    prob[:,x ]=self.w*prob[:,x]
                pred = torch.topk(prob.data, self.k)[1]
                pred = pred.cpu().detach().tolist()
                pred_list.extend(pred)
        return pred_list
