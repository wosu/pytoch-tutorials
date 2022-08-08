'''
https://github.com/shuxinyin/NLP-Loss-Pytorch
paper:https://arxiv.org/abs/1708.02002v2
code:https://github.com/clcarwin/focal_loss_pytorch

https://github.com/CoinCheung/pytorch-loss

alpha范围:[0,1]，论文实验中最佳为0.25 一般设为
gamma范围：[0,5] 论文实验中最佳为2.0

通过一系列调参，得到 α=0.25, γ=2（在他的模型上）的效果最好。注意在他的任务中，
正样本是属于少数样本，也就是说，本来正样本难以“匹敌”负样本，但经过 (1−ŷ )γ 和 ŷγ 的“操控”后，也许形势还逆转了，还要对正样本降权。

不过我认为这样调整只是经验结果，理论上很难有一个指导方案来决定 α 的值，如果没有大算力调参，倒不如直接让 α=0.5（均等）
'''
import random
import time

from torch import nn
import torch
import torch.nn.functional as F


class BinaryFocalLoss(nn.Module):
    '''
    loss=-alpha*(1-p_t)^gamma *log(p_t)
    '''
    def __init__(self,alpha=0.25, gamma=2, reduction='mean',need_logit=False, **kwargs):
        super(BinaryFocalLoss,self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.smooth = 1e-6
        self.need_logit = need_logit

    def forward(self,output,target):
        '''
        :param output: shape:(batch_size,1)
        :param target: shape:(batch_size,1)
        :return:
        '''
        # 预测结果先进行softmax,因为是二分类这里可以用sigmoid
        prob = output
        if self.need_logit:
            prob = torch.sigmoid(output)
        #prob = torch.sigmoid(output)
        # 将预测结果平滑，压缩到区间(smooth,1-smooth)
        prob = torch.clamp(prob,self.smooth,1.0-self.smooth)
        pos_mask = (target == 1.0).float()
        neg_mask = (target == 0.0).float()

        pos_weight = (pos_mask * torch.pow(1 - prob, self.gamma)).detach()
        pos_loss = -pos_weight * torch.log(prob)
        neg_weight = (neg_mask * torch.pow(prob, self.gamma)).detach()
        neg_loss = -self.alpha * neg_weight * torch.log(1-prob)

        loss = pos_loss + neg_loss
        loss = loss.mean()
        return loss


class MultiFocalLoss(nn.Module):
    """
    Focal_Loss= -1*alpha*((1-pt)**gamma)*log(pt)
    Args:
        num_class: number of classes
        alpha: class balance factor shape=[num_class, ]
        gamma: hyper-parameter
        reduction: reduction type
    """

    def __init__(self, num_class, alpha=None, gamma=2, reduction='mean'):
        super(MultiFocalLoss, self).__init__()
        self.num_class = num_class
        self.gamma = gamma
        self.reduction = reduction
        self.smooth = 1e-4
        self.alpha = alpha
        if alpha is None:
            self.alpha = torch.ones(num_class, ) - 0.5
        elif isinstance(alpha, (int, float)):
            self.alpha = torch.as_tensor([alpha] * num_class)
        elif isinstance(alpha, (list, np.ndarray)):
            self.alpha = torch.as_tensor(alpha)
        if self.alpha.shape[0] != num_class:
            raise RuntimeError('the length not equal to number of class')

    def forward(self, logit, target):
        # assert isinstance(self.alpha,torch.Tensor)\
        alpha = self.alpha.to(logit.device)
        prob = F.softmax(logit, dim=1)

        if prob.dim() > 2:
            # used for 3d-conv:  N,C,d1,d2 -> N,C,m (m=d1*d2*...)
            N, C = logit.shape[:2]
            prob = prob.view(N, C, -1)
            prob = prob.transpose(1, 2).contiguous()  # [N,C,d1*d2..] -> [N,d1*d2..,C]
            prob = prob.view(-1, prob.size(-1))  # [N,d1*d2..,C]-> [N*d1*d2..,C]

        ori_shp = target.shape
        target = target.view(-1, 1)

        prob = prob.gather(1, target).view(-1) + self.smooth  # avoid nan
        logpt = torch.log(prob)
        # alpha_class = alpha.gather(0, target.squeeze(-1))
        alpha_weight = alpha[target.squeeze().long()]
        loss = -alpha_weight * torch.pow(torch.sub(1.0, prob), self.gamma) * logpt

        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'none':
            loss = loss.view(ori_shp)

        return loss


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class FocalLoss(nn.Module):
    def __init__(self, gamma=0, alpha=None, size_average=True,need_logit=False):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha,(float,int)): self.alpha = torch.Tensor([alpha,1-alpha])
        if isinstance(alpha,list): self.alpha = torch.Tensor(alpha)
        self.size_average = size_average
        self.need_logit = need_logit

    def forward(self, input, target):
        if input.dim()>2:
            input = input.view(input.size(0),input.size(1),-1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1,2)    # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1,input.size(2))   # N,H*W,C => N*H*W,C
        target = target.view(-1,1)
        logpt = input
        if self.need_logit:
            logpt = F.log_softmax(input)
        logpt = logpt.gather(1,target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type()!=input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0,target.data.view(-1))
            logpt = logpt * Variable(at)

        loss = -1 * (1-pt)**self.gamma * logpt
        if self.size_average: return loss.mean()
        else: return loss.sum()


def loss_self(inputs,targets,alpha=1,gamma=2,need_logit=False):
    bce_los = F.binary_cross_entropy(inputs,targets,reduction='none')
    if need_logit:
        bce_los = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
    pt = torch.exp(-bce_los)  # prevents nans when probability 0
    F_loss = alpha * (1 - pt) ** gamma * bce_los
    return F_loss.mean()

class FocalLossV1(nn.Module):

    def __init__(self,
                 alpha=0.25,
                 gamma=2,
                 reduction='mean',):
        super(FocalLossV1, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.crit = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, logits, label):
        '''
        Usage is same as nn.BCEWithLogits:
            >>> criteria = FocalLossV1()
            >>> logits = torch.randn(8, 19, 384, 384)
            >>> lbs = torch.randint(0, 2, (8, 19, 384, 384)).float()
            >>> loss = criteria(logits, lbs)
        '''
        probs = torch.sigmoid(logits)
        coeff = torch.abs(label - probs).pow(self.gamma).neg()
        log_probs = torch.where(logits >= 0,
                F.softplus(logits, -1, 50),
                logits - F.softplus(logits, 1, 50))
        log_1_probs = torch.where(logits >= 0,
                -logits + F.softplus(logits, -1, 50),
                -F.softplus(logits, 1, 50))
        loss = label * self.alpha * log_probs + (1. - label) * (1. - self.alpha) * log_1_probs
        loss = loss * coeff

        if self.reduction == 'mean':
            loss = loss.mean()
        if self.reduction == 'sum':
            loss = loss.sum()
        return


def py_sigmoid_focal_loss(pred,
                          target,
                          weight=None,
                          gamma=2.0,
                          alpha=0.25,
                          reduction='mean',
                          avg_factor=None):
    pred_sigmoid = pred.sigmoid()
    target = target.type_as(pred)
    pt = (1 - pred_sigmoid) * target + pred_sigmoid * (1 - target)
    focal_weight = (alpha * target + (1 - alpha) *
                    (1 - target)) * pt.pow(gamma)
    loss = F.binary_cross_entropy_with_logits(
        pred, target, reduction='none') * focal_weight
    loss = weight_reduce_loss(loss, weight, reduction, avg_factor)
    return loss

# y = torch.tensor([1,1,1,1,1,1,1,1,1,0,0,0,0,0,0],dtype=torch.float)
# p = torch.tensor([1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],dtype=torch.float)
# loss = BinaryFocalLoss(need_logit=False, alpha=0.75, gamma=4)
# loss2 = FocalLoss(alpha=0.75,gamma=4)
# print(loss(p,y))
# print(loss(p,y))

criteria = FocalLossV1()
logits = torch.randn(8, 19, 384, 384)
lbs = torch.randint(0, 2, (8, 19, 384, 384)).float()
loss = criteria(logits, lbs)
print(logits,lbs)
print(loss)

# start_time = time.time()
# maxe = 0
# for i in range(1000):
#     x = torch.rand(12800,1)*random.randint(1,10)
#     l = torch.rand(12800).ge(0.1).long()
#     print(x.shape,x,l)
#     output0 = FocalLoss(gamma=0,need_logit=True)(x,l)
#     o2 = loss_self(gamma=0,need_logit=True)(x,l)
#     print(output0)
#     print(o2)
#     break
#     output1 = nn.CrossEntropyLoss()(x,l)
#     a = output0.data[0]
#     b = output1.data[0]
#     if abs(a-b)>maxe: maxe = abs(a-b)
# print('time:',time.time()-start_time,'max_error:',maxe)


# start_time = time.time()
# maxe = 0
# for i in range(100):
#     x = torch.rand(128,1000,8,4)*random.randint(1,10)
#     x = Variable(x.cuda())
#     l = torch.rand(128,8,4)*1000    # 1000 is classes_num
#     l = l.long()
#     l = Variable(l.cuda())
#
#     output0 = FocalLoss(gamma=0)(x,l)
#     output1 = nn.NLLLoss2d()(F.log_softmax(x),l)
#     a = output0.data[0]
#     b = output1.data[0]
#     if abs(a-b)>maxe: maxe = abs(a-b)
# print('time:',time.time()-start_time,'max_error:',maxe)