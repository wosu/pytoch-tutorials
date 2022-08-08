'''
https://github.com/shuxinyin/NLP-Loss-Pytorch/blob/master/unbalanced_loss/GHM_loss.py
https://github.com/DHPO/GHM_Loss.pytorch

'''
#
# import torch
# from torch import nn
# import torch.nn.functional as F
#
#
# class GHM_Loss(nn.Module):
#     def __init__(self, bins=10, alpha=0.5):
#         '''
#         bins: split to n bins
#         alpha: hyper-parameter
#         '''
#         super(GHM_Loss, self).__init__()
#         self._bins = bins
#         self._alpha = alpha
#         self._last_bin_count = None
#
#     def _g2bin(self, g):
#         return torch.floor(g * (self._bins - 0.0001)).long()
#
#     def _custom_loss(self, x, target, weight):
#         raise NotImplementedError
#
#     def _custom_loss_grad(self, x, target):
#         raise NotImplementedError
#
#     def forward(self, x, target):
#         g = torch.abs(self._custom_loss_grad(x, target)).detach()
#
#         bin_idx = self._g2bin(g)
#
#         bin_count = torch.zeros((self._bins))
#         for i in range(self._bins):
#             bin_count[i] = (bin_idx == i).sum().item()
#
#         N = (x.size(0) * x.size(1))
#
#         if self._last_bin_count is None:
#             self._last_bin_count = bin_count
#         else:
#             bin_count = self._alpha * self._last_bin_count + (1 - self._alpha) * bin_count
#             self._last_bin_count = bin_count
#
#         nonempty_bins = (bin_count > 0).sum().item()
#
#         gd = bin_count * nonempty_bins
#         gd = torch.clamp(gd, min=0.0001)
#         beta = N / gd
#
#         return self._custom_loss(x, target, beta[bin_idx])
#
#
# class GHMC_Loss(GHM_Loss):
#     '''
#         GHM_Loss for classification
#     '''
#
#     def __init__(self, bins, alpha,need_logit=False):
#         super(GHMC_Loss, self).__init__(bins, alpha)
#         self.need_logit = need_logit
#
#     def _custom_loss(self, x, target, weight):
#         if self.need_logit:
#             return F.binary_cross_entropy_with_logits(x, target, weight=weight)
#         return F.binary_cross_entropy(x, target, weight=weight)
#
#     def _custom_loss_grad(self, x, target):
#         if self.need_logit:
#             return torch.sigmoid(x).detach() - target
#         return x.detach() - target
#
#
# class GHMR_Loss(GHM_Loss):
#     '''
#         GHM_Loss for regression
#     '''
#
#     def __init__(self, bins, alpha, mu):
#         super(GHMR_Loss, self).__init__(bins, alpha)
#         self._mu = mu
#
#     def _custom_loss(self, x, target, weight):
#         d = x - target
#         mu = self._mu
#         loss = torch.sqrt(d * d + mu * mu) - mu
#         N = x.size(0) * x.size(1)
#         return (loss * weight).sum() / N
#
#     def _custom_loss_grad(self, x, target):
#         d = x - target
#         mu = self._mu
#         return d / torch.sqrt(d * d + mu * mu)
#
#



import torch
from torch import nn
import torch.nn.functional as F


class GHM_Loss(nn.Module):
    def __init__(self, bins, alpha):
        super(GHM_Loss, self).__init__()
        self._bins = bins
        self._alpha = alpha
        self._last_bin_count = None

    def _g2bin(self, g):
        return torch.floor(g * (self._bins - 0.0001)).long()

    def _custom_loss(self, x, target, weight):
        raise NotImplementedError

    def _custom_loss_grad(self, x, target):
        raise NotImplementedError

    def forward(self, x, target):
        g = torch.abs(self._custom_loss_grad(x, target)).detach()

        bin_idx = self._g2bin(g)

        bin_count = torch.zeros((self._bins))
        for i in range(self._bins):
            bin_count[i] = (bin_idx == i).sum().item()

        N = (x.size(0) * x.size(1))

        if self._last_bin_count is None:
            self._last_bin_count = bin_count
        else:
            bin_count = self._alpha * self._last_bin_count + (1 - self._alpha) * bin_count
            self._last_bin_count = bin_count

        nonempty_bins = (bin_count > 0).sum().item()

        gd = bin_count * nonempty_bins
        gd = torch.clamp(gd, min=0.0001)
        beta = N / gd

        return self._custom_loss(x, target, beta[bin_idx])


class GHMC_Loss(GHM_Loss):
    # 分类损失
    def __init__(self, bins, alpha,need_logit=False):
        super(GHMC_Loss, self).__init__(bins, alpha)
        self.need_logit = need_logit

    def _custom_loss(self, x, target, weight):
        if self.need_logit:
            return F.binary_cross_entropy_with_logits(x, target, weight=weight)
        return F.binary_cross_entropy(x,target,weight=weight)

    def _custom_loss_grad(self, x, target):
        if self.need_logit:
            return torch.sigmoid(x).detach() - target
        return x.detach() - target


class GHMR_Loss(GHM_Loss):
    # 回归损失
    def __init__(self, bins, alpha, mu):
        super(GHMR_Loss, self).__init__(bins, alpha)
        self._mu = mu

    def _custom_loss(self, x, target, weight):
        d = x - target
        mu = self._mu
        loss = torch.sqrt(d * d + mu * mu) - mu
        N = x.size(0) * x.size(1)
        return (loss * weight).sum() / N

    def _custom_loss_grad(self, x, target):
        d = x - target
        mu = self._mu
        return d / torch.sqrt(d * d + mu * mu)


if __name__ == '__main__':
    # 这个损失函数不需要自己进行sigmoid
    # output = torch.FloatTensor(
    #     [
    #         [0.0550, -0.5005],
    #         [0.7060, 1.1139]
    #     ]
    # )
    output = torch.FloatTensor(
        [
            [0, 1],
            [1, 1]
        ]
    )
    label = torch.FloatTensor(
        [
            [1, 0],
            [0, 1]
        ]
    )
    loss_func = GHMC_Loss(bins=10, alpha=0.75,need_logit=False)
    loss = loss_func(output, label)
    print(loss)
    y = torch.tensor([1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0], dtype=torch.float)
    p = torch.tensor([0, 0.3, 0.3, 0.2, 0.9, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0], dtype=torch.float)

    p_t = torch.cat((torch.unsqueeze(p,dim=1),torch.unsqueeze(1-p,dim=1)),dim=1)
    y_t = torch.cat((torch.unsqueeze(y,dim=1),torch.unsqueeze(1-y,dim=1)),dim=1)
    print(p_t.size(),p_t)
    print(y_t)
    print(loss_func(p_t,y_t))