import torch


class BinaryDSCLoss(torch.nn.Module):
    r"""
    Creates a criterion that optimizes a multi-class Self-adjusting Dice Loss
    ("Dice Loss for Data-imbalanced NLP Tasks" paper)
    Args:
        alpha (float): a factor to push down the weight of easy examples
        gamma (float): a factor added to both the nominator and the denominator for smoothing purposes
        reduction (string): Specifies the reduction to apply to the output:
            ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction will be applied,
            ``'mean'``: the sum of the output will be divided by the number of
            elements in the output, ``'sum'``: the output will be summed.
    Shape:
        - logits: `(N, C)` where `N` is the batch size and `C` is the number of classes.
        - targets: `(N)` where each value is in [0, C - 1]
    """

    def __init__(self, alpha: float = 1.0, smooth: float = 1.0, reduction: str = "mean") -> None:
        super().__init__()
        self.alpha = alpha
        self.smooth = smooth
        self.reduction = reduction

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        # probs = torch.gather(probs, dim=1, index=targets.unsqueeze(1))
        #
        # targets = targets.unsqueeze(dim=1)
        pos_mask = (targets == 1).float()
        neg_mask = (targets == 0).float()

        pos_weight = pos_mask * ((1 - probs) ** self.alpha) * probs
        pos_loss = 1 - (2 * pos_weight + self.smooth) / (pos_weight + 1 + self.smooth)

        neg_weight = neg_mask * ((1 - probs) ** self.alpha) * probs
        neg_loss = 1 - (2 * neg_weight + self.smooth) / (neg_weight + self.smooth)

        loss = pos_loss + neg_loss
        loss = loss.mean()
        return loss


class MultiDSCLoss(torch.nn.Module):
    r"""
    Creates a criterion that optimizes a multi-class Self-adjusting Dice Loss
    ("Dice Loss for Data-imbalanced NLP Tasks" paper)
    Args:
        alpha (float): a factor to push down the weight of easy examples
        gamma (float): a factor added to both the nominator and the denominator for smoothing purposes
        reduction (string): Specifies the reduction to apply to the output:
            ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction will be applied,
            ``'mean'``: the sum of the output will be divided by the number of
            elements in the output, ``'sum'``: the output will be summed.
    Shape:
        - logits: `(N, C)` where `N` is the batch size and `C` is the number of classes.
        - targets: `(N)` where each value is in [0, C - 1]
    """

    def __init__(self, alpha: float = 2.0, smooth: float = 1.0, reduction: str = "mean"):
        super().__init__()
        self.alpha = alpha
        self.smooth = smooth
        self.reduction = reduction

    def forward(self, logits, targets):
        probs = torch.softmax(logits, dim=1)
        probs = torch.gather(probs, dim=1, index=targets.unsqueeze(1))

        probs_with_factor = ((1 - probs) ** self.alpha) * probs
        loss = 1 - (2 * probs_with_factor + self.smooth) / (probs_with_factor + 1 + self.smooth)

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        elif self.reduction == "none" or self.reduction is None:
            return loss
        else:
            raise NotImplementedError(f"Reduction `{self.reduction}` is not supported.")


class BinaryDiceLoss(torch.nn.Module):
    """Dice loss of binary class
    Args:
        smooth: A float number to smooth loss, and avoid NaN error, default: 1
        p: Denominator value: \sum{x^p} + \sum{y^p}, default: 2
        predict: A tensor of shape [N, *]
        target: A tensor of shape same with predict
        reduction: Reduction method to apply, return mean over batch if 'mean',
            return sum if 'sum', return a tensor of shape [N,] if 'none'
    Returns:
        Loss tensor according to arg reduction
    Raise:
        Exception if unexpected reduction
    """
    def __init__(self, smooth=1, alpha=2, reduction='mean'):
        super(BinaryDiceLoss, self).__init__()
        self.smooth = smooth
        self.p = alpha
        self.reduction = reduction

    def forward(self, predict, target):
        assert predict.shape[0] == target.shape[0], "predict & target batch size don't match"
        predict = predict.contiguous().view(predict.shape[0], -1)
        target = target.contiguous().view(target.shape[0], -1)

        num = torch.sum(torch.mul(predict, target), dim=1) + self.smooth
        den = torch.sum(predict.pow(self.p) + target.pow(self.p), dim=1) + self.smooth
        print(1-num)
        print(den)
        loss = 1 - num / den

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        elif self.reduction == 'none':
            return loss
        else:
            raise Exception('Unexpected reduction {}'.format(self.reduction))



class BinaryDiceLoss2(torch.nn.Module):
    """DiceLoss implemented from 'Dice Loss for Data-imbalanced NLP Tasks'
    Useful in dealing with unbalanced data
    Add softmax automatically
    """

    def __init__(self, alpha=0.5, smooth=0.5):
        super(BinaryDiceLoss2, self).__init__()
        self.alpha = alpha
        self.gamma = smooth

    def forward(self, y_pred, y_true, reduction='mean'):
        """
        :param y_pred: [N, C, ]
        :param y_true: [N, C, ]
        :param reduction: 'mean' or 'sum'
        """
        batch_size = y_true.size(0)
        y_pred = y_pred.contiguous().view(batch_size, -1)
        y_true = y_true.contiguous().view(batch_size, -1)

        numerator = torch.sum(2 * torch.pow((1 - y_pred), self.alpha) * y_pred * y_true, dim=1) + self.gamma
        denominator = torch.sum(torch.pow((1 - y_pred), self.alpha)  * y_pred + y_true, dim=1) + self.gamma
        loss = 1 - (numerator / denominator)
        if reduction == 'mean':
            return loss.mean()
        elif reduction == 'sum':
            return loss.sum()
        else:
            return loss


y = torch.tensor([1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0], dtype=torch.float)
p = torch.tensor([0, 0.3, 0.3, 0.2, 0.9, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0], dtype=torch.float)
print(y.shape,y)
print(torch.unsqueeze(y,dim=1).shape,y.unsqueeze(dim=1))
loss = BinaryDSCLoss(alpha=0.6,smooth=0.001)
loss2 = BinaryDiceLoss(alpha=0.6,smooth=0.001)
loss3 = BinaryDiceLoss2(alpha=0.6,smooth=0.001)
print(loss(p,y))
print(loss2(p,y))
print(loss3(p,y))
