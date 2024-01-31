import torch
import torch.nn as nn
import torch.nn.functional as F


# # source: https://gist.github.com/weiliu620/52d140b22685cf9552da4899e2160183
# def dice_loss(input, target, smooth=1.):
#     iflat = inputs.contiguous().view(-1)
#     tflat = targets.contiguous().view(-1)
#     intersection = (iflat * tflat).sum()

#     A_sum = torch.sum(tflat * iflat)
#     B_sum = torch.sum(tflat * tflat)
    
#     return 1 - ((2. * intersection + smooth) / (A_sum + B_sum + smooth))


# source: https://github.com/kevinzakka/pytorch-goodies/blob/master/losses.py
def dice_loss(logits, true, smooth=1., eps=1e-7):
    """Computes the Sørensen–Dice loss.

    Note that PyTorch optimizers minimize a loss. In this
    case, we would like to maximize the dice loss so we
    return the negated dice loss.

    Args:
        true: a tensor of shape [B, 1, H, W].
        logits: a tensor of shape [B, C, H, W]. Corresponds to
            the raw output or logits of the model.
        eps: added to the denominator for numerical stability.

    Returns:
        dice_loss: the Sørensen–Dice loss.
    """
    num_classes = logits.shape[1]
    if num_classes == 1:
        true_1_hot = torch.eye(num_classes + 1)[true.squeeze(1)]
        true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
        true_1_hot_f = true_1_hot[:, 0:1, :, :]
        true_1_hot_s = true_1_hot[:, 1:2, :, :]
        true_1_hot = torch.cat([true_1_hot_s, true_1_hot_f], dim=1)
        pos_prob = torch.sigmoid(logits)
        neg_prob = 1 - pos_prob
        probas = torch.cat([pos_prob, neg_prob], dim=1)
    else:
        true_1_hot = torch.eye(num_classes, device="cuda")[true.squeeze(1)]
        true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
        probas = F.softmax(logits, dim=1)
    true_1_hot = true_1_hot.type(logits.type())
    dims = (0,) + tuple(range(2, true.ndimension()))
    intersection = torch.sum(probas * true_1_hot, dims)
    cardinality = torch.sum(probas + true_1_hot, dims)
    dice_loss = (2. * intersection / (cardinality + eps)).mean()
    return (1 - dice_loss)


# source: https://pytorch.org/vision/main/_modules/torchvision/ops/focal_loss.html
def focal_loss(inputs, targets, alpha, gamma, reduction="none"):
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = inputs * targets + (1 - inputs) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    if reduction == "mean":
        loss = loss.mean()
    elif reduction == "sum":
        loss = loss.sum()

    return loss

class DiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        intersection = (inputs * targets).sum()                            
        dice = (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)  
        
        return 1 - dice

# the final loss function is a combination of focal and dice loss, both for locator and refiner trainings
class FocalDiceLoss(nn.Module):
    def __init__(self, focal_alpha=0.65, focal_gamma=2.0, lambda_focal=1.75, lambda_dice=1.0, apply_sigmoid=False):
        super(FocalDiceLoss, self).__init__()
        self.alpha = focal_alpha
        self.gamma = focal_gamma
        self.lambda_focal = lambda_focal
        self.lambda_dice = lambda_dice
        self.apply_sigmoid = apply_sigmoid

    def forward(self, inputs, targets):
        targets = targets.to(inputs.dtype)

        if self.apply_sigmoid:
            inputs = torch.sigmoid(inputs)
        
        f_loss = focal_loss(inputs, targets, alpha=self.alpha, gamma=self.gamma, reduction="mean")
        d_loss = DiceLoss().forward(inputs, targets);
        loss = self.lambda_focal * f_loss + self.lambda_dice * d_loss # values used --> (1.75 * focal) + (1 * dice)

        return loss
