import torch
import torch.nn as nn
import torch.nn.functional as F

def dice_loss(inputs, targets, smooth=1.):
    iflat = inputs.contiguous().view(-1)
    tflat = targets.contiguous().view(-1)
    intersection = (iflat * tflat).sum()

    A_sum = torch.sum(tflat * iflat)
    B_sum = torch.sum(tflat * tflat)
    
    return 1 - ((2. * intersection + smooth) / (A_sum + B_sum + smooth))


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
        d_loss = dice_loss(inputs, targets)
        loss = self.lambda_focal * f_loss + self.lambda_dice * d_loss

        return loss
