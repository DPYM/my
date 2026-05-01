import torch
import torch.nn as nn

class FocalLoss(nn.Module):
    def __init__(self,alpha,gamma):
        super().__init__()
        self.alpha=alpha
        self.gamma=gamma

    def forward(self,score,label):
        bce=nn.functional.binary_cross_entropy_with_logits(score,label,reduction='none')
        pt=torch.exp(-bce)
        alpha=self.alpha*label+(1-self.alpha)*(1-label)
        f_loss=alpha*((1-pt)**self.gamma)*bce
        return f_loss.mean()