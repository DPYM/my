import torch
import torch.nn as nn 

class BPRLoss(nn.Module):
    def __init__(self,l2_reg=0.0):
        super().__init__()
        self.l2_reg=l2_reg

    def forward(self,pos_score,neg_score,user_emb=None,pos_emb=None,neg_emb=None):
        if pos_score.dim()==1 and neg_score.dim()==2:
            pos_score=pos_score.unsqueeze(1)
        
        diff=torch.sigmoid(pos_score-neg_score)
        loss=-torch.log(diff+1e-8).mean()

        if self.l2_reg>0 and user_emb is not None and pos_emb is not None and neg_emb is not None:
            l2_loss=(user_emb.pow(2).sum(dim=1).mean()
                     +pos_emb.pow(2).sum(dim=1).mean()
                     +neg_emb.pow(2).sum(dim=1).mean()
            )
            loss=loss+self.l2_reg*l2_loss

        return loss