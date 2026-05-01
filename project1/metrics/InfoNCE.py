import torch
import torch.nn as nn

class InfoNCELoss(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, pos_score, neg_scores, in_batch_neg_scores=None):
        pos_score = pos_score / self.temperature
        neg_scores = neg_scores / self.temperature

        all_neg = neg_scores
        if in_batch_neg_scores is not None:
            in_batch_neg_scores = in_batch_neg_scores / self.temperature
            all_neg = torch.cat([neg_scores, in_batch_neg_scores], dim=1)

        all_scores = torch.cat([pos_score.unsqueeze(1), all_neg], dim=1)

        log_probs = torch.log_softmax(all_scores, dim=1)
        loss = -log_probs[:, 0].mean()

        return loss
