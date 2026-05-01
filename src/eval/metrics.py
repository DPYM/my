"""Evaluation metrics for DualGuard experiments."""

import torch
import numpy as np
from sklearn.metrics import roc_auc_score, log_loss


@torch.no_grad()
def evaluate_model(model, data_dict, split, device="cuda",
                   batch_size=16384, max_batches=None):
    """Return (AUC, LogLoss) on a data split."""
    from src.training.trainers import make_dataloader

    model.eval()
    loader = make_dataloader(data_dict, split, batch_size)
    n_dense = data_dict["n_dense"]

    preds, ys = [], []
    for i, batch in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        if n_dense > 0:
            xs, xd, yb = batch
            xs, xd = xs.to(device), xd.to(device)
            p = model(xs, xd).cpu().numpy()
        else:
            xs, yb = batch
            xs = xs.to(device)
            p = model(xs).cpu().numpy()
        preds.append(p)
        ys.append(yb.numpy())

    p_all = np.concatenate(preds)
    y_all = np.concatenate(ys)
    return float(roc_auc_score(y_all, p_all)), float(log_loss(y_all, np.clip(p_all, 1e-7, 1 - 1e-7)))
