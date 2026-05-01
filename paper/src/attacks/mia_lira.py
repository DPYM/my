"""
LiRA (Likelihood Ratio Attack) for Membership Inference.

Reference: Carlini et al., "Membership Inference Attacks From First Principles",
           IEEE S&P 2022.

Protocol (offline variant):
  1. Train K shadow models on random 50 % subsets of the training data.
  2. For each shadow model, collect confidence scores (logits / probabilities)
     of every sample under multiple dropout-enabled forward passes.
  3. Fit IN and OUT Gaussian distributions per sample.
  4. On the TARGET model, compute the LiRA score for each query sample.
  5. Report TPR @ low FPR and MIA AUC.
"""

import torch
import numpy as np
from sklearn.metrics import roc_auc_score


def collect_confidences(model, X_sparse, X_dense, n_aug=30, device="cuda"):
    """
    Collect n_aug confidence vectors for a single sample.
    Keeps dropout active to emulate an ensemble.
    """
    model.train()  # keep dropout on
    if X_dense is not None:
        xs = X_sparse.unsqueeze(0).to(device)
        xd = X_dense.unsqueeze(0).to(device)
    else:
        xs = X_sparse.unsqueeze(0).to(device)

    confs = []
    with torch.no_grad():
        for _ in range(n_aug):
            if X_dense is not None:
                p = model(xs, xd).item()
            else:
                p = model(xs).item()
            confs.append(p)
    return np.array(confs, dtype=np.float32)


def fit_lira_distributions(shadow_conf_in, shadow_conf_out):
    """
    Fit diagonal Gaussian to IN and OUT confidence vectors.

    Parameters
    ----------
    shadow_conf_in  : list of np.array (n_samples,) — confidences when sample was IN
    shadow_conf_out : list of np.array (n_samples,) — confidences when sample was OUT

    Returns
    -------
    mu_in, sigma_in   : floats
    mu_out, sigma_out : floats
    """
    all_in  = np.concatenate([c.ravel() for c in shadow_conf_in])
    all_out = np.concatenate([c.ravel() for c in shadow_conf_out])
    return (np.mean(all_in), np.std(all_in) + 1e-8,
            np.mean(all_out), np.std(all_out) + 1e-8)


def lira_score(conf_vector, mu_in, sigma_in, mu_out, sigma_out):
    """Log-likelihood-ratio for a single confidence vector."""
    ll_in  = -0.5 * np.mean(((conf_vector - mu_in)  / sigma_in)  ** 2)
    ll_out = -0.5 * np.mean(((conf_vector - mu_out) / sigma_out) ** 2)
    return ll_in - ll_out


def run_lira_attack(model, in_loader, out_loader,
                    mu_in, sigma_in, mu_out, sigma_out,
                    n_aug=30, device="cuda", max_samples=2000):
    """
    Evaluate LiRA against a target model.

    Returns
    -------
    mia_auc    : float
    tpr_at_001 : float  (TPR @ FPR = 0.01)
    """
    scores_in, scores_out = [], []
    n_dense = False

    # Determine if model takes dense input
    for batch in in_loader:
        n_dense = len(batch) == 3
        break

    # --- IN samples ---
    count = 0
    for batch in in_loader:
        if count >= max_samples:
            break
        if n_dense:
            xs, xd, yb = batch
        else:
            xs, yb = batch
            xd = None

        for i in range(len(xs)):
            if count >= max_samples:
                break
            conf = collect_confidences(model, xs[i],
                                       xd[i] if xd is not None else None,
                                       n_aug, device)
            scores_in.append(lira_score(conf, mu_in, sigma_in, mu_out, sigma_out))
            count += 1

    # --- OUT samples ---
    count = 0
    for batch in out_loader:
        if count >= max_samples:
            break
        if n_dense:
            xs, xd, yb = batch
        else:
            xs, yb = batch
            xd = None

        for i in range(len(xs)):
            if count >= max_samples:
                break
            conf = collect_confidences(model, xs[i],
                                       xd[i] if xd is not None else None,
                                       n_aug, device)
            scores_out.append(lira_score(conf, mu_in, sigma_in, mu_out, sigma_out))
            count += 1

    s_in  = np.array(scores_in)
    s_out = np.array(scores_out)

    # AUC
    labels = np.concatenate([np.ones_like(s_in), np.zeros_like(s_out)])
    scores = np.concatenate([s_in, s_out])
    mia_auc = roc_auc_score(labels, scores)

    # TPR @ FPR = 0.01
    thr = np.percentile(s_out, 99)  # FPR = 0.01
    tpr = np.mean(s_in > thr)

    return mia_auc, tpr
