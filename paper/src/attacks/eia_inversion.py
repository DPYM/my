"""
Embedding Inversion Attack (EIA) evaluation.

Two complementary metrics:
  1. Linear Probe R²   – can a linear regressor map rotated embedding dims
                          back to original embedding dims?
  2. Label Propagation  – KMeans clustering on rotated embeddings;
                          accuracy of propagating semantic labels from anchors.
"""

import torch
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.cluster import KMeans
from sklearn.metrics import r2_score


def linear_probe_r2(model_before, model_after, n_samples=5000, train_ratio=0.5):
    """
    Train linear regressors to predict each original embedding dimension
    from the rotated embedding.

    Parameters
    ----------
    model_before : DeepFM (pre-rotation)
    model_after  : DeepFM (post-rotation)
    n_samples    : int — number of embedding rows to sample
    train_ratio  : float — fraction used for training (rest for testing)

    Returns
    -------
    mean_r2 : float   (→ 1.0 for no protection, → 0.0 when rotation works)
    """
    V_orig = model_before.embedding.weight.detach().cpu().numpy()
    V_rot  = model_after.embedding.weight.detach().cpu().numpy()

    n = min(n_samples, V_orig.shape[0])
    idx = np.random.choice(V_orig.shape[0], n, replace=False)

    X = V_rot[idx]
    n_train = int(n * train_ratio)

    r2_scores = []
    for dim in range(V_orig.shape[1]):
        y = V_orig[idx, dim]
        probe = Ridge(alpha=1.0)
        probe.fit(X[:n_train], y[:n_train])
        y_pred = probe.predict(X[n_train:])
        r2_scores.append(r2_score(y[n_train:], y_pred))

    return float(np.mean(r2_scores))


def linear_probe_r2_from_tensors(V_orig, V_rot, n_samples=5000, train_ratio=0.5):
    """
    Same as linear_probe_r2 but takes raw tensors instead of models.
    Useful when comparing the same model before/after rotation.

    Parameters
    ----------
    V_orig : np.ndarray (vocab, k) — original embeddings
    V_rot  : np.ndarray (vocab, k) — rotated embeddings
    """
    if isinstance(V_orig, torch.Tensor):
        V_orig = V_orig.detach().cpu().numpy()
    if isinstance(V_rot, torch.Tensor):
        V_rot = V_rot.detach().cpu().numpy()

    n = min(n_samples, V_orig.shape[0])
    idx = np.random.choice(V_orig.shape[0], n, replace=False)

    X = V_rot[idx]
    n_train = int(n * train_ratio)

    r2_scores = []
    for dim in range(V_orig.shape[1]):
        y = V_orig[idx, dim]
        probe = Ridge(alpha=1.0)
        probe.fit(X[:n_train], y[:n_train])
        y_pred = probe.predict(X[n_train:])
        r2_scores.append(r2_score(y[n_train:], y_pred))

    return float(np.mean(r2_scores))


def label_propagation_accuracy(model_after, feature_names, anchor_map,
                               n_clusters=100, seed=42):
    """
    KMeans-cluster rotated embeddings, propagate labels from anchors,
    and measure accuracy on held-out labelled items.

    Parameters
    ----------
    model_after     : DeepFM (post-RotEmb)
    feature_names   : list[str] — semantic label for each unique feature value
    anchor_map      : dict {idx → semantic_label}  — small set of known anchors
    n_clusters      : int
    seed            : int

    Returns
    -------
    accuracy : float  (higher = attack succeeds, lower = rotation works)
    """
    V = model_after.embedding.weight.detach().cpu().numpy()

    n_vocab = V.shape[0]
    if len(feature_names) != n_vocab:
        raise ValueError(
            f"feature_names length ({len(feature_names)}) does not match "
            f"embedding table size ({n_vocab})"
        )

    km = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10)
    clusters = km.fit_predict(V)

    cluster_label = {}
    anchor_votes = {}

    for idx, label in anchor_map.items():
        c = clusters[idx]
        if c not in anchor_votes:
            anchor_votes[c] = {}
        if label not in anchor_votes[c]:
            anchor_votes[c][label] = 0
        anchor_votes[c][label] += 1

    for c, votes in anchor_votes.items():
        cluster_label[c] = max(votes, key=votes.get)

    correct, total = 0, 0
    for idx, true_label in enumerate(feature_names):
        if idx in anchor_map:
            continue
        c = clusters[idx]
        pred_label = cluster_label.get(c)
        if pred_label is not None:
            if pred_label == true_label:
                correct += 1
            total += 1

    return correct / total if total > 0 else 0.0
