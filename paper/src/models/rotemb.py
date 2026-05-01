"""
RotEmb: post-training random rotation for embedding protection.

Applies a single Haar-distributed orthogonal matrix R to all embeddings
and compensates the first DNN Linear layer so that model predictions
are mathematically unchanged (Theorem 1).

After this transformation the model can be deployed WITHOUT R.
An adversary who obtains the checkpoint sees only rotated embeddings
and faces the ‖v‖² reconstruction lower bound (Theorem 3).
"""

import torch
import torch.nn as nn


def random_orthogonal(k):
    """Sample R ~ Haar(O(k)) using QR decomposition on a Gaussian matrix."""
    A = torch.randn(k, k)
    Q, R = torch.linalg.qr(A)
    # Flip columns with negative diagonal to ensure uniform Haar measure
    d = torch.diag(R)
    sign = torch.where(d >= 0, 1.0, -1.0)
    Q = Q * sign  # broadcast over columns → now Q ∈ SO(k) ⊂ O(k)
    return Q


def apply_rotemb(model, R=None):
    """
    Rotate embeddings and compensate W^(0) in-place.

    Parameters
    ----------
    model : DeepFM
        Trained model (will be modified in-place).
    R : Tensor (k, k), optional
        Rotation matrix. Generated from Haar measure if not provided.

    Returns
    -------
    R : Tensor (k, k)
        The rotation matrix (should be stored securely, NOT deployed).
    """
    k = model.embed_dim
    F = model.n_sparse

    if R is None:
        R = random_orthogonal(k)
    R = R.to(next(model.parameters()).device)

    with torch.no_grad():
        # ---- 1. Rotate embedding table ----
        # weight: (vocab_size, k)   each row = one embedding
        # v' = R @ v  →  weight' = weight @ R^T
        model.embedding.weight.data = model.embedding.weight.data @ R.T

        # ---- 2. Compensate first DNN Linear layer ----
        lin = model.first_dnn_linear       # nn.Linear(in, h1)
        h1 = lin.out_features
        W = lin.weight.data                # (h1, in)  where in = F·k + n_dense

        W_sparse = W[:, :F * k]            # (h1, F·k)
        W_dense  = W[:, F * k:]            # (h1, n_dense)  – not rotated

        # Reshape → (h1, F, k), apply R^T per field (Theorem 1, DNN part)
        W_comp = W_sparse.view(h1, F, k) @ R.T   # (h1, F, k)
        lin.weight.data[:, :F * k] = W_comp.reshape(h1, F * k)
        # W_dense block is left unchanged

    return R


def verify_accuracy_preservation(model, before_state_dict, X_sparse, X_dense=None, tol=1e-6):
    """
    Sanity check: confirm that ŷ'(x) == ŷ(x) after RotEmb.

    Parameters
    ----------
    model : DeepFM (after apply_rotemb)
    before_state_dict : state_dict from before rotation
    X_sparse, X_dense : sample batch

    Returns
    -------
    max_diff : float   (should be < tol)
    """
    model.eval()

    # Temporary model with pre-rotation weights just for verification
    from copy import deepcopy
    model_before = deepcopy(model)
    model_before.load_state_dict(before_state_dict)
    model_before.eval()

    with torch.no_grad():
        y_before = model_before(X_sparse, X_dense)
        y_after  = model(X_sparse, X_dense)

    max_diff = (y_before - y_after).abs().max().item()
    ok = max_diff < tol
    print(f"Accuracy preservation check: max|Δŷ| = {max_diff:.2e}  {'OK' if ok else 'FAIL'}")
    return max_diff
