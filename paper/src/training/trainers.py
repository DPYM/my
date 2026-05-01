"""
Training utilities for DualGuard experiments.

  - train_plain()  : standard SGD without DP (baselines B1, B4)
  - train_dp()     : DP-SGD via Opacus (baselines B2, B3, B5)
  - create_model() : build DeepFM from dataset metadata
"""

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import roc_auc_score
import numpy as np


def create_model(data_dict, embed_dim=16, dnn_hidden_units=(400, 400, 400),
                 dropout=0.2):
    return DeepFM(
        sparse_vocab_sizes=data_dict["vocab_sizes"],
        n_dense=data_dict["n_dense"],
        embed_dim=embed_dim,
        dnn_hidden_units=dnn_hidden_units,
        dropout=dropout,
    )


def make_dataloader(data_dict, split_name, batch_size, shuffle=False):
    Xs = torch.as_tensor(data_dict[f"X_sparse_{split_name}"])
    y  = torch.as_tensor(data_dict[f"y_{split_name}"], dtype=torch.float32)
    if data_dict["n_dense"] > 0:
        Xd = torch.as_tensor(data_dict[f"X_dense_{split_name}"])
        ds = TensorDataset(Xs, Xd, y)
    else:
        ds = TensorDataset(Xs, y)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


# ================================================================
#  Plain SGD training  (B1, B4)
# ================================================================

def train_plain(model, data_dict, batch_size=4096, epochs=10,
                lr=1e-3, patience=3, device="cuda", verbose=True):
    """
    Returns model trained on the train split.
    Reports epoch-level AUC on validation split for early stopping.
    """
    model = model.to(device)
    train_loader = make_dataloader(data_dict, "train", batch_size, shuffle=True)

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.BCELoss()

    best_auc = 0
    best_state = None
    stall = 0
    n_dense = data_dict["n_dense"]

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch in train_loader:
            if n_dense > 0:
                xs, xd, yb = batch
                xs, xd, yb = xs.to(device), xd.to(device), yb.to(device)
                preds = model(xs, xd)
            else:
                xs, yb = batch
                xs, yb = xs.to(device), yb.to(device)
                preds = model(xs)

            loss = loss_fn(preds, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item()

        val_auc = _evaluate_auc(model, data_dict, "val", device, n_dense)
        if verbose:
            print(f"  Epoch {epoch+1:2d}  loss={total_loss/len(train_loader):.4f}  val_auc={val_auc:.4f}")

        if val_auc > best_auc:
            best_auc = val_auc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            stall = 0
        else:
            stall += 1
            if stall >= patience:
                if verbose:
                    print(f"  Early stopping at epoch {epoch+1}")
                break

    model.load_state_dict(best_state)
    return model


# ================================================================
#  DP-SGD training via Opacus  (B2, B3, B5)
# ================================================================

def train_dp(model, data_dict, target_epsilon=4.0, delta=1e-5,
             batch_size=4096, epochs=10, max_grad_norm=5.0,
             lr=1e-3, device="cuda", verbose=True):
    """
    DP-SGD training with Opacus PrivacyEngine.

    Noise multiplier sigma is computed automatically from (target_epsilon, delta, q, T).

    Returns model and the actual epsilon consumed.
    """
    try:
        from opacus import PrivacyEngine
        from opacus.validators import ModuleValidator
    except ImportError:
        raise ImportError("pip install opacus  (Opacus is required for DP-SGD)")

    model = model.to(device)

    # Opacus needs the model in train mode to replace autograd grad samplers
    model.train()

    # Disable dropout for Opacus compatibility (vmap doesn't support per-sample randomness)
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.p = 0.0

    # Make Opacus-compatible (replace BatchNorm with GroupNorm etc. if present)
    try:
        model = ModuleValidator.fix(model)
    except Exception:
        pass  # fix() is optional for our model (no BN/LN)
    model = model.to(device)

    train_loader = make_dataloader(data_dict, "train", batch_size, shuffle=True)

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.BCELoss()

    n_train = len(data_dict["y_train"])
    sample_rate = min(batch_size / n_train, 1.0)

    privacy_engine = PrivacyEngine(accountant="rdp")
    model, opt, train_loader = privacy_engine.make_private(
        module=model,
        optimizer=opt,
        data_loader=train_loader,
        noise_multiplier=_compute_noise_multiplier(
            target_epsilon, delta,
            sample_rate=sample_rate,
            total_steps=-(-n_train // batch_size) * epochs,
        ),
        max_grad_norm=max_grad_norm,
    )

    n_dense = data_dict["n_dense"]
    best_auc = 0
    best_state = None
    stall = 0
    patience = 3

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch in train_loader:
            if n_dense > 0:
                xs, xd, yb = batch
                xs, xd, yb = xs.to(device), xd.to(device), yb.to(device)
                preds = model(xs, xd)
            else:
                xs, yb = batch
                xs, yb = xs.to(device), yb.to(device)
                preds = model(xs)

            loss = loss_fn(preds, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item()

        eps_spent = privacy_engine.accountant.get_epsilon(delta=delta)
        val_auc = _evaluate_auc(model, data_dict, "val", device, n_dense)
        if verbose:
            print(f"  Epoch {epoch+1:2d}  loss={total_loss/len(train_loader):.4f}  "
                  f"val_auc={val_auc:.4f}  eps_spent={eps_spent:.2f}")

        if val_auc > best_auc:
            best_auc = val_auc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            stall = 0
        else:
            stall += 1
            if stall >= patience:
                if verbose:
                    print(f"  Early stopping at epoch {epoch+1}")
                break

        if privacy_engine.accountant.get_epsilon(delta=delta) >= target_epsilon:
            if verbose:
                print(f"  Privacy budget exhausted at epoch {epoch+1}")
            break

    model.load_state_dict(best_state)

    # Remove Opacus GradSampleModule wrapper to get raw model back
    if hasattr(model, "_module"):
        model = model._module

    final_eps = privacy_engine.accountant.get_epsilon(delta=delta)
    return model, final_eps


def _compute_noise_multiplier(target_eps, delta, sample_rate, total_steps,
                              start_sigma=0.5, end_sigma=100.0, steps=200):
    """Binary search to find sigma that hits target_epsilon after total_steps steps."""
    from opacus.accountants import RDPAccountant

    T = total_steps
    lo, hi = start_sigma, end_sigma

    for _ in range(steps):
        mid = (lo + hi) / 2
        acct = RDPAccountant()
        for _ in range(T):
            acct.step(noise_multiplier=mid, sample_rate=sample_rate)
        eps = acct.get_epsilon(delta=delta)
        if eps > target_eps:
            lo = mid
        else:
            hi = mid

    return hi


# ================================================================
#  Helpers
# ================================================================

@torch.no_grad()
def _evaluate_auc(model, data_dict, split, device, n_dense, max_batches=50):
    """Evaluate AUC over at most max_batches batches of the split."""
    model.eval()
    loader = make_dataloader(data_dict, split, batch_size=16384)
    all_preds, all_y = [], []

    for i, batch in enumerate(loader):
        if i >= max_batches:
            break
        if n_dense > 0:
            xs, xd, yb = batch
            xs, xd = xs.to(device), xd.to(device)
            p = model(xs, xd).cpu().numpy()
        else:
            xs, yb = batch
            xs = xs.to(device)
            p = model(xs).cpu().numpy()
        all_preds.append(p)
        all_y.append(yb.numpy())

    preds = np.concatenate(all_preds)
    ys = np.concatenate(all_y)
    return roc_auc_score(ys, preds)


# late import
from src.models.deepfm import DeepFM
