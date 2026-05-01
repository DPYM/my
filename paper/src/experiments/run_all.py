"""
DualGuard main experiment runner.

Baselines:
  B1 – No Protection   (plain SGD)
  B2 – DP-SGD  (ε=4)
  B3 – DP-SGD  (ε=8)
  B4 – RotEmb only     (plain SGD + rotation)
  B5 – DualGuard       (DP-SGD ε=4 + rotation, ours)

Experiments:
  Exp 1 – AUC vs ε  sweep
  Exp 2 – AUC vs k  (embedding dimension) sweep
  Exp 3 – MIA  (LiRA) evaluation
  Exp 4 – EIA  (linear probe R²) evaluation
  Exp 5 – Combined  (dual-layer) metrics

Usage:
  python -m src.experiments.run_all  --dataset criteo  --data_path /path/to/criteo.txt
"""

import os, sys, json, argparse
import torch
import numpy as np

SRC = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if SRC not in sys.path:
    sys.path.insert(0, os.path.dirname(SRC))

from src.utils.seed     import set_all_seeds
from src.data.preprocess   import load_criteo, load_avazu, load_data
from src.models.deepfm     import DeepFM
from src.models.rotemb     import apply_rotemb, verify_accuracy_preservation
from src.training.trainers import train_plain, train_dp, create_model
from src.eval.metrics      import evaluate_model
from src.attacks.eia_inversion import linear_probe_r2, linear_probe_r2_from_tensors


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset",   default="criteo", choices=["criteo", "avazu"])
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--nrows",     type=int, default=8_000_000)
    parser.add_argument("--device",    default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed",      type=int, default=42)
    parser.add_argument("--embed_dim", type=int, default=16)
    parser.add_argument("--batch_size",type=int, default=4096)
    parser.add_argument("--epochs",    type=int, default=10)
    parser.add_argument("--out_dir",   default="./results")
    parser.add_argument("--epsilons",  nargs="+", type=float,
                        default=[0.5, 1, 2, 4, 8, 16])
    parser.add_argument("--skip_mia",  action="store_true",
                        help="Skip MIA (faster iteration during development)")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # ============================================================
    #  0. Load + preprocess data
    # ============================================================
    print("=" * 60)
    print(f"Loading {args.dataset.upper()} …")
    load_fn = load_criteo if args.dataset == "criteo" else load_avazu
    data = load_data(args.data_path, load_fn, nrows=args.nrows)

    print(f"  Train: {len(data['y_train']):,}")
    print(f"  Val:   {len(data['y_val']):,}")
    print(f"  Test:  {len(data['y_test']):,}")
    print(f"  Vocab sizes: {data['vocab_sizes']}")
    print(f"  n_dense: {data['n_dense']}")

    results = {}

    # ============================================================
    #  Exp 1: AUC vs ε
    # ============================================================
    print("\n" + "=" * 60)
    print("Experiment 1 — AUC vs ε")

    SEEDS_EXP1 = [args.seed, args.seed + 100, args.seed + 200]

    # B1 – upper bound (plain, no DP)
    b1_aucs = []
    for seed in SEEDS_EXP1:
        set_all_seeds(seed)
        m = create_model(data)
        train_plain(m, data, batch_size=args.batch_size, epochs=args.epochs,
                    device=args.device, verbose=False)
        auc, _ = evaluate_model(m, data, "test", device=args.device)
        b1_aucs.append(auc)
    auc_b1_mean, auc_b1_std = np.mean(b1_aucs), np.std(b1_aucs)
    print(f"  B1 (No Protection)  test AUC = {auc_b1_mean:.4f} ± {auc_b1_std:.4f}")

    # B4 – RotEmb only (sanity check: must equal B1)
    b4_aucs = []
    for seed in SEEDS_EXP1:
        set_all_seeds(seed)
        m_b4 = create_model(data)
        train_plain(m_b4, data, batch_size=args.batch_size, epochs=args.epochs,
                    device=args.device, verbose=False)
        state_before = {k: v.cpu().clone() for k, v in m_b4.state_dict().items()}
        from src.training.trainers import make_dataloader
        loader = make_dataloader(data, "test", batch_size=256)
        batch = next(iter(loader))
        if data["n_dense"] > 0:
            xs, xd, _ = batch
            xd = xd.to(args.device)
        else:
            xs, _ = batch
            xd = None
        xs = xs.to(args.device)
        apply_rotemb(m_b4)
        _ = verify_accuracy_preservation(m_b4, state_before, xs, xd)
        auc, _ = evaluate_model(m_b4, data, "test", device=args.device)
        b4_aucs.append(auc)
    auc_b4_mean, auc_b4_std = np.mean(b4_aucs), np.std(b4_aucs)
    print(f"  B4 (RotEmb only)   test AUC = {auc_b4_mean:.4f} ± {auc_b4_std:.4f}  "
          f"(Δ vs B1 = {abs(auc_b4_mean - auc_b1_mean):.6f})")

    e1_b2_means, e1_b2_stds, e1_b5_means, e1_b5_stds = [], [], [], []

    for eps in args.epsilons:
        b2_aucs, b5_aucs = [], []
        for seed in SEEDS_EXP1:
            # B2 – DP-SGD only
            set_all_seeds(seed)
            m = create_model(data)
            m, actual_eps = train_dp(m, data, target_epsilon=eps, delta=1e-5,
                                     batch_size=args.batch_size, epochs=args.epochs,
                                     device=args.device, verbose=False)
            auc, _ = evaluate_model(m, data, "test", device=args.device)
            b2_aucs.append(auc)

            # B5 – DualGuard (DP-SGD + RotEmb)
            set_all_seeds(seed)
            m = create_model(data)
            m, actual_eps = train_dp(m, data, target_epsilon=eps, delta=1e-5,
                                     batch_size=args.batch_size, epochs=args.epochs,
                                     device=args.device, verbose=False)
            apply_rotemb(m)
            auc, _ = evaluate_model(m, data, "test", device=args.device)
            b5_aucs.append(auc)

        e1_b2_means.append(float(np.mean(b2_aucs)))
        e1_b2_stds.append(float(np.std(b2_aucs)))
        e1_b5_means.append(float(np.mean(b5_aucs)))
        e1_b5_stds.append(float(np.std(b5_aucs)))
        print(f"  ε={eps:.1f}  B2 AUC={e1_b2_means[-1]:.4f}±{e1_b2_stds[-1]:.4f}  "
              f"B5 AUC={e1_b5_means[-1]:.4f}±{e1_b5_stds[-1]:.4f}  "
              f"Δ={abs(e1_b2_means[-1]-e1_b5_means[-1]):.6f}")

    results["exp1"] = {
        "b1_auc_mean": auc_b1_mean, "b1_auc_std": auc_b1_std,
        "b4_auc_mean": auc_b4_mean, "b4_auc_std": auc_b4_std,
        "epsilons": args.epsilons,
        "b2_means": e1_b2_means, "b2_stds": e1_b2_stds,
        "b5_means": e1_b5_means, "b5_stds": e1_b5_stds,
    }

    # ============================================================
    #  Exp 3: MIA evaluation (at ε = 4)
    # ============================================================
    if not args.skip_mia:
        print("\n" + "=" * 60)
        print("Experiment 3 — MIA (LiRA) @ ε = 4")
        _run_mia(data, args, results)
    else:
        results["exp3"] = {"skipped": True}

    # ============================================================
    #  Exp 4: EIA evaluation
    # ============================================================
    print("\n" + "=" * 60)
    print("Experiment 4 — EIA (Linear Probe R²)")

    e4 = {"b1_r2": None, "b2_r2": None, "b4_r2": None, "b5_r2": None}

    # B1 – before any rotation → R² ≈ 1.0
    set_all_seeds(args.seed)
    m_b1_eia = create_model(data)
    train_plain(m_b1_eia, data, batch_size=args.batch_size, epochs=args.epochs,
                device=args.device, verbose=False)
    e4["b1_r2"] = linear_probe_r2(m_b1_eia, m_b1_eia)
    print(f"  B1 (No Protection)   R² = {e4['b1_r2']:.4f}")

    # B2 – DP-SGD only (no rotation) → R² ≈ 1.0 (DP doesn't protect embeddings)
    set_all_seeds(args.seed)
    m_b2_eia = create_model(data)
    m_b2_eia, _ = train_dp(m_b2_eia, data, target_epsilon=4.0, delta=1e-5,
                           batch_size=args.batch_size, epochs=args.epochs,
                           device=args.device, verbose=False)
    V_b2 = m_b2_eia.embedding.weight.detach().cpu().clone()
    e4["b2_r2"] = linear_probe_r2_from_tensors(V_b2, V_b2)
    print(f"  B2 (DP-SGD only)     R² = {e4['b2_r2']:.4f}")

    # B4 – RotEmb only → R² ≈ 0.0
    set_all_seeds(args.seed)
    m_b4_eia = create_model(data)
    train_plain(m_b4_eia, data, batch_size=args.batch_size, epochs=args.epochs,
                device=args.device, verbose=False)
    V_orig = m_b4_eia.embedding.weight.detach().cpu().clone()
    apply_rotemb(m_b4_eia)
    V_rot = m_b4_eia.embedding.weight.detach().cpu()
    e4["b4_r2"] = linear_probe_r2_from_tensors(V_orig, V_rot)
    print(f"  B4 (RotEmb only)     R² = {e4['b4_r2']:.4f}")

    # B5 – DualGuard → R² ≈ 0.0 (rotation works even after DP training)
    set_all_seeds(args.seed)
    m_b5_eia = create_model(data)
    m_b5_eia, _ = train_dp(m_b5_eia, data, target_epsilon=4.0, delta=1e-5,
                           batch_size=args.batch_size, epochs=args.epochs,
                           device=args.device, verbose=False)
    V_b5_orig = m_b5_eia.embedding.weight.detach().cpu().clone()
    apply_rotemb(m_b5_eia)
    V_b5_rot = m_b5_eia.embedding.weight.detach().cpu()
    e4["b5_r2"] = linear_probe_r2_from_tensors(V_b5_orig, V_b5_rot)
    print(f"  B5 (DualGuard)       R² = {e4['b5_r2']:.4f}")

    results["exp4"] = e4

    # ============================================================
    #  Exp 2: AUC vs k (embedding dimension)  [optional]
    # ============================================================
    print("\n" + "=" * 60)
    print("Experiment 2 — EIA R² vs k (embedding dimension)")
    ks = [8, 16, 32, 64]
    e2 = {"ks": ks, "r2_scores": []}
    for k_val in ks:
        set_all_seeds(args.seed)
        m = DeepFM(sparse_vocab_sizes=data["vocab_sizes"],
                    n_dense=data["n_dense"],
                    embed_dim=k_val,
                    dnn_hidden_units=(400, 400, 400))
        train_plain(m, data, batch_size=args.batch_size, epochs=args.epochs,
                     device=args.device)
        V_orig = m.embedding.weight.detach().cpu().clone()
        R = apply_rotemb(m)
        V_rot = m.embedding.weight.detach().cpu()
        r2 = linear_probe_r2_from_tensors(V_orig, V_rot)
        e2["r2_scores"].append(r2)
        print(f"  k={k_val:2d}  R² = {r2:.4f}")

    results["exp2"] = e2

    # ============================================================
    #  Save
    # ============================================================
    out_path = os.path.join(args.out_dir, "results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=float)
    print(f"\nResults saved to {out_path}")


def _run_mia(data, args, results):
    """Full LiRA MIA evaluation — trains K=4 shadow model pairs."""

    from src.attacks.mia_lira import (
        collect_confidences, fit_lira_distributions, run_lira_attack,
    )
    from src.training.trainers import make_dataloader
    from torch.utils.data import TensorDataset, DataLoader, Subset

    K_SHADOWS = 4
    n_train = len(data["y_train"])
    n_dense = data["n_dense"]

    print(f"  Training {K_SHADOWS} shadow model pairs (IN + OUT) …")

    shadow_confs_in = []
    shadow_confs_out = []

    for k_idx in range(K_SHADOWS):
        np.random.seed(args.seed + k_idx * 1000)
        in_idx = np.random.choice(n_train, n_train // 2, replace=False)
        out_idx = np.array([i for i in range(n_train) if i not in set(in_idx)])

        data_shadow_in = _subset_data(data, in_idx)
        data_shadow_out = _subset_data(data, out_idx)

        set_all_seeds(args.seed + k_idx)
        m_in = create_model(data)
        train_plain(m_in, data_shadow_in, batch_size=args.batch_size,
                    epochs=args.epochs, device=args.device, verbose=False)

        set_all_seeds(args.seed + k_idx + 500)
        m_out = create_model(data)
        train_plain(m_out, data_shadow_out, batch_size=args.batch_size,
                    epochs=args.epochs, device=args.device, verbose=False)

        n_probe = min(500, len(data["y_val"]))
        probe_loader = make_dataloader(data, "val", batch_size=256, shuffle=True)

        confs_in_k = []
        confs_out_k = []
        m_in = m_in.to(args.device)
        m_out = m_out.to(args.device)

        count = 0
        for batch in probe_loader:
            if count >= n_probe:
                break
            if n_dense > 0:
                xs, xd, yb = batch
            else:
                xs, yb = batch
                xd = None
            for i in range(len(xs)):
                if count >= n_probe:
                    break
                c_in = collect_confidences(m_in, xs[i],
                                           xd[i] if xd is not None else None,
                                           n_aug=30, device=args.device)
                c_out = collect_confidences(m_out, xs[i],
                                            xd[i] if xd is not None else None,
                                            n_aug=30, device=args.device)
                confs_in_k.append(c_in)
                confs_out_k.append(c_out)
                count += 1

        shadow_confs_in.append(confs_in_k)
        shadow_confs_out.append(confs_out_k)
        print(f"    Shadow pair {k_idx+1}/{K_SHADOWS} done")

    all_in = np.concatenate([np.concatenate(c) for c in shadow_confs_in])
    all_out = np.concatenate([np.concatenate(c) for c in shadow_confs_out])
    mu_in, sigma_in, mu_out, sigma_out = fit_lira_distributions(
        [all_in], [all_out]
    )
    print(f"  Fitted LiRA distributions: IN({mu_in:.3f},{sigma_in:.3f}) "
          f"OUT({mu_out:.3f},{sigma_out:.3f})")

    in_loader = make_dataloader(data, "train", batch_size=256, shuffle=True)
    out_loader = make_dataloader(data, "val", batch_size=256, shuffle=True)

    # B1
    set_all_seeds(args.seed)
    m = create_model(data)
    train_plain(m, data, batch_size=args.batch_size, epochs=args.epochs,
                device=args.device)
    auc_b1, tpr_b1 = run_lira_attack(m, in_loader, out_loader,
                                      mu_in, sigma_in, mu_out, sigma_out,
                                      device=args.device, max_samples=1000)
    print(f"  B1 (No Protection)  MIA AUC = {auc_b1:.4f}  TPR@0.01 = {tpr_b1:.4f}")

    # B2
    set_all_seeds(args.seed)
    m = create_model(data)
    m, _ = train_dp(m, data, target_epsilon=4.0, delta=1e-5,
                    batch_size=args.batch_size, epochs=args.epochs,
                    device=args.device)
    auc_b2, tpr_b2 = run_lira_attack(m, in_loader, out_loader,
                                      mu_in, sigma_in, mu_out, sigma_out,
                                      device=args.device, max_samples=1000)
    print(f"  B2 (DP-SGD ε=4)     MIA AUC = {auc_b2:.4f}  TPR@0.01 = {tpr_b2:.4f}")

    # B5
    set_all_seeds(args.seed)
    m = create_model(data)
    m, _ = train_dp(m, data, target_epsilon=4.0, delta=1e-5,
                    batch_size=args.batch_size, epochs=args.epochs,
                    device=args.device)
    apply_rotemb(m)
    auc_b5, tpr_b5 = run_lira_attack(m, in_loader, out_loader,
                                      mu_in, sigma_in, mu_out, sigma_out,
                                      device=args.device, max_samples=1000)
    print(f"  B5 (DualGuard)      MIA AUC = {auc_b5:.4f}  TPR@0.01 = {tpr_b5:.4f}")
    print("  NOTE: B2 and B5 MIA metrics should be identical "
          "(RotEmb does not affect output probabilities)")

    results["exp3"] = {
        "b1_mia_auc": auc_b1, "b1_tpr_001": tpr_b1,
        "b2_mia_auc": auc_b2, "b2_tpr_001": tpr_b2,
        "b5_mia_auc": auc_b5, "b5_tpr_001": tpr_b5,
        "n_shadow_pairs": K_SHADOWS,
    }


def _subset_data(data_dict, indices):
    """Create a subset of data_dict with only the given training indices."""
    sub = dict(data_dict)
    sub["X_sparse_train"] = data_dict["X_sparse_train"][indices]
    sub["y_train"] = data_dict["y_train"][indices]
    if data_dict["n_dense"] > 0:
        sub["X_dense_train"] = data_dict["X_dense_train"][indices]
    return sub


if __name__ == "__main__":
    main()
