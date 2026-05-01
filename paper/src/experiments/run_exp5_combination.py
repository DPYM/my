"""
Experiment 5 — Combined dual-layer evaluation.

Compares DP-SGD only, RotEmb only, and DualGuard at multiple ε values.
Reports both utility (AUC) and privacy (MIA AUC, EIA R²) metrics.

Usage:
  python -m src.experiments.run_exp5_combination --dataset criteo --data_path ./data/train.txt
"""

import os, sys, json, argparse
import torch
import numpy as np

SRC = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if SRC not in sys.path:
    sys.path.insert(0, os.path.dirname(SRC))

from src.utils.seed import set_all_seeds
from src.data.preprocess import load_criteo, load_avazu, load_data
from src.models.deepfm import DeepFM
from src.models.rotemb import apply_rotemb
from src.training.trainers import train_plain, train_dp, create_model, make_dataloader
from src.eval.metrics import evaluate_model
from src.attacks.eia_inversion import linear_probe_r2_from_tensors
from src.attacks.mia_lira import (
    collect_confidences, fit_lira_distributions, run_lira_attack,
)
from src.utils.logger import ExperimentLogger

SEEDS = [42, 123, 456]
K_SHADOWS = 4


def _subset_data(data_dict, indices):
    sub = dict(data_dict)
    sub["X_sparse_train"] = data_dict["X_sparse_train"][indices]
    sub["y_train"] = data_dict["y_train"][indices]
    if data_dict["n_dense"] > 0:
        sub["X_dense_train"] = data_dict["X_dense_train"][indices]
    return sub


def _fit_lira(data, args, seed):
    """Train shadow models and fit LiRA distributions."""
    n_train = len(data["y_train"])
    n_dense = data["n_dense"]

    shadow_confs_in, shadow_confs_out = [], []
    for k_idx in range(K_SHADOWS):
        rng = np.random.RandomState(seed + k_idx * 1000)
        in_idx = rng.choice(n_train, n_train // 2, replace=False)
        out_idx = np.array([i for i in range(n_train) if i not in set(in_idx)])

        set_all_seeds(seed + k_idx)
        m_in = create_model(data)
        train_plain(m_in, _subset_data(data, in_idx), batch_size=args.batch_size,
                    epochs=args.epochs, device=args.device, verbose=False)

        set_all_seeds(seed + k_idx + 500)
        m_out = create_model(data)
        train_plain(m_out, _subset_data(data, out_idx), batch_size=args.batch_size,
                    epochs=args.epochs, device=args.device, verbose=False)

        m_in, m_out = m_in.to(args.device), m_out.to(args.device)
        probe_loader = make_dataloader(data, "val", batch_size=256, shuffle=True)
        confs_in_k, confs_out_k = [], []
        count, n_probe = 0, min(300, len(data["y_val"]))

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
                                           n_aug=20, device=args.device)
                c_out = collect_confidences(m_out, xs[i],
                                            xd[i] if xd is not None else None,
                                            n_aug=20, device=args.device)
                confs_in_k.append(c_in)
                confs_out_k.append(c_out)
                count += 1

        shadow_confs_in.append(np.concatenate(confs_in_k))
        shadow_confs_out.append(np.concatenate(confs_out_k))

    all_in = np.concatenate(shadow_confs_in)
    all_out = np.concatenate(shadow_confs_out)
    return fit_lira_distributions([all_in], [all_out])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="criteo", choices=["criteo", "avazu"])
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--nrows", type=int, default=8_000_000)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--embed_dim", type=int, default=16)
    parser.add_argument("--batch_size", type=int, default=4096)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--out_dir", default="./results")
    parser.add_argument("--epsilons", nargs="+", type=float, default=[2, 4, 8])
    parser.add_argument("--n_seeds", type=int, default=3)
    parser.add_argument("--skip_mia", action="store_true")
    args = parser.parse_args()

    logger = ExperimentLogger(args.out_dir, "exp5_combination")
    logger.log_config(dataset=args.dataset, epsilons=args.epsilons,
                      embed_dim=args.embed_dim, n_seeds=args.n_seeds)

    print("=" * 60)
    print(f"Experiment 5 — Combined Dual-Layer Evaluation  ({args.dataset})")
    load_fn = load_criteo if args.dataset == "criteo" else load_avazu
    data = load_data(args.data_path, load_fn, nrows=args.nrows)

    seeds = SEEDS[:args.n_seeds]
    results = {}

    # B1 — No Protection
    b1_aucs = []
    for seed in seeds:
        set_all_seeds(seed)
        m = create_model(data, embed_dim=args.embed_dim)
        train_plain(m, data, batch_size=args.batch_size, epochs=args.epochs,
                    device=args.device, verbose=False)
        auc, _ = evaluate_model(m, data, "test", device=args.device)
        b1_aucs.append(auc)
    results["b1_auc_mean"] = float(np.mean(b1_aucs))
    results["b1_auc_std"] = float(np.std(b1_aucs))
    print(f"  B1 (No Protection)  AUC = {results['b1_auc_mean']:.4f} ± {results['b1_auc_std']:.4f}")

    # B4 — RotEmb only
    b4_aucs, b4_r2s = [], []
    for seed in seeds:
        set_all_seeds(seed)
        m = create_model(data, embed_dim=args.embed_dim)
        train_plain(m, data, batch_size=args.batch_size, epochs=args.epochs,
                    device=args.device, verbose=False)
        V_orig = m.embedding.weight.detach().cpu().clone()
        apply_rotemb(m)
        V_rot = m.embedding.weight.detach().cpu()
        auc, _ = evaluate_model(m, data, "test", device=args.device)
        r2 = linear_probe_r2_from_tensors(V_orig, V_rot)
        b4_aucs.append(auc)
        b4_r2s.append(r2)
    results["b4_auc_mean"] = float(np.mean(b4_aucs))
    results["b4_auc_std"] = float(np.std(b4_aucs))
    results["b4_eia_r2_mean"] = float(np.mean(b4_r2s))
    print(f"  B4 (RotEmb only)    AUC = {results['b4_auc_mean']:.4f} ± {results['b4_auc_std']:.4f}  "
          f"EIA R² = {results['b4_eia_r2_mean']:.4f}")

    # DP-SGD only & DualGuard at each ε
    for eps in args.epsilons:
        b2_aucs, b5_aucs = [], []
        b2_r2s, b5_r2s = [], []
        b2_mias, b5_mias = [], []

        for seed in seeds:
            # B2 — DP-SGD only
            set_all_seeds(seed)
            m = create_model(data, embed_dim=args.embed_dim)
            m, actual_eps = train_dp(m, data, target_epsilon=eps, delta=1e-5,
                                     batch_size=args.batch_size, epochs=args.epochs,
                                     device=args.device, verbose=False)
            auc, _ = evaluate_model(m, data, "test", device=args.device)
            V_dp = m.embedding.weight.detach().cpu().clone()
            r2 = linear_probe_r2_from_tensors(V_dp, V_dp)
            b2_aucs.append(auc)
            b2_r2s.append(r2)

            # B5 — DualGuard
            set_all_seeds(seed)
            m = create_model(data, embed_dim=args.embed_dim)
            m, actual_eps = train_dp(m, data, target_epsilon=eps, delta=1e-5,
                                     batch_size=args.batch_size, epochs=args.epochs,
                                     device=args.device, verbose=False)
            V_orig = m.embedding.weight.detach().cpu().clone()
            apply_rotemb(m)
            V_rot = m.embedding.weight.detach().cpu()
            auc, _ = evaluate_model(m, data, "test", device=args.device)
            r2 = linear_probe_r2_from_tensors(V_orig, V_rot)
            b5_aucs.append(auc)
            b5_r2s.append(r2)

        eps_key = f"eps{eps:.0f}"
        results[f"b2_{eps_key}_auc_mean"] = float(np.mean(b2_aucs))
        results[f"b2_{eps_key}_auc_std"] = float(np.std(b2_aucs))
        results[f"b2_{eps_key}_eia_r2_mean"] = float(np.mean(b2_r2s))
        results[f"b5_{eps_key}_auc_mean"] = float(np.mean(b5_aucs))
        results[f"b5_{eps_key}_auc_std"] = float(np.std(b5_aucs))
        results[f"b5_{eps_key}_eia_r2_mean"] = float(np.mean(b5_r2s))

        print(f"  ε={eps:.0f}  B2 AUC={np.mean(b2_aucs):.4f}  EIA R²={np.mean(b2_r2s):.4f}  |  "
              f"B5 AUC={np.mean(b5_aucs):.4f}  EIA R²={np.mean(b5_r2s):.4f}")

    # MIA evaluation (optional — expensive)
    if not args.skip_mia:
        print("\n  Running MIA evaluation …")
        seed = seeds[0]
        mu_in, sigma_in, mu_out, sigma_out = _fit_lira(data, args, seed)
        in_loader = make_dataloader(data, "train", batch_size=256, shuffle=True)
        out_loader = make_dataloader(data, "val", batch_size=256, shuffle=True)

        for eps in args.epsilons:
            eps_key = f"eps{eps:.0f}"

            # B2 MIA
            set_all_seeds(seed)
            m = create_model(data, embed_dim=args.embed_dim)
            m, _ = train_dp(m, data, target_epsilon=eps, delta=1e-5,
                            batch_size=args.batch_size, epochs=args.epochs,
                            device=args.device, verbose=False)
            mia_b2, tpr_b2 = run_lira_attack(m, in_loader, out_loader,
                                              mu_in, sigma_in, mu_out, sigma_out,
                                              device=args.device, max_samples=500)
            results[f"b2_{eps_key}_mia_auc"] = mia_b2

            # B5 MIA
            set_all_seeds(seed)
            m = create_model(data, embed_dim=args.embed_dim)
            m, _ = train_dp(m, data, target_epsilon=eps, delta=1e-5,
                            batch_size=args.batch_size, epochs=args.epochs,
                            device=args.device, verbose=False)
            apply_rotemb(m)
            mia_b5, tpr_b5 = run_lira_attack(m, in_loader, out_loader,
                                              mu_in, sigma_in, mu_out, sigma_out,
                                              device=args.device, max_samples=500)
            results[f"b5_{eps_key}_mia_auc"] = mia_b5

            print(f"  ε={eps:.0f}  B2 MIA AUC={mia_b2:.4f}  B5 MIA AUC={mia_b5:.4f}")

    logger.log_metrics(results)
    logger.close()

    print("\n" + "=" * 60)
    print("Combination Summary:")
    print(f"  B1 (No Protection)  AUC = {results['b1_auc_mean']:.4f}")
    print(f"  B4 (RotEmb only)    AUC = {results['b4_auc_mean']:.4f}  EIA R² = {results['b4_eia_r2_mean']:.4f}")
    for eps in args.epsilons:
        eps_key = f"eps{eps:.0f}"
        print(f"  B2 (DP ε={eps:.0f})       AUC = {results[f'b2_{eps_key}_auc_mean']:.4f}  "
              f"EIA R² = {results[f'b2_{eps_key}_eia_r2_mean']:.4f}")
        print(f"  B5 (DualGuard ε={eps:.0f}) AUC = {results[f'b5_{eps_key}_auc_mean']:.4f}  "
              f"EIA R² = {results[f'b5_{eps_key}_eia_r2_mean']:.4f}")


if __name__ == "__main__":
    main()
