"""
Experiment 3 — MIA (LiRA) evaluation at ε=4.

Trains K=4 shadow model pairs to fit IN/OUT distributions,
then evaluates MIA AUC and TPR@FPR=0.01 on B1, B2, B5.

Usage:
  python -m src.experiments.run_exp3_mia --dataset criteo --data_path ./data/train.txt
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
from src.attacks.mia_lira import (
    collect_confidences, fit_lira_distributions, run_lira_attack,
)
from src.utils.logger import ExperimentLogger

SEEDS = [42, 123, 456, 789, 1024]
K_SHADOWS = 4


def _subset_data(data_dict, indices):
    sub = dict(data_dict)
    sub["X_sparse_train"] = data_dict["X_sparse_train"][indices]
    sub["y_train"] = data_dict["y_train"][indices]
    if data_dict["n_dense"] > 0:
        sub["X_dense_train"] = data_dict["X_dense_train"][indices]
    return sub


def _train_shadow_models(data, args, seed):
    """Train K_SHADOWS shadow model pairs and collect confidence distributions."""
    n_train = len(data["y_train"])
    n_dense = data["n_dense"]

    shadow_confs_in = []
    shadow_confs_out = []

    for k_idx in range(K_SHADOWS):
        rng = np.random.RandomState(seed + k_idx * 1000)
        in_idx = rng.choice(n_train, n_train // 2, replace=False)
        out_idx = np.array([i for i in range(n_train) if i not in set(in_idx)])

        data_in = _subset_data(data, in_idx)
        data_out = _subset_data(data, out_idx)

        set_all_seeds(seed + k_idx)
        m_in = create_model(data)
        train_plain(m_in, data_in, batch_size=args.batch_size,
                    epochs=args.epochs, device=args.device, verbose=False)

        set_all_seeds(seed + k_idx + 500)
        m_out = create_model(data)
        train_plain(m_out, data_out, batch_size=args.batch_size,
                    epochs=args.epochs, device=args.device, verbose=False)

        m_in = m_in.to(args.device)
        m_out = m_out.to(args.device)

        probe_loader = make_dataloader(data, "val", batch_size=256, shuffle=True)
        confs_in_k, confs_out_k = [], []
        count = 0
        n_probe = min(500, len(data["y_val"]))

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

        shadow_confs_in.append(np.concatenate(confs_in_k))
        shadow_confs_out.append(np.concatenate(confs_out_k))
        print(f"    Shadow pair {k_idx+1}/{K_SHADOWS} done")

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
    parser.add_argument("--epsilon", type=float, default=4.0)
    parser.add_argument("--n_seeds", type=int, default=3, help="Training 4×2 shadow models per seed is expensive")
    args = parser.parse_args()

    logger = ExperimentLogger(args.out_dir, "exp3_mia")
    logger.log_config(dataset=args.dataset, epsilon=args.epsilon,
                      n_shadow_pairs=K_SHADOWS, n_seeds=args.n_seeds)

    print("=" * 60)
    print(f"Experiment 3 — MIA (LiRA) @ ε={args.epsilon}  ({args.dataset})")
    load_fn = load_criteo if args.dataset == "criteo" else load_avazu
    data = load_data(args.data_path, load_fn, nrows=args.nrows)

    seeds = SEEDS[:args.n_seeds]
    b1_mia, b2_mia, b5_mia = [], [], []
    b1_tpr, b2_tpr, b5_tpr = [], [], []

    for seed in seeds:
        print(f"\n--- Seed {seed} ---")
        print("  Training shadow models …")
        mu_in, sigma_in, mu_out, sigma_out = _train_shadow_models(data, args, seed)
        print(f"  LiRA distributions: IN({mu_in:.3f},{sigma_in:.3f}) OUT({mu_out:.3f},{sigma_out:.3f})")

        in_loader = make_dataloader(data, "train", batch_size=256, shuffle=True)
        out_loader = make_dataloader(data, "val", batch_size=256, shuffle=True)

        # B1
        set_all_seeds(seed)
        m = create_model(data, embed_dim=args.embed_dim)
        train_plain(m, data, batch_size=args.batch_size, epochs=args.epochs, device=args.device, verbose=False)
        auc_b1, tpr_b1 = run_lira_attack(m, in_loader, out_loader,
                                          mu_in, sigma_in, mu_out, sigma_out,
                                          device=args.device, max_samples=1000)
        b1_mia.append(auc_b1)
        b1_tpr.append(tpr_b1)
        print(f"  B1  MIA AUC={auc_b1:.4f}  TPR@0.01={tpr_b1:.4f}")

        # B2
        set_all_seeds(seed)
        m = create_model(data, embed_dim=args.embed_dim)
        m, _ = train_dp(m, data, target_epsilon=args.epsilon, delta=1e-5,
                        batch_size=args.batch_size, epochs=args.epochs,
                        device=args.device, verbose=False)
        auc_b2, tpr_b2 = run_lira_attack(m, in_loader, out_loader,
                                          mu_in, sigma_in, mu_out, sigma_out,
                                          device=args.device, max_samples=1000)
        b2_mia.append(auc_b2)
        b2_tpr.append(tpr_b2)
        print(f"  B2  MIA AUC={auc_b2:.4f}  TPR@0.01={tpr_b2:.4f}")

        # B5
        set_all_seeds(seed)
        m = create_model(data, embed_dim=args.embed_dim)
        m, _ = train_dp(m, data, target_epsilon=args.epsilon, delta=1e-5,
                        batch_size=args.batch_size, epochs=args.epochs,
                        device=args.device, verbose=False)
        apply_rotemb(m)
        auc_b5, tpr_b5 = run_lira_attack(m, in_loader, out_loader,
                                          mu_in, sigma_in, mu_out, sigma_out,
                                          device=args.device, max_samples=1000)
        b5_mia.append(auc_b5)
        b5_tpr.append(tpr_b5)
        print(f"  B5  MIA AUC={auc_b5:.4f}  TPR@0.01={tpr_b5:.4f}")

    results = {
        "b1_mia_auc_mean": float(np.mean(b1_mia)), "b1_mia_auc_std": float(np.std(b1_mia)),
        "b1_tpr_001_mean": float(np.mean(b1_tpr)), "b1_tpr_001_std": float(np.std(b1_tpr)),
        "b2_mia_auc_mean": float(np.mean(b2_mia)), "b2_mia_auc_std": float(np.std(b2_mia)),
        "b2_tpr_001_mean": float(np.mean(b2_tpr)), "b2_tpr_001_std": float(np.std(b2_tpr)),
        "b5_mia_auc_mean": float(np.mean(b5_mia)), "b5_mia_auc_std": float(np.std(b5_mia)),
        "b5_tpr_001_mean": float(np.mean(b5_tpr)), "b5_tpr_001_std": float(np.std(b5_tpr)),
    }
    logger.log_metrics(results)
    logger.close()

    print("\n" + "=" * 60)
    print("Summary:")
    print(f"  B1  MIA AUC = {results['b1_mia_auc_mean']:.4f} ± {results['b1_mia_auc_std']:.4f}")
    print(f"  B2  MIA AUC = {results['b2_mia_auc_mean']:.4f} ± {results['b2_mia_auc_std']:.4f}")
    print(f"  B5  MIA AUC = {results['b5_mia_auc_mean']:.4f} ± {results['b5_mia_auc_std']:.4f}")


if __name__ == "__main__":
    main()
