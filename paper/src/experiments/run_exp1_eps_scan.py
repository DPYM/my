"""
Experiment 1 — AUC vs ε sweep.

Trains models at different ε values and plots AUC curves for:
  B1 (No Protection): horizontal baseline
  B2 (DP-SGD only): AUC vs ε
  B5 (DualGuard): AUC vs ε (should overlap B2)

Usage:
  python -m src.experiments.run_exp1_eps_scan --dataset criteo --data_path ./data/train.txt
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
from src.training.trainers import train_plain, train_dp, create_model
from src.eval.metrics import evaluate_model
from src.utils.logger import ExperimentLogger

SEEDS = [42, 123, 456, 789, 1024]


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
    parser.add_argument("--epsilons", nargs="+", type=float, default=[0.5, 1, 2, 4, 8, 16])
    parser.add_argument("--n_seeds", type=int, default=5)
    args = parser.parse_args()

    logger = ExperimentLogger(args.out_dir, "exp1_eps_scan")
    logger.log_config(dataset=args.dataset, nrows=args.nrows, epsilons=args.epsilons,
                      embed_dim=args.embed_dim, batch_size=args.batch_size,
                      epochs=args.epochs, n_seeds=args.n_seeds)

    print("=" * 60)
    print(f"Experiment 1 — AUC vs ε  ({args.dataset})")
    load_fn = load_criteo if args.dataset == "criteo" else load_avazu
    data = load_data(args.data_path, load_fn, nrows=args.nrows)
    print(f"  Train: {len(data['y_train']):,}  Val: {len(data['y_val']):,}  Test: {len(data['y_test']):,}")

    seeds = SEEDS[:args.n_seeds]

    # B1 — No Protection
    b1_aucs = []
    for seed in seeds:
        set_all_seeds(seed)
        m = create_model(data, embed_dim=args.embed_dim)
        train_plain(m, data, batch_size=args.batch_size, epochs=args.epochs, device=args.device, verbose=False)
        auc, _ = evaluate_model(m, data, "test", device=args.device)
        b1_aucs.append(auc)
    b1_mean, b1_std = np.mean(b1_aucs), np.std(b1_aucs)
    print(f"  B1 (No Protection)  AUC = {b1_mean:.4f} ± {b1_std:.4f}")

    # B2 & B5 — sweep ε
    b2_results = {eps: [] for eps in args.epsilons}
    b5_results = {eps: [] for eps in args.epsilons}

    for eps in args.epsilons:
        for seed in seeds:
            # B2 — DP-SGD only
            set_all_seeds(seed)
            m = create_model(data, embed_dim=args.embed_dim)
            m, actual_eps = train_dp(m, data, target_epsilon=eps, delta=1e-5,
                                     batch_size=args.batch_size, epochs=args.epochs,
                                     device=args.device, verbose=False)
            auc, _ = evaluate_model(m, data, "test", device=args.device)
            b2_results[eps].append(auc)

            # B5 — DualGuard (DP-SGD + RotEmb)
            set_all_seeds(seed)
            m = create_model(data, embed_dim=args.embed_dim)
            m, actual_eps = train_dp(m, data, target_epsilon=eps, delta=1e-5,
                                     batch_size=args.batch_size, epochs=args.epochs,
                                     device=args.device, verbose=False)
            apply_rotemb(m)
            auc, _ = evaluate_model(m, data, "test", device=args.device)
            b5_results[eps].append(auc)

        b2_mean = np.mean(b2_results[eps])
        b5_mean = np.mean(b5_results[eps])
        print(f"  ε={eps:5.1f}  B2 AUC={b2_mean:.4f}  B5 AUC={b5_mean:.4f}  Δ={abs(b2_mean-b5_mean):.6f}")

    results = {
        "b1_mean": b1_mean, "b1_std": b1_std,
        "epsilons": args.epsilons,
        "b2_means": [float(np.mean(b2_results[eps])) for eps in args.epsilons],
        "b2_stds": [float(np.std(b2_results[eps])) for eps in args.epsilons],
        "b5_means": [float(np.mean(b5_results[eps])) for eps in args.epsilons],
        "b5_stds": [float(np.std(b5_results[eps])) for eps in args.epsilons],
    }
    logger.log_metrics(results)
    logger.close()


if __name__ == "__main__":
    main()
