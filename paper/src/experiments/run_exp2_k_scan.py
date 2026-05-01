"""
Experiment 2 — EIA R² vs k (embedding dimension) sweep.

Trains plain models at different k values, applies RotEmb,
and measures linear probe R². Expect R² → 0 as k increases.

Usage:
  python -m src.experiments.run_exp2_k_scan --dataset criteo --data_path ./data/train.txt
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
from src.training.trainers import train_plain, create_model
from src.attacks.eia_inversion import linear_probe_r2_from_tensors
from src.utils.logger import ExperimentLogger

SEEDS = [42, 123, 456, 789, 1024]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="criteo", choices=["criteo", "avazu"])
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--nrows", type=int, default=8_000_000)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch_size", type=int, default=4096)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--out_dir", default="./results")
    parser.add_argument("--ks", nargs="+", type=int, default=[8, 16, 32, 64])
    parser.add_argument("--n_seeds", type=int, default=5)
    args = parser.parse_args()

    logger = ExperimentLogger(args.out_dir, "exp2_k_scan")
    logger.log_config(dataset=args.dataset, nrows=args.nrows, ks=args.ks,
                      batch_size=args.batch_size, epochs=args.epochs, n_seeds=args.n_seeds)

    print("=" * 60)
    print(f"Experiment 2 — EIA R² vs k  ({args.dataset})")
    load_fn = load_criteo if args.dataset == "criteo" else load_avazu
    data = load_data(args.data_path, load_fn, nrows=args.nrows)

    seeds = SEEDS[:args.n_seeds]
    r2_results = {k_val: [] for k_val in args.ks}

    for k_val in args.ks:
        for seed in seeds:
            set_all_seeds(seed)
            m = DeepFM(sparse_vocab_sizes=data["vocab_sizes"],
                       n_dense=data["n_dense"],
                       embed_dim=k_val,
                       dnn_hidden_units=(400, 400, 400))
            train_plain(m, data, batch_size=args.batch_size, epochs=args.epochs,
                        device=args.device, verbose=False)

            V_orig = m.embedding.weight.detach().cpu().clone()
            apply_rotemb(m)
            V_rot = m.embedding.weight.detach().cpu()

            r2 = linear_probe_r2_from_tensors(V_orig, V_rot)
            r2_results[k_val].append(r2)

        mean_r2 = np.mean(r2_results[k_val])
        std_r2 = np.std(r2_results[k_val])
        print(f"  k={k_val:2d}  R² = {mean_r2:.4f} ± {std_r2:.4f}")

    results = {
        "ks": args.ks,
        "r2_means": [float(np.mean(r2_results[k])) for k in args.ks],
        "r2_stds": [float(np.std(r2_results[k])) for k in args.ks],
    }
    logger.log_metrics(results)
    logger.close()


if __name__ == "__main__":
    main()
