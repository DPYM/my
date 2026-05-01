"""
Experiment 4 — EIA (Embedding Inversion Attack) evaluation.

Measures linear probe R² for all 5 baselines:
  B1 (No Protection):  R² ≈ 1.0
  B2 (DP-SGD ε=4):    R² ≈ 1.0 (DP doesn't protect embeddings)
  B3 (DP-SGD ε=8):    R² ≈ 1.0
  B4 (RotEmb only):    R² ≈ 0.0
  B5 (DualGuard):      R² ≈ 0.0

Usage:
  python -m src.experiments.run_exp4_eia --dataset criteo --data_path ./data/train.txt
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
from src.attacks.eia_inversion import linear_probe_r2, linear_probe_r2_from_tensors
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
    parser.add_argument("--epsilon", type=float, default=4.0)
    parser.add_argument("--n_seeds", type=int, default=5)
    args = parser.parse_args()

    logger = ExperimentLogger(args.out_dir, "exp4_eia")
    logger.log_config(dataset=args.dataset, epsilon=args.epsilon,
                      embed_dim=args.embed_dim, n_seeds=args.n_seeds)

    print("=" * 60)
    print(f"Experiment 4 — EIA (Linear Probe R²)  ({args.dataset})")
    load_fn = load_criteo if args.dataset == "criteo" else load_avazu
    data = load_data(args.data_path, load_fn, nrows=args.nrows)

    seeds = SEEDS[:args.n_seeds]
    all_r2 = {"b1": [], "b2": [], "b3": [], "b4": [], "b5": []}

    for seed in seeds:
        print(f"\n--- Seed {seed} ---")

        # B1 — No Protection
        set_all_seeds(seed)
        m_b1 = create_model(data, embed_dim=args.embed_dim)
        train_plain(m_b1, data, batch_size=args.batch_size, epochs=args.epochs,
                    device=args.device, verbose=False)
        r2_b1 = linear_probe_r2(m_b1, m_b1)
        all_r2["b1"].append(r2_b1)
        print(f"  B1 (No Protection)   R² = {r2_b1:.4f}")

        # B2 — DP-SGD ε=4
        set_all_seeds(seed)
        m_b2 = create_model(data, embed_dim=args.embed_dim)
        m_b2, _ = train_dp(m_b2, data, target_epsilon=args.epsilon, delta=1e-5,
                           batch_size=args.batch_size, epochs=args.epochs,
                           device=args.device, verbose=False)
        V_b2_orig = m_b2.embedding.weight.detach().cpu().clone()
        r2_b2 = linear_probe_r2_from_tensors(V_b2_orig, V_b2_orig)
        all_r2["b2"].append(r2_b2)
        print(f"  B2 (DP-SGD ε={args.epsilon})   R² = {r2_b2:.4f}")

        # B3 — DP-SGD ε=8
        set_all_seeds(seed)
        m_b3 = create_model(data, embed_dim=args.embed_dim)
        m_b3, _ = train_dp(m_b3, data, target_epsilon=8.0, delta=1e-5,
                           batch_size=args.batch_size, epochs=args.epochs,
                           device=args.device, verbose=False)
        V_b3_orig = m_b3.embedding.weight.detach().cpu().clone()
        r2_b3 = linear_probe_r2_from_tensors(V_b3_orig, V_b3_orig)
        all_r2["b3"].append(r2_b3)
        print(f"  B3 (DP-SGD ε=8)      R² = {r2_b3:.4f}")

        # B4 — RotEmb only
        set_all_seeds(seed)
        m_b4 = create_model(data, embed_dim=args.embed_dim)
        train_plain(m_b4, data, batch_size=args.batch_size, epochs=args.epochs,
                    device=args.device, verbose=False)
        V_orig = m_b4.embedding.weight.detach().cpu().clone()
        apply_rotemb(m_b4)
        V_rot = m_b4.embedding.weight.detach().cpu()
        r2_b4 = linear_probe_r2_from_tensors(V_orig, V_rot)
        all_r2["b4"].append(r2_b4)
        print(f"  B4 (RotEmb only)     R² = {r2_b4:.4f}")

        # B5 — DualGuard
        set_all_seeds(seed)
        m_b5 = create_model(data, embed_dim=args.embed_dim)
        m_b5, _ = train_dp(m_b5, data, target_epsilon=args.epsilon, delta=1e-5,
                           batch_size=args.batch_size, epochs=args.epochs,
                           device=args.device, verbose=False)
        V_b5_orig = m_b5.embedding.weight.detach().cpu().clone()
        apply_rotemb(m_b5)
        V_b5_rot = m_b5.embedding.weight.detach().cpu()
        r2_b5 = linear_probe_r2_from_tensors(V_b5_orig, V_b5_rot)
        all_r2["b5"].append(r2_b5)
        print(f"  B5 (DualGuard)       R² = {r2_b5:.4f}")

    results = {}
    for key in ["b1", "b2", "b3", "b4", "b5"]:
        results[f"{key}_r2_mean"] = float(np.mean(all_r2[key]))
        results[f"{key}_r2_std"] = float(np.std(all_r2[key]))

    logger.log_metrics(results)
    logger.close()

    print("\n" + "=" * 60)
    print("Summary:")
    for key, label in [("b1", "No Protection"), ("b2", f"DP-SGD ε={args.epsilon}"),
                       ("b3", "DP-SGD ε=8"), ("b4", "RotEmb only"), ("b5", "DualGuard")]:
        print(f"  {key.upper()} ({label:20s})  R² = {results[f'{key}_r2_mean']:.4f} ± {results[f'{key}_r2_std']:.4f}")


if __name__ == "__main__":
    main()
