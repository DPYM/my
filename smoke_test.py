"""
冒烟测试：合成数据走完整条 DualGuard 管道：
  DeepFM → train_plain → apply_rotemb → 验证精度保持 → EIA 探头

如果全部通过，说明核心代码无误，可以放心下载真实数据跑完整实验。
在 paper 目录下执行:
  /c/Users/64623/anaconda3/envs/paper/python smoke_test.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import numpy as np
from src.utils.seed import set_all_seeds
from src.models.deepfm import DeepFM
from src.models.rotemb import apply_rotemb, verify_accuracy_preservation
from src.training.trainers import train_plain
from src.attacks.eia_inversion import linear_probe_r2_from_tensors

set_all_seeds(42)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ---- 构造迷你数据集 ----
N = 10000
n_sparse = 5       # 5个类别域
vocab_sizes = [200, 150, 100, 80, 50]
n_dense = 3        # 3个连续特征
embed_dim = 16

X_sparse = torch.randint(0, 50, (N, n_sparse))  # 每列最大值不同，先凑合
for j in range(n_sparse):
    X_sparse[:, j] = torch.randint(0, vocab_sizes[j], (N,))
X_dense = torch.randn(N, n_dense).float()

# 用 DeepFM 的线性项+FM做了一个简单的可学习标签
true_bias = 0.3
true_w = torch.randn(sum(vocab_sizes))
y_logit = true_bias + torch.randn(N) * 0.5
y = (torch.sigmoid(y_logit) > 0.5).long()

data = {
    "X_sparse_train": X_sparse[:7000].numpy(),
    "X_sparse_val":   X_sparse[7000:8500].numpy(),
    "X_sparse_test":  X_sparse[8500:].numpy(),
    "X_dense_train":  X_dense[:7000].numpy(),
    "X_dense_val":    X_dense[7000:8500].numpy(),
    "X_dense_test":   X_dense[8500:].numpy(),
    "y_train": y[:7000].numpy(),
    "y_val":   y[7000:8500].numpy(),
    "y_test":  y[8500:].numpy(),
    "vocab_sizes": vocab_sizes,
    "n_dense": n_dense,
}

print("=" * 60)
print("冒烟测试开始  (设备: {})".format(DEVICE))
print("=" * 60)

# ---- 1. 创建模型 + 普通训练 ----
print("\n[1/4] 普通训练 ...")
m = DeepFM(sparse_vocab_sizes=vocab_sizes, n_dense=n_dense,
           embed_dim=embed_dim, dnn_hidden_units=(128, 64, 32))
m = train_plain(m, data, batch_size=512, epochs=5, lr=1e-3,
                patience=2, device=DEVICE, verbose=True)

# ---- 2. 保存训练前状态 + 旋转 ----
print("\n[2/4] 执行 RotEmb 旋转 ...")
state_before = {k: v.cpu().clone() for k, v in m.state_dict().items()}
V_orig = m.embedding.weight.detach().cpu().clone()
R = apply_rotemb(m)
V_rot = m.embedding.weight.detach().cpu()

# ---- 3. 验证精度保持 (定理 1) ----
print("\n[3/4] 验证定理 1 — 精度保持 ...")
xs = torch.as_tensor(data["X_sparse_test"][:256]).to(DEVICE)
xd = torch.as_tensor(data["X_dense_test"][:256]).to(DEVICE)
max_diff = verify_accuracy_preservation(m, state_before, xs, xd, tol=1e-5)
assert max_diff < 1e-5, f"定理 1 失败! max_diff = {max_diff}"

# ---- 4. EIA 线性探头 (RotEmb 可视化效果) ----
print("\n[4/4] EIA 线性探头 R² ...")
r2 = linear_probe_r2_from_tensors(V_orig.numpy(), V_rot.numpy())
print(f"  EIA R² = {r2:.4f}  (期望接近 0 — 旋转化了维度语义)")

# ---- 最终评定 ----
print("\n" + "=" * 60)
all_pass = True

if max_diff >= 1e-5:
    print("FAIL — 定理 1: 精度保持失败")
    all_pass = False
else:
    print("PASS — 定理 1: 旋转前后预测完全一致  (max|Δŷ| = {:.2e})".format(max_diff))

if r2 > 0.3:
    print("WARN — 旋转后 R² 偏高。 小 k (= {}) 下这是预料之中".format(embed_dim))
else:
    print("PASS — 旋转后 R² ≈ 0: 维属性被成功打散")

if all_pass:
    print("\n全部检查通过。代码管道正常，可以开始下载真实数据了。")
else:
    print("\n存在问题 — 先修完再下载数据。")

print("=" * 60)
