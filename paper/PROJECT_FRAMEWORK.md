# DualGuard: 双层隐私保护框架 — 完整项目开发文档

## 项目概述

**DualGuard** 是一个面向 DeepFM 推荐模型的双层隐私保护框架，目标是投稿 CCF-C 会议。

| 阶段 | 技术 | 防御目标 |
|------|------|----------|
| 训练阶段 | DeepFM + DP-SGD (Opacus) | 保护训练数据，防 Membership Inference Attack (MIA) |
| 后处理阶段 | Embedding 随机旋转 (RotEmb) | 保护模型权重，防 Embedding Inversion Attack (EIA) |

**核心定理（3条，已证）：**
- **定理 1**：旋转 + W⁰ 补偿后模型输出完全相同（零精度代价）
- **定理 2**：对手无法区分旋转后的 embedding 方向（等价类引理）
- **定理 3**：最优重建估计器的 MSE 下界 = ‖v‖²，等于猜全零向量

---

## 项目路径

```
C:\github\paper\
```

## 代码结构

```
paper/
├── duanguard_theory.md          # 理论推导（已完成）
├── implementation_plan.md       # 实验方案（已完成）
├── literature_review.md         # 文献调研（已完成）
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── download.py          # 下载 Criteo + Avazu  [已完成]
│   │   └── preprocess.py        # 特征处理 + 归一化 + 时间划分 [已完成]
│   ├── models/
│   │   ├── __init__.py
│   │   ├── deepfm.py            # DeepFM 模型定义 [已完成]
│   │   └── rotemb.py            # 旋转变换 + W⁰ 补偿 [已完成]
│   ├── training/
│   │   ├── __init__.py
│   │   └── trainers.py          # 统一的训练入口 (plain + dp) [已完成]
│   ├── attacks/
│   │   ├── __init__.py
│   │   ├── mia_lira.py          # LiRA membership inference [已完成]
│   │   └── eia_inversion.py     # Embedding 重建 + 线性探测 [已完成, BUG已修复]
│   ├── eval/
│   │   ├── __init__.py
│   │   └── metrics.py           # AUC, LogLoss [已完成]
│   ├── experiments/
│   │   ├── __init__.py
│   │   ├── run_all.py           # 主实验脚本 [已完成, BUG已修复]
│   │   ├── run_exp1_eps_scan.py       # ε 扫描 [已完成]
│   │   ├── run_exp2_k_scan.py         # k 扫描 [已完成]
│   │   ├── run_exp3_mia.py            # MIA 评估 [已完成]
│   │   ├── run_exp4_eia.py            # EIA 评估 [已完成]
│   │   └── run_exp5_combination.py    # 组合实验 [已完成]
│   └── utils/
│       ├── __init__.py
│       ├── seed.py               # 全局 seed 控制 [已完成]
│       └── logger.py             # 实验日志 + tensorboard [已完成]
└── results/
    ├── figures/                   # 所有图 [待生成]
    └── tables/                    # LaTeX 格式表格 [待生成]
```

---

## 已完成模块的详细说明

### 1. DeepFM 模型 (`src/models/deepfm.py`)

定制实现的 DeepFM，关键设计：
- **单一 embedding 表**：所有 sparse feature 共享一个 `nn.Embedding(total_vocab, k)`，这样一次旋转就能保护所有 embedding
- **暴露 first_dnn_linear**：`self.first_dnn_linear` 指向 DNN 的第一个 `nn.Linear` 层，RotEmb 需要直接修改它的权重
- **前向传播**：linear 项 + FM 配对项 + DNN 项 → sigmoid

```python
class DeepFM(nn.Module):
    def __init__(self, sparse_vocab_sizes, n_dense, embed_dim=16, 
                 dnn_hidden_units=(400, 400, 400), dropout=0.2):
```

**输入格式**：
- `X_sparse`: (batch, n_sparse) — LongTensor，每个值是该 field 内的特征索引
- `X_dense`: (batch, n_dense) — FloatTensor，连续特征，可为 None

**偏移机制**：不同 field 的索引通过 `offsets` 映射到共享 embedding 表的不同区间。

---

### 2. RotEmb 旋转 (`src/models/rotemb.py`)

三个函数：

**`random_orthogonal(k)`**：从 Haar 测度采样随机正交矩阵
- 对 k×k 高斯矩阵做 QR 分解
- 修正对角符号确保均匀覆盖 O(k)

**`apply_rotemb(model, R=None)`**：原地修改模型
- `embedding.weight` ← `weight @ R.T`（旋转所有 embedding）
- `first_dnn_linear.weight` 的 sparse 部分 ← `W_sparse ⊗ R.T`（补偿 DNN 第一层）
- dense 部分不变
- 返回 R（需安全存储，不随模型部署）

**`verify_accuracy_preservation(model, before_state, ...)`**：精度验证
- 比较旋转前后的输出差异，应 < 1e-6

---

### 3. 训练模块 (`src/training/trainers.py`)

**`train_plain()`**：普通 Adam 训练，用于 B1 (No Protection) 和 B4 (RotEmb only)
- 早停 patience=3，基于 val AUC
- 保存最佳 checkpoint

**`train_dp()`**：DP-SGD 训练（需要 `pip install opacus`）
- 使用 Opacus PrivacyEngine，RDP 会计
- 二分搜索自动计算 noise_multiplier
- 隐私预算耗尽时提前停止

**`create_model(data_dict)`**：从预处理数据字典构建模型

**`make_dataloader(data_dict, split_name, ...)`**：创建 DataLoader

---

### 4. 数据预处理 (`src/data/preprocess.py`)

**`load_criteo(path, nrows, min_freq=10)`**：
- 13 个连续特征 (I1-I13) + 26 个类别特征 (C1-C26)
- 类别特征 < min_freq 的替换为 `<UNK>`
- QuantileTransformer 归一化（只 fit training 集）
- 时间序划分：80% train / 10% val / 10% test

**`load_avazu(path, nrows, min_freq=10)`**：
- 22 个类别特征，无连续特征
- 同样时间序划分
- Avazu 数据已按 hour 列时间排序

**返回格式**（统一字典）：
```python
{
    "X_sparse_train": (n_train, n_sparse) int64,
    "X_sparse_val":   (n_val, n_sparse) int64,
    "X_sparse_test":  (n_test, n_sparse) int64,
    "X_dense_train":  (n_train, n_dense) float32,  # 只有 Criteo
    "X_dense_val":    ...,
    "X_dense_test":   ...,
    "y_train":        (n_train,) int64,
    "y_val":          (n_val,) int64,
    "y_test":         (n_test,) int64,
    "vocab_sizes":    list[int]  # 每个 sparse field 的 vocab 大小,
    "n_dense":        int  # 连续特征数量,
}
```

---

### 5. 攻击模块

**MIA — LiRA (`src/attacks/mia_lira.py`)**：
- `collect_confidences()`：多轮 dropout 推理收集置信度
- `fit_lira_distributions()`：拟合 IN/OUT 高斯分布
- `lira_score()`：对数似然比
- `run_lira_attack()`：评估主实验的 MIA AUC 和 TPR@FPR=0.01

**EIA (`src/attacks/eia_inversion.py`)**：
- `linear_probe_r2()`：Ridge 回归从两个模型（旋转前后）的 embedding 预测原始维度值
- `linear_probe_r2_from_tensors()`：同上，但直接接受 tensor 参数（用于同一模型旋转前后对比）
- `label_propagation_accuracy()`：KMeans 聚类 + 锚点标签传播

---

## 已发现的 BUG（已全部修复）

### Bug 1：`eia_inversion.py` 第 76-77 行 — setdefault 不递增 ✅ 已修复

已改为显式 if-not-in 模式：
```python
if c not in anchor_votes:
    anchor_votes[c] = {}
if label not in anchor_votes[c]:
    anchor_votes[c][label] = 0
anchor_votes[c][label] += 1
```

### Bug 2：`run_all.py` Exp 2 — k 扫描对比无效 ✅ 已修复

已改为同一个模型旋转前后对比，使用新增的 `linear_probe_r2_from_tensors()`：
```python
V_orig = m.embedding.weight.detach().cpu().clone()
apply_rotemb(m)
V_rot = m.embedding.weight.detach().cpu()
r2 = linear_probe_r2_from_tensors(V_orig, V_rot)
```

### Bug 3：`run_all.py` MIA 评估使用启发式分布 ✅ 已修复

已实现完整的 LiRA 影子模型训练管线（K=4 shadow pairs），不再使用硬编码分布。

### Bug 4：`eia_inversion.py` `linear_probe_r2` 用相同数据训练和测试 ✅ 已修复

已改为 train/test split（默认 50%/50%）：
```python
n_train = int(n * train_ratio)
probe.fit(X[:n_train], y[:n_train])
y_pred = probe.predict(X[n_train:])
```

---

## 待完成的任务（按优先级排列）

### 高优先级 — 全部完成 ✅

1. ~~**修复上述 4 个 BUG**~~ ✅
2. ~~**编写 `src/data/download.py`**~~ ✅
3. ~~**实现完整的 LiRA 影子模型训练管线**（`run_exp3_mia.py`）~~ ✅
4. ~~**把 `run_all.py` 拆分成独立的实验脚本**~~ ✅

### 中优先级 — 全部完成 ✅

5. ~~**`src/utils/logger.py`**：TensorBoard 日志 + 实验配置记录~~ ✅
6. ~~**`run_exp5_combination.py`**：DP-SGD only vs RotEmb only vs DualGuard (ε=2,4,8) 的组合对比~~ ✅

### 低优先级（待完成）

7. **结果可视化脚本**：读取 `results.json` 生成论文用的 6 张图
8. **LaTeX 表格生成**：自动生成论文表格
9. **单元测试**：验证定理 1（旋转精度保持）、定理 2（等价类）、定理 3（MSE 下界）
10. **layer-wise clipping 支持**：`trainers.py` 中给不同参数组不同的 max_grad_norm

---

## 实验设计（5组Baseline + 5个实验）

### Baseline

| # | 名称 | 训练方式 | 后处理 | 防御 MIA | 防御 EIA |
|---|------|----------|--------|----------|----------|
| B1 | No Protection | 普通 SGD | 无 | 否 | 否 |
| B2 | DP-SGD only | DP-SGD ε=4 | 无 | 是 | 否 |
| B3 | DP-SGD only (loose) | DP-SGD ε=8 | 无 | 是（弱） | 否 |
| B4 | RotEmb only | 普通 SGD | 旋转 | 否 | 是 |
| B5 | DualGuard (ours) | DP-SGD + RotEmb | 旋转 | 是 | 是 |

### 实验

| 实验 | 内容 | 预期结果 |
|------|------|----------|
| Exp 1 | AUC vs ε 扫描 {0.5, 1, 2, 4, 8, 16, ∞} | B2和B5曲线完全重叠 |
| Exp 2 | EIA R² vs k 扫描 {8, 16, 32, 64} | R² 随 k 增大单调下降 |
| Exp 3 | MIA LiRA 评估 @ ε=4 | B1 AUC≈0.75, B2/B5 AUC≤0.55 |
| Exp 4 | EIA 线性探测 R² | B1/B2≈1.0, B4/B5≈0.0 |
| Exp 5 | 组合强度实验 | 验证双层互补 |

### 论文核心图（6张）

1. **AUC vs ε**：B2 和 B5 曲线重叠下降，B1 水平虚线在上方
2. **MIA TPR @ FPR=0.01 vs ε**：B1 高、B2/B5 低且重叠
3. **EIA R² vs ε**：B1 高（≈1.0）、B4/B5 低（≈0.0）、B2 中高
4. **EIA R² vs k**：R² 随 k 增大单调下降
5. **训练时间对比**：柱状图
6. **精度损失分解**：堆叠柱状图

---

## 技术决策记录

1. **只用两个数据集**（Criteo + Avazu），不用 MovieLens（太小，DP 噪声下不收敛）
2. **固定超参不调**：贡献在隐私框架，不在 SOTA 精度
3. **时间序划分**（非随机），避免 CTR 场景的数据泄露
4. **δ = 10⁻⁵**：标准选择，远小于 1/N
5. **每实验 5 个 seed**：DP-SGD 噪声方差大，需报告 mean±std（Exp 3 MIA 例外—影子模型开销大，默认 3 seeds）
6. **硬裁剪 + 分层 C**：Embedding 和 DNN 层用不同的 max_grad_norm
7. **旋转后不叠加噪声**（论文 Discussion 部分可讨论这个扩展）

---

## 依赖

```
torch>=2.0
opacus>=1.4
scikit-learn
numpy
pandas
matplotlib
tensorboard
kagglehub  (或 kaggle CLI)
```

---

## 运行命令示例

```bash
# 下载数据
python -m src.data.download --dataset criteo --out_dir ./data
python -m src.data.download --dataset avazu --out_dir ./data

# 预处理
python -m src.data.preprocess --dataset criteo --data_path ./data/train.txt --nrows 10000000

# 运行实验
python -m src.experiments.run_exp1_eps_scan --dataset criteo --data_path ./data/train.txt --nrows 10000000
python -m src.experiments.run_exp2_k_scan   --dataset criteo --data_path ./data/train.txt
python -m src.experiments.run_exp3_mia      --dataset criteo --data_path ./data/train.txt
python -m src.experiments.run_exp4_eia      --dataset criteo --data_path ./data/train.txt
python -m src.experiments.run_exp5_combination --dataset criteo --data_path ./data/train.txt

# 或一键运行全部
python -m src.experiments.run_all --dataset criteo --data_path ./data/train.txt --nrows 10000000
```
