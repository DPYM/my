# DualGuard 完整实验实现方案

## 一、数据集选择

### 主实验数据集（两个必须跑）

| 数据集 | 来源 | 样本量 | 特征域数 | 任务 | 用途 |
|--------|------|--------|----------|------|------|
| **Criteo** | Kaggle Criteo Display Ads | ~45M（用前10M） | 39（13连续+26类别） | CTR二分类 | 主实验 |
| **Avazu** | Kaggle Avazu CTR Prediction | ~40M（用前8M） | 22（类别为主） | CTR二分类 | 验证泛化性 |

**为什么选这两个**：
- Criteo 是 CTR 预测领域**最标准的 benchmark**，所有 DeepFM 论文都用
- Avazu 特征构成不同（类别特征占比更高），验证方法不挑数据
- 两个都是公开 Kaggle 数据集，审稿人可复现

### 选这两个就够了吗

够。CCF-C 会议不要求 5-6 个数据集。两个大规模真实 CTR 数据集跑透、把实验做扎实，比堆砌数据集更可信。Movielens 这种小数据集（10万条）跑 DeepFM 本身就是过拟合，加了 DP 噪声后 AUC 方差巨大，反而削弱结论可信度。

### 数据预处理（每一步精确到代码）

**步骤 1**：下载原始数据

```bash
# Criteo: 下载 kaggle competitions download -c criteo-display-ad-challenge
# 得到 train.txt (约11GB)
# Avazu: 下载 kaggle competitions download -c avazu-ctr-prediction
# 得到 train.gz (约1GB)
```

**步骤 2**：取子集

```python
import pandas as pd

# Criteo: 取前 1000万行
criteo = pd.read_csv('train.txt', sep='\t', nrows=10_000_000)

# Avazu: 取前 800万行
avazu = pd.read_csv('train.gz', nrows=8_000_000)
```

**步骤 3**：时间划分训练/验证/测试集（**不是随机划分**）

CTR 场景有时间偏移（用户行为随时间变化），随机划分会导致数据泄露。必须按时间切分：

```python
# Criteo 数据已是时间排序的
n = len(criteo)
train_end = int(n * 0.8)
val_end   = int(n * 0.9)

train = criteo[:train_end]
val   = criteo[train_end:val_end]
test  = criteo[val_end:]
```

**步骤 4**：特征处理

连续特征：用训练集的分位数做归一化（clipping 到 [0.01, 0.99] 分位再 min-max 到 [-1,1]），避免极端值：

```python
from sklearn.preprocessing import QuantileTransformer

qt = QuantileTransformer(
    n_quantiles=1000,
    output_distribution='uniform',
    subsample=200_000
)
X_cont_train = qt.fit_transform(X_cont_train) * 2 - 1  # 映射到 [-1, 1]
```

类别特征：低频值（出现次数 < 10）替换为 `<UNK>` token。

**步骤 5**：负采样处理类别不平衡

CTR = 1 的比例约 25%，对于 MIA 实验这不是问题。但为公平对比，验证集和测试集保持原始分布。

---

## 二、模型与训练配置

### DeepFM 结构（固定超参，不调）

| 参数 | 值 | 说明 |
|------|-----|------|
| Embedding dim $k$ | 16 | 标准配置 |
| DNN layers | [400, 400, 400] | 3层 |
| DNN activation | ReLU | |
| Dropout | 0.5 | 仅测试时对比 |
| Batch size | 4096 | |
| Optimizer | Adam | lr=1e-3 |
| Epochs | 10 | 带早停 patience=3 |

**为什么固定超参不调**：我们的贡献是隐私框架，不是 SOTA 精度。调参改善的 AUC 会被归因为 engineering 而非方法本身，反而降低可信度。

### DP-SGD 配置

| 参数 | 探索范围 |
|------|----------|
| $\varepsilon$ target | $\{1, 2, 4, 8, \infty\}$ |
| $\delta$ | $10^{-5}$（标准选择，远小于 1/N） |
| Clipping norm $C$ | $\{1.0, 5.0, 10.0\}$（网格搜索，选验证集最佳） |
| Noise multiplier $\sigma$ | 由 Opacus privacy engine 根据 $\varepsilon$ 目标自动计算 |

**$\delta = 10^{-5}$ 的解释**：
- Criteo 用 1000万行，$1/N \approx 10^{-7}$
- $\delta \ll 1/N$ 是标准做法
- 所有 DP 论文（Abadi et al.）都用这个数量级

---

## 三、Baseline 体系

### 必须跑的 5 个 baseline

| # | 名称 | 训练方式 | 后处理 | 防御 MIA | 防御 EIA | 作用 |
|---|------|----------|--------|----------|----------|------|
| B1 | **No Protection** | 普通 SGD | 无 | 否 | 否 | 精度上界 |
| B2 | **DP-SGD only** | DP-SGD, $\varepsilon$=4 | 无 | 是 | 否 | 验证 DP 开销 |
| B3 | **DP-SGD only (loose)** | DP-SGD, $\varepsilon$=8 | 无 | 是（弱） | 否 | 精度-MIA tradeoff |
| B4 | **RotEmb only** | 普通 SGD | 旋转 | 否 | 是 | 验证旋转开销（期望零） |
| B5 | **DualGuard (ours)** | DP-SGD + RotEmb | 旋转 | 是 | 是 | 我们的方法 |

### 为什么要跑 B4

B4 验证 RotEmb 真的零精度代价。如果 B4 的 AUC != B1 的 AUC，说明代码有 bug。这是自检环节，不能省略。

### 扩展实验（如有余力）

- **B6：Laplace noise on embeddings**：普通训练 + 直接在 embedding 上加 Laplace 噪声（不用旋转），对比精度下降 vs RotEmb，说明旋转的价值
- **B7：Random Projection**：用随机投影代替旋转做降维保护，验证正交矩阵的必要性

---

## 四、评估协议

### 4.1 效用评估：AUC + LogLoss

每次实验跑 5 个不同的 random seed（控制参数初始化、batch 顺序、DP 噪声），报告 mean ± std。

```python
seeds = [42, 123, 456, 789, 1024]
auc_scores = []

for seed in seeds:
    set_all_seeds(seed)
    model = train_deepfm(config, dp_config)
    auc = evaluate(model, test_set)
    auc_scores.append(auc)

print(f"AUC: {np.mean(auc_scores):.4f} ± {np.std(auc_scores):.4f}")
```

**为什么 5 个 seed**：DP-SGD 的噪声随机性较大，单次运行不可信。5 次报告的均值和标准差是标准做法。

**不用 paired test（如 paired t-test）比较模型**：因为实验目的不是证明 DualGuard 显著优于某 baseline（它必然有精度损失），而是报告清楚 utility-privacy tradeoff 曲线。

### 4.2 MIA 评估：LiRA Attack

用 **LiRA (Likelihood Ratio Attack, Carlini et al. 2022)**，这是目前最强的 MIA 方法。如果 LiRA 打不穿，更弱的攻击方法更打不穿。

**具体操作流程**：

**第一步**：训练 reference models（影子模型）

```python
# 从训练集中随机采样 4 个子集，每个子集与原始训练集是随机对半划分
# 训练 4 × 2 = 8 个影子模型（4个 IN 模型 + 4个 OUT 模型）
# 这些都是"没有 DP 保护的模型"，帮助攻击者校准似然比

shadow_models_in  = []  # 目标样本在训练集中
shadow_models_out = []  # 目标样本不在训练集中

for i in range(4):
    half = random.sample(range(n_train), n_train // 2)
    shadow_in  = train_plain_deepfm(train_data[half])
    shadow_out = train_plain_deepfm(train_data[complement(half)])
    shadow_models_in.append(shadow_in)
    shadow_models_out.append(shadow_out)
```

**第二步**：对每个影子模型的每个样本，收集置信度

```python
def get_confidence_vector(model, sample):
    """获取模型对样本的多轮推理置信度分布"""
    # 使用多种数据增强（随机 dropout、输入扰动）做多次推理
    # 得到置信度向量的分布
    confs = []
    model.train()  # 保持 dropout 活跃
    for _ in range(100):
        with torch.no_grad():
            prob = model(sample).item()
        confs.append(prob)
    return confs
```

**第三步**：训练攻击模型

对每个目标样本，用影子模型的置信度分布拟合两个高斯分布的似然比：

$$\Lambda(\mathbf{x}, y) = \frac{\mathcal{N}(\text{conf} \mid \mu_{\text{in}}, \sigma_{\text{in}}^2)}{\mathcal{N}(\text{conf} \mid \mu_{\text{out}}, \sigma_{\text{out}}^2)}$$

$\Lambda > 1$ → 预测 IN，$\Lambda \leq 1$ → 预测 OUT。

**第四步**：对目标模型（我们的 baseline / DualGuard）执行攻击

```python
# 取测试集的一半作为"IN"样本（假设它们在训练集中）
# 取验证集作为"OUT"样本（它们肯定不在训练集中）
# 报告 TPR @ low FPR (如 FPR=0.01, 0.001)

def evaluate_mia(target_model, in_samples, out_samples, attack_model):
    scores_in  = [lira_score(target_model, x) for x in in_samples]
    scores_out = [lira_score(target_model, x) for x in out_samples]

    # 计算 TPR @ FPR=0.01
    threshold = np.percentile(scores_out, 99)
    tpr = np.mean(scores_in > threshold)
    return tpr
```

**第五步**：报告指标

| 指标 | 含义 | 无保护期望值 | 有效防御值 |
|------|------|-------------|-----------|
| **MIA AUC** | 攻击者区分 IN/OUT 的 AUC | ~0.70-0.85 | ≤ 0.55 |
| **TPR @ FPR=0.01** | 低假正例率下的真阳性率 | ~0.30-0.50 | ≤ 0.02 |

**为什么用 TPR @ low FPR**：实际场景中攻击者关心的是"在几乎不误判的前提下能抓住多少"，而不是全局 AUC。

### 4.3 EIA 评估：Embedding 重建攻击

**自建攻击流程**（论文里没有标准 EIA 测试库，需要自己实现）：

**第一步**：准备半监督标签

从测试集中抽样 10% 的特征（如某些 category features 的值），赋予已知语义标签。实际操作：对 Criteo 的 26 个类别特征，随机挑 3 个，取它们最常见的 50 个值，认为是"已知语义"的锚点。

**第二步**：在旋转后 embedding 空间做 KMeans 聚类（K=100）

```python
from sklearn.cluster import KMeans

embeddings = extract_all_embeddings(deployed_model)  # 获取所有 ṽ_f
kmeans = KMeans(n_clusters=100, random_state=0)
clusters = kmeans.fit_predict(embeddings)
```

**第三步**：用锚点给簇贴标签

```python
# 对每个锚点，看它落在哪个簇
# 将锚点的语义标签传播到同簇的其他特征
# 计算传播准确率

correct = 0
total = 0
for anchor in anchor_set:
    anchor_cluster = clusters[anchor_idx[anchor]]
    same_cluster_items = [i for i, c in enumerate(clusters) if c == anchor_cluster]
    for item in same_cluster_items:
        if item != anchor and item in labeled_test_set:
            if true_label(item) == true_label(anchor):
                correct += 1
            total += 1

label_propagation_accuracy = correct / total
```

**第四步**：报告

| 指标 | 无保护 | RotEmb | 解释 |
|------|--------|--------|------|
| 标签传播准确率 | ~0.60-0.80 | ~0（接近随机聚类的标签传播） | 聚类有意义→传播有效 |
| 重建 MSE / $\|\mathbf{v}\|^2$ | 0.0（直接看） | → 1.0（定理 3） | 方向重建误差 |

### 4.4 EIA 补充验证：线性探测 (Linear Probe)

训练一个线性分类器，用旋转后的 embedding 维度预测原始 embedding 的某个维度值：

```python
from sklearn.linear_model import Ridge

# 对原始 embedding v 的第 j 维
# 用旋转后 embedding ṽ 的所有维度做特征，预测 v_j
probe = Ridge(alpha=1.0)
probe.fit(tilde_V, V_original[:, j])  # 用一半特征训练
r2 = probe.score(tilde_V_test, V_original_test[:, j])  # 用另一半测试

# 无保护：R² ≈ 1.0（直接取第 j 维就是完美的）
# RotEmb：R² ≈ 0.0（任何维度都无法线性预测原始维度）
```

这个实验直观、易于理解，审稿人看完就明白"旋转让每个维度失去了语义"。

---

## 五、参数扫描实验

### 5.1 $\varepsilon$ 扫描

固定其他参数，扫描 $\varepsilon \in \{0.5, 1, 2, 4, 8, 16, \infty\}$。

画三条曲线在同一张图上：
- B1（No DP）：一条水平虚线（精度上界）
- B2（DP-SGD only）：AUC 随 $\varepsilon$ 下降
- B5（DualGuard）：AUC 随 $\varepsilon$ 下降（期望与 B2 完全重叠，因为 RotEmb 不增精度损失）

**关键看点**：B2 和 B5 的曲线完全重叠 → 证明 RotEmb 零额外效用损失。

### 5.2 Embedding 维度 $k$ 扫描

固定 $\varepsilon=4$，扫描 $k \in \{8, 16, 32, 64\}$。

- EIA 线性探测 $R^2$ 随 $k$ 增大 → 0（$k$ 越大，旋转混淆效果越好，因为原始语义被稀释到更多维度）
- 画 EIA $R^2$ vs $k$ 曲线

### 5.3 组合强度实验

固定 $\varepsilon=4$，对比：
- DP-SGD only
- RotEmb only
- DualGuard ($\varepsilon=4$ + 旋转)
- DualGuard ($\varepsilon=2$ + 旋转) — 用更小的 $\varepsilon$ 能否在旋转帮助下仍维持可接受精度
- DualGuard ($\varepsilon=8$ + 旋转) — 用更大的 $\varepsilon$ 确认 MIA 侧的上限

---

## 六、可信度保证清单

### 6.1 防止数据泄露

| 风险 | 防护措施 |
|------|----------|
| 特征归一化用到了测试集 | 所有 scaler **只对训练集 fit**，transform 应用到 val/test |
| 时间穿越 | 按时间戳划分（非随机），训练集时间 < 验证集 < 测试集 |
| MIA 的 IN/OUT 标签泄露 | 影子模型训练时严格分离，不共享任何数据 |
| Embedding 重建的标签泄露 | 锚点只用训练集特征，不在测试集上选锚点 |

### 6.2 可复现性

```python
def set_all_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# 每次实验前调用 set_all_seeds(seed_value)
# 论文附录列出所有用到的 seed
```

### 6.3 硬件与时间

统一在相同 GPU 上跑所有实验。记录每轮实验的 wall-clock 时间。论文中报告：

| 方法 | GPU | 训练时间 | RotEmb 时间 | 总时间 |
|------|-----|----------|-------------|--------|
| DP-SGD only | 单卡 V100 32GB | ~X hours | 0 | X |
| DualGuard | 单卡 V100 32GB | ~X hours | ~Y seconds | X+Y |

RotEmb 的后处理时间应该是以秒计的（一次性旋转 + 补偿 W⁰），相比训练时间可忽略。

### 6.4 代码结构

```
paper/
├── duanguard_theory.md          # 理论推导
├── src/
│   ├── data/
│   │   ├── download.py          # 下载 Criteo + Avazu
│   │   ├── preprocess.py        # 特征处理 + 归一化
│   │   └── split.py             # 时间划分
│   │
│   ├── models/
│   │   ├── deepfm.py            # DeepFM 模型定义
│   │   └── rotemb.py            # 旋转变换 + W⁰ 补偿
│   │
│   ├── training/
│   │   ├── train_plain.py       # 普通 SGD 训练
│   │   ├── train_dp.py          # DP-SGD (Opacus) 训练
│   │   └── train_duanguard.py   # DP-SGD 训练 + RotEmb 后处理
│   │
│   ├── attacks/
│   │   ├── mia_lira.py          # LiRA membership inference
│   │   └── eia_inversion.py     # Embedding 重建 + 线性探测
│   │
│   ├── eval/
│   │   └── metrics.py           # AUC, LogLoss, MIA AUC, EIA R²
│   │
│   ├── experiments/
│   │   ├── run_exp1_eps_scan.py       # ε 扫描
│   │   ├── run_exp2_k_scan.py         # k 扫描
│   │   ├── run_exp3_mia.py            # MIA 评估
│   │   ├── run_exp4_eia.py            # EIA 评估
│   │   └── run_exp5_combination.py    # 组合实验
│   │
│   └── utils/
│       ├── seed.py               # 全局 seed 控制
│       └── logger.py             # 实验日志 + tensorboard
│
└── results/
    ├── figures/                   # 所有图
    └── tables/                    # LaTeX 格式表格
```

---

## 七、预期结果与结论支撑

### 关键实验结果的预期值

| 实验 | 预期结果 | 结论 |
|------|----------|------|
| B1 vs B4 AUC | 完全相等（差值 < 1e-6） | RotEmb 零精度代价 ✅ |
| B2 vs B5 AUC @ 相同 ε | 完全相等 | DP + 旋转不叠加效用损失 ✅ |
| B1 MIA AUC | ~0.75 | 无保护模型容易攻破 |
| B2 MIA AUC @ ε=4 | ~0.53-0.55 | DP 有效降低 MIA |
| B2 MIA AUC @ ε=8 | ~0.58-0.62 | 宽松 ε 下 MIA 防御减弱 |
| B1 EIA R² (线性探测) | ~1.0 | 无保护可直接按维度读 |
| B4 EIA R² (线性探测) | ~0.0 | 旋转消灭维度语义 ✅ |
| B5 EIA R² @ ε=4 | ~0.0 | 双层保护同时有效 ✅ |

### 论文核心图（6张）

1. **AUC vs ε**：B2 和 B5 曲线重叠下降，B1 水平虚线在上方
2. **MIA TPR @ FPR=0.01 vs ε**：B1 高、B2/B5 低且曲线重叠
3. **EIA R² vs ε**：B1 高（≈1.0）、B4/B5 低（≈0.0）、B2 中高（DP 不防 EIA）
4. **EIA R² vs k（Embedding 维度）**：R² 随 k 增大单调下降
5. **训练时间对比**：柱状图，DP-SGD vs DualGuard（几乎一样高，RotEmb 看不出）
6. **精度损失分解**：堆叠柱状图，DP 噪声导致的损失占比 vs RotEmb 导致的（期望为零）
