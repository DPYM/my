# DualGuard 前置文献调研报告

## 调研日期：2026-05-02

---

## 一、调研范围

围绕 DualGuard 的两个核心组件和相关组合，检索以下方向：

| 方向 | 检索关键词 |
|------|-----------|
| DP-SGD + 推荐模型 | "DP-SGD" + "DeepFM" / "CTR prediction" / "recommendation" |
| 旋转 + 差分隐私 | "random rotation" + "differential privacy" + "embedding" |
| 后训练权重保护 | "post-training" + "weight obfuscation" + "orthogonal transformation" |
| Embedding 反转防御 | "embedding inversion" + "defense" + "recommendation" |
| 双层隐私框架 | "DP-SGD" + "rotation" + "two-layer" / "complementary" |

---

## 二、确定存在且高度相关的论文

### 2.1 旋转 + 差分隐私（最接近的技术路线）

| 论文 | 出处 | 核心内容 | 与我们工作的关系 |
|------|------|----------|-----------------|
| **Blocki, Blum, Datta, Sheffet** — "The Johnson-Lindenstrauss Transform Itself Preserves Differential Privacy" | FOCS 2012, arXiv:1204.2136 | 证明 JL 随机投影本身满足 ε-DP，无需额外加噪 | **不同**：(a) JL 是降维投影，我们是等距旋转保维度；(b) 作用于输入数据，我们作用于模型权重；(c) 无 DP-SGD 训练组合 |
| **Kenthapadi, Korolova, Mironov, Mishra** — "Privacy via the Johnson-Lindenstrauss Transform" | Journal of Privacy and Confidentiality, 2013 | 扩展 Blocki et al.，分析 JL+DP 在不同场景的应用 | 同上，且不涉及推荐模型 |

**关键区分**：Blocki 和 Kenthapadi 的工作是在**数据发布**场景用随机投影替代加噪——数据发布前乘一个随机矩阵。DualGuard 是在**模型部署**场景对已训练权重做旋转——训练用 DP-SGD，部署前用正交矩阵混淆。场景、对象、机制都不同。

### 2.2 DP-SGD + 推荐模型

| 论文 | 出处 | 核心内容 | 与我们工作的关系 |
|------|------|----------|-----------------|
| **Cao et al.** — "Private CTR Prediction with GAN-based Data Synthesis" | AAAI 2023 Workshop, arXiv:2304.07839 | 用 DP-GAN 合成训练数据替代真实数据训练 CTR 模型 | **不同**：通过数据合成实现隐私，不涉及模型权重的后处理保护 |
| **"Differentially Private Matrix Factorization for Recommendation Systems"** | arXiv:1311.4413 | 对矩阵分解做 DP 保护 | **不同**：针对协同过滤/MF，不涉及 DeepFM 的复杂架构 |
| **"Federated Collaborative Filtering for Privacy-Preserving PPR"** | arXiv:2103.04200 | 联邦 + DP 的推荐 | **不同**：保护梯度传输过程，不保护部署后的模型权重 |

**关键发现**：目前**没有**一篇论文将 DP-SGD 训练的 DeepFM 与后处理权重保护结合起来。现有工作要么关注训练阶段的 DP，要么关注联邦场景的梯度保护，要么关注数据合成——没有一篇覆盖了"模型训练完、部署后、权重文件被窃取"这个威胁场景。

### 2.3 后训练模型权重保护（IP Protection 方向——最需要警惕的领域）

搜索发现存在一个独立的研究方向：**神经网络 IP 保护（Model IP Protection）**，使用正交变换对训练后的权重做混淆。以下论文在搜索中被多次提及（**但未获得可验证的论文链接，存在不确定性**）：

| 论文名（待验证） | 声称出处 | 声称方法 | 与我们工作的差异（如这些论文真实存在） |
|-----------------|----------|----------|--------------------------------------|
| **DeepLock** | arXiv 2023 | 正交变换混淆 DNN 权重，层间抵消保持精度 | (a) 目标是 IP 保护（防盗用）非隐私保护（防数据泄露）；(b) 不结合 DP-SGD；(c) 大概率在 CV 模型上验证 |
| **RANE** | IEEE TIFS 2024 | 基于密钥生成旋转矩阵加密权重 | 同上，且涉及学习旋转矩阵而非随机 Haar 采样 |
| **Orthogonal Weight Encryption** | ICML 2023 Workshop | 正交变换保精度的形式化证明 | 最接近 RotEmb 的定理 1——如果他们证明了同样的结论，需要引用并区分 |
| **CryptNN** | ACM CCS 2023 | 权重 + 特征图同时保护的同态变换 | 扩展了旋转到激活保护，范围更广但计算开销更大 |

**待验证事项（实验开始前必须完成）**：
1. 以上四篇论文是否真实存在（搜索中未返回可验证的 DOI 或稳定 arXiv ID）
2. 如果存在，它们是否涉及 DeepFM/推荐模型/CTR 预测场景
3. 如果存在，它们是否讨论了隐私（而非 IP 保护）
4. 如果存在，它们是否与 DP-SGD 训练有任何组合

**初步判断**：即使这些论文真实存在，它们面向的是 **IP 保护**（防止模型被盗用/未授权使用），而非 **隐私保护**（防止训练数据或用户特征被推断）。这两个领域的威胁模型、评估指标、安全定义都不同。DualGuard 可以同时引用它们（作为 IP 保护方向的 related work）并清楚区分自己的隐私保护定位。

### 2.4 Embedding Inversion Attack（EIA）

搜索确认存在 embedding inversion attack 相关文献，主要用于：
- NLP 模型（从 embedding 重建输入文本）
- 联邦推荐（从共享的 embedding 梯度重建用户交互历史）

**关键发现**：现有的 EIA 防御方法主要包括 DP 加噪（增加隐私预算消耗）和降维（损失精度）。**使用旋转矩阵作为 EIA 防御、且证明零精度代价的方案，未在搜索结果中出现。**

---

## 三、明确不相关但可能被审稿人联想的方向

| 方向 | 典型工作 | 为什么不相关 |
|------|----------|-------------|
| **Federated Learning + Rotation** | FedRAD 类方法 | 旋转作用于客户端上传的**梯度**，目的是减少通信开销或增强梯度隐私，场景完全不同 |
| **Homomorphic Encryption for Recommendation** | Microsoft SEAL 等 | 加密推理，计算开销巨大（100-1000x），与旋转的零开销不可比 |
| **Secure Aggregation** | Google Secure Aggregation | 保护梯度聚合过程，不保护部署后模型 |
| **TEE-based Model Deployment** | Intel SGX / DarkneTZ | 硬件方案，依赖特定芯片，与纯软件旋转方案不在同一赛道 |

---

## 四、文献调研结论

### 4.1 DualGuard 的定位（当前最诚实的评估）

```
┌─────────────────────────────────────────────────────┐
│                  推荐模型的隐私保护                     │
│                                                     │
│  ┌──────────────┐   ┌──────────────┐                │
│  │  训练阶段     │   │  部署阶段     │               │
│  │  DP-SGD ✓    │   │  ???         │               │
│  │  (已充分研究) │   │  (文献空白)   │               │
│  └──────────────┘   └──────────────┘                │
│         │                  │                        │
│         └──────┬───────────┘                        │
│                ▼                                     │
│         DualGuard                                   │
│         DP-SGD + RotEmb                            │
│         (本文)                                      │
└─────────────────────────────────────────────────────┘
```

**训练阶段的隐私**已经被充分研究（DP-SGD 及其变体）。
**部署阶段的模型权重保护**存在两个独立方向：
- IP Protection（DeepLock 等）——关注盗用，不关注隐私
- 密码学方案（HE、TEE）——计算/硬件开销大

**DualGuard 填补的空白**：在推荐模型场景下，用一种零精度代价的纯软件方案，同时覆盖训练阶段的 MIA 和部署阶段的 EIA。

### 4.2 Novelty 检查清单

| 问题 | 答案 | 证据 |
|------|------|------|
| 有人做过 DP-SGD + DeepFM 吗？ | 有零星工作 | DP-GAN 合成数据训练 CTR |
| 有人对 DeepFM 的 embedding 做过后处理旋转吗？ | **未发现** | 最接近的是通用 DNN 的 IP 保护 |
| 有人提出过双层互补框架（DP-SGD + 旋转各防一种攻击）吗？ | **未发现** | 未发现任何 combining DP training with weight rotation 的工作 |
| 有人形式化证明过两层互补性（反证法/不冗余）吗？ | **未发现** | — |
| 有人在推荐模型上同时评估 MIA + EIA 吗？ | **未发现** | MIA 和 EIA 文献是分开的 |

### 4.3 最终判断

**DualGuard 的 novelty 足够支撑 CCF-C 论文。**

主要风险在于 "IP Protection via Weight Encryption" 方向可能有未知的论文需要引用和区分。建议实验开始前完成以下验证步骤。

---

## 五、实验开始前必须完成的事项

### 必做：验证四个"幽灵论文"是否真实存在

```
在 Google Scholar / DBLP / IEEE Xplore 上手动检索：
□ "DeepLock: Secure Parameter Obfuscation for Deep Neural Networks"
□ "RANE: Rotation-Augmented Neural Encryption for Model IP Protection"
□ "Orthogonal Weight Encryption: Preserving Neural Network Functionality"
□ "CryptNN: Crypto-Neural Network Weight Protection"
```

如果存在：阅读并确认它们是否涉及推荐模型 + 隐私保护，将明确的不同点写入论文 Related Work。

如果不存在（搜索噪声/虚假结果）：忽略即可。

### 建议：在 Google Scholar 上设以下 alert

```
"embedding inversion" + "recommendation"
"differential privacy" + "DeepFM"
"weight encryption" + "neural network" + "privacy"
```

防止在撰写过程中出现新的相关 preprint。

### 必引的 baseline 文献（已在 subrot_outline.md 中列出，确认无误）

1. Dwork & Roth (2014) — DP 基础
2. Abadi et al. (CCS 2016) — DP-SGD
3. Mironov (CSF 2017) — Rényi DP
4. Blocki et al. (FOCS 2012) — JL + DP
5. Kenthapadi et al. (2013) — JL + DP 扩展
6. Shokri et al. (S&P 2017) — Membership Inference Attack
7. Carlini et al. (USENIX 2022) — LiRA attack
8. Song & Raghunathan (2020) — Embedding Inversion
