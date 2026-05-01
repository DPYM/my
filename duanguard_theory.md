# 双层隐私保护框架：DP-SGD + RotEmb 完整数学推导

## 方案概览

| 阶段 | 技术 | 防御目标 |
|------|------|----------|
| 训练阶段 | DeepFM + DP-SGD | 保护训练数据，防 Membership Inference |
| 后处理阶段 | Embedding + W⁽⁰⁾ 随机旋转 | 保护模型权重，防 Embedding Inversion |

> 两层各防御不同攻击向量，互补不冗余。

---

# 第一篇：DP-SGD 在 DeepFM 上的理论推导

## 1.1 DeepFM 模型定义

### 符号表

| 符号 | 含义 | 维度 |
|------|------|------|
| $F$ | 特征域数量 | 标量 |
| $k$ | 每个域的 Embedding 维度 | 标量 |
| $m$ | 所有特征的 vocabulary 大小 | 标量 |
| $\mathbf{V} \in \mathbb{R}^{m \times k}$ | Embedding 矩阵 | $m \times k$ |
| $\mathbf{v}_f \in \mathbb{R}^k$ | 第 $f$ 个活跃特征的 Embedding | 向量 |
| $\mathbf{a}^{(0)} \in \mathbb{R}^{Fk}$ | 拼接后的第一层输入 $[\mathbf{v}_1; \dots; \mathbf{v}_F]$ | 向量 |
| $\mathbf{W}^{(0)} \in \mathbb{R}^{h_1 \times Fk}$ | DNN 第一层权重 | 矩阵 |
| $\mathbf{b}^{(0)} \in \mathbb{R}^{h_1}$ | DNN 第一层偏置 | 向量 |
| $w_0 \in \mathbb{R}$ | 全局偏置 | 标量 |
| $\mathbf{w} \in \mathbb{R}^{m}$ | 一阶特征权重 | 向量 |

### 前向传播

$$
\begin{aligned}
y_{\text{lin}} &= w_0 + \sum_{i} w_i \cdot x_i \\
y_{\text{fm}} &= \frac{1}{2} \sum_{f=1}^{F} \sum_{f' \neq f} \langle \mathbf{v}_f, \mathbf{v}_{f'} \rangle \\
\mathbf{a}^{(0)} &= [\mathbf{v}_1; \mathbf{v}_2; \dots; \mathbf{v}_F] \in \mathbb{R}^{Fk} \\
\mathbf{a}^{(1)} &= \sigma\left(\mathbf{W}^{(0)} \mathbf{a}^{(0)} + \mathbf{b}^{(0)}\right) \\
y_{\text{dnn}} &= \text{MLP}\left(\mathbf{a}^{(1)}\right) \\
\hat{y} &= \sigma\left(y_{\text{lin}} + y_{\text{fm}} + y_{\text{dnn}}\right)
\end{aligned}
$$

> 其中 $\sigma(\cdot)$ 为激活函数（通常为 ReLU），最后一层 $\sigma$ 为 sigmoid。

### 全部可训练参数

$$
\Theta = \left\{\mathbf{V}, \mathbf{w}, w_0, \mathbf{W}^{(0)}, \mathbf{b}^{(0)}, \mathbf{W}^{(1)}, \mathbf{b}^{(1)}, \dots, \mathbf{W}^{(L)}, \mathbf{b}^{(L)}\right\}
$$

---

## 1.2 差分隐私基本定义

### 定义 1：$(\varepsilon, \delta)$-差分隐私

一个随机算法 $\mathcal{M}: \mathcal{D} \to \mathcal{R}$ 满足 $(\varepsilon, \delta)$-DP，当且仅当对所有相邻数据集 $D, D'$（相差一条记录）和所有输出子集 $S \subseteq \mathcal{R}$：

$$
\Pr\left[\mathcal{M}(D) \in S\right] \leq e^\varepsilon \cdot \Pr\left[\mathcal{M}(D') \in S\right] + \delta
$$

### 定义 2：Rényi 差分隐私 (RDP)

随机算法 $\mathcal{M}$ 满足 $(\alpha, \varepsilon)$-RDP，当且仅当对所有相邻数据集 $D, D'$：

$$
D_\alpha\left(\mathcal{M}(D) \parallel \mathcal{M}(D')\right) \leq \varepsilon
$$

其中 $D_\alpha(P \parallel Q) = \frac{1}{\alpha-1} \log \mathbb{E}_{x \sim Q}\left[\left(\frac{P(x)}{Q(x)}\right)^\alpha\right]$ 为 $\alpha$ 阶 Rényi 散度。

> **RDP 的优势**：提供紧致的组合界，便于追踪训练过程中 $\varepsilon$ 的累计。

---

## 1.3 DP-SGD 算法

对 mini-batch $B_t$（采样概率 $q = |B_t|/N$），对每个样本 $i \in B_t$：

### 步骤 1：计算 per-sample gradient

$$
\mathbf{g}_t^{(i)} = \nabla_{\Theta} \mathcal{L}\left(\Theta_t; (\mathbf{x}^{(i)}, y^{(i)})\right)
$$

### 步骤 2：梯度裁剪（上界约束灵敏度，$\ell_2$ 范数）

$$
\bar{\mathbf{g}}_t^{(i)} = \mathbf{g}_t^{(i)} \cdot \min\left(1, \frac{C}{\left\|\mathbf{g}_t^{(i)}\right\|_2}\right)
$$

### 步骤 3：聚合加噪

$$
\tilde{\mathbf{g}}_t = \frac{1}{|B_t|} \left(\sum_{i \in B_t} \bar{\mathbf{g}}_t^{(i)} + \mathcal{N}\left(0, \sigma^2 C^2 \mathbf{I}\right)\right)
$$

> 其中 $\sigma$ 为 noise multiplier，$C$ 为裁剪阈值。

### 步骤 4：参数更新

$$
\Theta_{t+1} = \Theta_t - \eta_t \cdot \tilde{\mathbf{g}}_t
$$

### 关键参数关系

给定采样概率 $q = B/N$、迭代步数 $T$、noise multiplier $\sigma$：

**RDP 隐私会计**（每个 step）：

$$
\varepsilon_{\text{step}}(\alpha) = \frac{1}{\alpha-1} \log\left(
(1-q)^{\alpha-1} \cdot (\alpha q - q + 1) +
\sum_{\ell=2}^{\alpha} \binom{\alpha}{\ell} (1-q)^{\alpha-\ell} q^\ell \cdot
\exp\left(\frac{(\ell-1)\ell}{2\sigma^2}\right)
\right)
$$

**总隐私预算**（RDP 组合）：

$$
\varepsilon_{\text{total}}(\alpha) = T \cdot \varepsilon_{\text{step}}(\alpha)
$$

**转换为 $(\varepsilon, \delta)$-DP**：

$$
\varepsilon_{\text{DP}} = \min_{\alpha > 1} \left( \varepsilon_{\text{total}}(\alpha) + \frac{\log(1/\delta)}{\alpha - 1} \right)
$$

---

## 1.4 DeepFM 的 per-sample gradient 结构分析

### Embedding 层的梯度

对每个活跃特征 $f$，其 Embedding $\mathbf{v}_f$ 的梯度来自三个通路：

$$
\nabla_{\mathbf{v}_f} \mathcal{L} =
\underbrace{\frac{\partial \mathcal{L}}{\partial y_{\text{lin}}} \cdot \frac{\partial y_{\text{lin}}}{\partial \mathbf{v}_f}}_{= 0 \text{（线性项不含 }\mathbf{v}_f\text{）}} +
\underbrace{\frac{\partial \mathcal{L}}{\partial y_{\text{fm}}} \cdot \frac{\partial y_{\text{fm}}}{\partial \mathbf{v}_f}}_{\text{FM 通路}} +
\underbrace{\frac{\partial \mathcal{L}}{\partial y_{\text{dnn}}} \cdot \frac{\partial y_{\text{dnn}}}{\partial \mathbf{a}^{(0)}} \cdot \frac{\partial \mathbf{a}^{(0)}}{\partial \mathbf{v}_f}}_{\text{DNN 通路}}
$$

### FM 通路梯度

$$
\frac{\partial y_{\text{fm}}}{\partial \mathbf{v}_f} = \sum_{f' \neq f} \mathbf{v}_{f'}
$$

> 即向量求和——对样本中所有其他特征的 Embedding 求和。

### DNN 通路梯度

令 $\boldsymbol{\delta}^{(0)} = \frac{\partial \mathcal{L}}{\partial y_{\text{dnn}}} \cdot \frac{\partial y_{\text{dnn}}}{\partial \mathbf{a}^{(0)}} \in \mathbb{R}^{Fk}$，则：

$$
\frac{\partial y_{\text{dnn}}}{\partial \mathbf{v}_f} = \boldsymbol{\delta}^{(0)}_{(f-1)k+1 : fk}
$$

> 即 $\boldsymbol{\delta}^{(0)}$ 中第 $f$ 块（大小为 $k$）。

**关键观察**：Embedding 梯度是稀疏的——只有样本中出现的特征（$F$ 个）有梯度，其余 $m-F$ 个特征的梯度为零。这与 DNN 层的稠密梯度形成对比。

---

# 第二篇：RotEmb 后处理旋转的理论推导

## 2.1 旋转变换定义

设 $R \in O(k)$ 为从 Haar 测度采样的随机正交矩阵。后训练变换如下：

| 参数 | 变换前 | 变换后 |
|------|--------|--------|
| $\mathbf{v}_f$ | $\mathbf{v}_f$ | $\tilde{\mathbf{v}}_f = R \cdot \mathbf{v}_f$ |
| $\mathbf{W}^{(0)}$ | $\mathbf{W}^{(0)}$ | $\tilde{\mathbf{W}}^{(0)} = \mathbf{W}^{(0)} \cdot (\mathbf{I}_F \otimes R^\top)$ |
| 其余所有参数 | — | 不变 |

其中 $\mathbf{I}_F \otimes R$ 是 $F \times F$ 块对角 Kronecker 积（每个块为 $k \times k$ 的矩阵 $R$）：

$$
\mathbf{I}_F \otimes R =
\begin{bmatrix}
R & \mathbf{0} & \cdots & \mathbf{0} \\
\mathbf{0} & R & \cdots & \mathbf{0} \\
\vdots & \vdots & \ddots & \vdots \\
\mathbf{0} & \mathbf{0} & \cdots & R
\end{bmatrix}_{Fk \times Fk}
$$

---

## 2.2 定理 1：精确精度保持

> **定理 1**：对任意输入 $\mathbf{x}$，设 $\hat{y}(\mathbf{x})$ 为原始 DeepFM 的预测，$\tilde{y}(\mathbf{x})$ 为 RotEmb 变换后模型的预测。则 $\tilde{y}(\mathbf{x}) = \hat{y}(\mathbf{x})$。

### 证明（逐分量验证）

**① 线性部分**

线性参数 $\{w_0, \mathbf{w}\}$ 不变，输入不变 → $\tilde{y}_{\text{lin}} = y_{\text{lin}}$。$\square$

**② FM 配对部分**

利用正交矩阵保内积性质 $R^\top R = \mathbf{I}_k$：

$$
\begin{aligned}
\langle \tilde{\mathbf{v}}_f, \tilde{\mathbf{v}}_{f'} \rangle
&= \langle R\mathbf{v}_f, R\mathbf{v}_{f'} \rangle \\
&= (R\mathbf{v}_f)^\top (R\mathbf{v}_{f'}) \\
&= \mathbf{v}_f^\top R^\top R \mathbf{v}_{f'} \\
&= \mathbf{v}_f^\top \mathbf{I}_k \mathbf{v}_{f'} \\
&= \langle \mathbf{v}_f, \mathbf{v}_{f'} \rangle
\end{aligned}
$$

因此 $\tilde{y}_{\text{fm}} = \frac{1}{2}\sum_{f \neq f'} \langle \tilde{\mathbf{v}}_f, \tilde{\mathbf{v}}_{f'} \rangle = y_{\text{fm}}$。$\square$

**③ DNN 通路**

*Step 1 — 变换后的拼接输入*：

$$
\begin{aligned}
\tilde{\mathbf{a}}^{(0)}
&= \left[\tilde{\mathbf{v}}_1; \tilde{\mathbf{v}}_2; \dots; \tilde{\mathbf{v}}_F\right] \\
&= \left[R\mathbf{v}_1; R\mathbf{v}_2; \dots; R\mathbf{v}_F\right] \\
&= (\mathbf{I}_F \otimes R) \cdot \left[\mathbf{v}_1; \dots; \mathbf{v}_F\right] \\
&= (\mathbf{I}_F \otimes R) \cdot \mathbf{a}^{(0)}
\end{aligned}
$$

*Step 2 — 第一层隐藏层（唯一需要补偿的层）*：

$$
\begin{aligned}
\tilde{\mathbf{W}}^{(0)} \cdot \tilde{\mathbf{a}}^{(0)}
&= \left[\mathbf{W}^{(0)} \cdot (\mathbf{I}_F \otimes R^\top)\right] \cdot \left[(\mathbf{I}_F \otimes R) \cdot \mathbf{a}^{(0)}\right] \\
&= \mathbf{W}^{(0)} \cdot (\mathbf{I}_F \otimes R^\top)(\mathbf{I}_F \otimes R) \cdot \mathbf{a}^{(0)} \\
&= \mathbf{W}^{(0)} \cdot \left[\mathbf{I}_F \otimes (R^\top R)\right] \cdot \mathbf{a}^{(0)} \quad \text{（Kronecker 积性质）} \\
&= \mathbf{W}^{(0)} \cdot (\mathbf{I}_F \otimes \mathbf{I}_k) \cdot \mathbf{a}^{(0)} \\
&= \mathbf{W}^{(0)} \cdot \mathbf{I}_{Fk} \cdot \mathbf{a}^{(0)} \\
&= \mathbf{W}^{(0)} \cdot \mathbf{a}^{(0)}
\end{aligned}
$$

加上不变的偏置 $\mathbf{b}^{(0)}$，pre-activation 相同，因此 $\tilde{\mathbf{a}}^{(1)} = \mathbf{a}^{(1)}$。$\square$

*Step 3 — 归纳*：$\tilde{\mathbf{a}}^{(1)} = \mathbf{a}^{(1)}$，更深层参数不变 → $\tilde{\mathbf{a}}^{(\ell)} = \mathbf{a}^{(\ell)}$ 对所有 $\ell \geq 1$ 成立。因此 $\tilde{y}_{\text{dnn}} = y_{\text{dnn}}$。$\square$

**④ 最终输出**

$$
\tilde{y}(\mathbf{x}) = \sigma(\tilde{y}_{\text{lin}} + \tilde{y}_{\text{fm}} + \tilde{y}_{\text{dnn}}) = \sigma(y_{\text{lin}} + y_{\text{fm}} + y_{\text{dnn}}) = \hat{y}(\mathbf{x})
$$

$\blacksquare$

---

## 2.3 定理 2：旋转后的 Embedding 不可逆性

> **定理 2（等价类引理）**：对任意观测到的旋转后 Embedding $\tilde{\mathbf{v}} = R\mathbf{v}$，存在与 $\tilde{\mathbf{v}}$ 一致的不可区分的原始 Embedding 构成整个球面。具体地，对任意 $Q \in O(k)$，$\mathbf{v}' = Q^\top \mathbf{v}$ 和 $R' = RQ$ 满足 $R' \mathbf{v}' = \tilde{\mathbf{v}}$。

### 证明

$$
R' \mathbf{v}' = (RQ) \cdot (Q^\top \mathbf{v}) = R \cdot (QQ^\top) \cdot \mathbf{v} = R \cdot \mathbf{I}_k \cdot \mathbf{v} = R\mathbf{v} = \tilde{\mathbf{v}}
$$

由于对手不知道 $R$，所有满足 $\left\|\mathbf{v}'\right\| = \left\|\tilde{\mathbf{v}}\right\|$ 的 $\mathbf{v}'$ 都是可能的原始 Embedding。满足此条件的向量构成 $S^{k-1}(\left\|\tilde{\mathbf{v}}\right\|)$，即 $k$ 维空间中以原点为球心、$\left\|\tilde{\mathbf{v}}\right\|$ 为半径的球面。

> 这意味着对手无法确定 Embedding 在 $k$ 维空间中的**方向**——方向信息被旋转完全消灭。$\blacksquare$

### 公式化地表述信息泄露

从 $\tilde{\mathbf{v}} = R\mathbf{v}$，对手可以计算：

$$
\left\|\tilde{\mathbf{v}}\right\| = \left\|R\mathbf{v}\right\| = \left\|\mathbf{v}\right\| \quad \text{（范数保持—正交矩阵性质）}
$$

$$
\langle \tilde{\mathbf{v}}_i, \tilde{\mathbf{v}}_j \rangle = \langle R\mathbf{v}_i, R\mathbf{v}_j \rangle = \langle \mathbf{v}_i, \mathbf{v}_j \rangle \quad \text{（内积保持）}
$$

> 这两个量是旋转下**唯一**的不变量。任何依赖绝对坐标值的查询被完全随机化。

---

## 2.4 定理 3：重建误差下界

> **定理 3**：对任意估计器 $g: \mathbb{R}^k \to \mathbb{R}^k$，在 $R \sim \text{Haar}(O(k))$ 下：
>
> $$
> \min_g \mathbb{E}_R\left[\left\|\mathbf{v} - g(R\mathbf{v})\right\|^2\right] = \left\|\mathbf{v}\right\|^2
> $$
>
> 由平凡估计器 $g(\mathbf{y}) \equiv \mathbf{0}$ 达到。即对手无法做得比全零猜测更好。

### 证明

不失一般性，设 $\mathbf{v} = \rho \cdot \mathbf{e}_1$，其中 $\rho = \left\|\mathbf{v}\right\|$，$\mathbf{e}_1$ 为第一个标准基向量。则观测 $\mathbf{y} = R\mathbf{v} = \rho \cdot \mathbf{r}_1$，其中 $\mathbf{r}_1$ 为 $R$ 的第一列。

由 Haar 测度性质，$\mathbf{r}_1$ 在单位球面 $S^{k-1}(1)$ 上均匀分布。

**① 线性估计器**

考虑线性估计器 $g(\mathbf{y}) = \alpha \cdot \mathbf{y} = \alpha\rho \cdot \mathbf{r}_1$：

$$
\begin{aligned}
\mathbb{E}\left[\left\|\mathbf{v} - g(\mathbf{y})\right\|^2\right]
&= \mathbb{E}\left[\left\|\rho\mathbf{e}_1 - \alpha\rho\mathbf{r}_1\right\|^2\right] \\
&= \rho^2 \cdot \mathbb{E}\left[\left\|\mathbf{e}_1 - \alpha\mathbf{r}_1\right\|^2\right] \\
&= \rho^2 \cdot \left(1 + \alpha^2 - 2\alpha \cdot \mathbb{E}\left[\mathbf{e}_1^\top \mathbf{r}_1\right]\right)
\end{aligned}
$$

对球面上均匀分布的 $\mathbf{r}_1$：

$$
\mathbb{E}\left[\mathbf{e}_1^\top \mathbf{r}_1\right] = \mathbb{E}\left[r_{11}\right] = 0 \quad \text{（对称性）}
$$

因此 $\mathbb{E}\left[\left\|\mathbf{v} - g(\mathbf{y})\right\|^2\right] = \rho^2(1 + \alpha^2) \geq \rho^2$，最小值 $\rho^2$ 在 $\alpha = 0$（即 $g \equiv 0$）处取得。

**② 一般估计器**

将 $g(\mathbf{y})$ 分解为沿 $\mathbf{y}$ 方向的分量和垂直分量：$g(\mathbf{y}) = \alpha\mathbf{y} + \mathbf{h}(\mathbf{y})$ 且 $\langle \mathbf{h}(\mathbf{y}), \mathbf{y} \rangle = 0$。

由 $\mathbf{r}_1$ 的球面对称性，$\mathbb{E}\left[\mathbf{v}^\top g(\mathbf{y})\right] = 0$，得：

$$
\mathbb{E}\left[\left\|\mathbf{v} - g(\mathbf{y})\right\|^2\right] = \rho^2 + \mathbb{E}\left[\left\|g(\mathbf{y})\right\|^2\right] \geq \rho^2
$$

最小值 $\rho^2$ 仅在 $g \equiv 0$ 时取得。

$\blacksquare$

> **物理意义**：$\rho^2 = \left\|\mathbf{v}\right\|^2$ 是平凡下界——这正是你猜全零向量时的 MSE。**对手无法比随机猜测做得更好**。Embedding 的方向信息被旋转完全消灭。

---

## 2.5 变换的安全性假设

旋转矩阵 $R$ 必须保密存储，**不随模型一同部署**。若 $R$ 泄露，保护完全失效（对手可计算 $\mathbf{v} = R^\top \tilde{\mathbf{v}}$）。

这类似于对称加密中密钥管理问题：

| 角色 | 类比 |
|------|------|
| 训练方持有 $R$ | 加密密钥 |
| 部署方只持有变换后的模型权重 | 密文 |
| 白盒访问模型的对手 | 只看到密文 |

---

# 第三篇：两层互补性分析

## 3.1 两层防御的威胁模型

| 防御层 | 部署位置 | 威胁模型 | 对手能力 | 防御目标 |
|--------|----------|----------|----------|----------|
| **DP-SGD** | 训练阶段 | 差分攻击 | 通过多次查询观察模型输出的统计差异 | 阻止推断某条训练数据是否在训练集中（Membership Inference） |
| **RotEmb** | 后处理阶段 | 白盒权重窃取 | 获取完整模型权重文件 | 阻止从权重中重建 Embedding 的语义方向（Embedding Inversion） |

---

## 3.2 形式化攻击定义

### Attack 1：Membership Inference Attack (MIA)

**攻击目标**：给定目标样本 $(\mathbf{x}^*, y^*)$，判断该样本是否属于训练集 $D_{\text{train}}$。

**形式化定义**：攻击者 $\mathcal{A}$ 可查询模型 $\mathcal{M}(D_{\text{train}})$，输出二元判断：

$$
\mathcal{A}\left(\mathbf{x}^*, y^*, \mathcal{M}(D_{\text{train}})\right) \in \{0, 1\}
$$

**DP-SGD 防御原理**：$(\varepsilon, \delta)$-DP 保证对任意攻击者：

$$
\Pr\left[\mathcal{A} = 1 \mid (\mathbf{x}^*, y^*) \in D\right] \leq e^\varepsilon \cdot \Pr\left[\mathcal{A} = 1 \mid (\mathbf{x}^*, y^*) \notin D\right] + \delta
$$

当 $\varepsilon$ 很小时（如 $\varepsilon \leq 8$），TPR 和 FPR 非常接近，AUC 趋近 0.5（随机猜测）。

> **关键**：RotEmb **不提供**对抗 MIA 的保护——对手通过观察模型**输出**（概率值）做推断，而 Theorem 1 保证了输出不因旋转改变。

### Attack 2：Embedding Inversion Attack (EIA)

**攻击目标**：从模型权重中提取 Embedding $\{\mathbf{v}_f\}$，将 Embedding 维度映射到可解释的语义特征。

**攻击者的形式化能力**：给定变换后的模型权重 $\tilde{\Theta}$，攻击者已知部分特征的语义标签（如已知某些电影是"恐怖片"），希望推断未知特征的语义标签。

**典型攻击流程**（参考 Song & Raghunathan, 2020）：

1. 提取所有 $\tilde{\mathbf{v}}_f$
2. 聚类（利用内积保持性质，相似 Embedding 仍然相似）
3. 用已知标签的少量特征锚定聚类 → 传播标签到同簇未知特征

**RotEmb 防御原理**：

虽然聚类可行（内积保持），但定理 3 证明了：**无法从 $\tilde{\mathbf{v}}_f$ 中重建方向信息**。

形式化地，给定 $\tilde{\mathbf{v}} = R\mathbf{v}$，攻击者的最优估计 $\hat{\mathbf{v}}$ 满足：

$$
\mathbb{E}\left[\left\|\hat{\mathbf{v}} - \mathbf{v}\right\|^2\right] \geq \left\|\mathbf{v}\right\|^2
$$

而在标准的（无隐私保护）场景下，攻击者可直接读取 $\mathbf{v}$，MSE = 0。旋转将 MSE 从 0 提升到 $\left\|\mathbf{v}\right\|^2$（等于猜全零向量的误差）。

**维度解释的不可能**：假设 Embedding 的某一维度编码"恐怖程度"（如 $\mathbf{v}$ 的第 1 维度为正 = 恐怖片）。旋转后，这一维度信号被散布到全部 $k$ 个维度：

$$
\tilde{\mathbf{v}} = R\mathbf{v} = \sum_{j=1}^{k} \mathbf{r}_j \cdot v_j
$$

第 1 维 $\tilde{v}_1 = \sum_{j=1}^{k} r_{1j} \cdot v_j$ 是一个随机线性组合，原始"恐怖维度"的信息被均匀混合，单个维度不再可解释。

> **关键**：DP-SGD **不提供**对抗 EIA 的保护——DP-SGD 加噪在梯度上，训练后的模型权重仍然以浮点数形式暴露 Embedding 值。

---

## 3.3 互补性定理

> **定理 4（互补性）**：设攻击者 $\mathcal{A}$ 同时具有白盒访问模型权重 $\tilde{\Theta}$ 和黑盒查询模型输出 $\hat{y}(\mathbf{x})$ 的能力。则：
> - DP-SGD 约束 $\mathcal{A}$ 通过输出查询推断训练集成员关系的能力
> - RotEmb 约束 $\mathcal{A}$ 通过权重分析重建 Embedding 语义的能力
> - 两层之间不存在冗余覆盖

### 证明（反证法）

**假设 DP-SGD 可以防御 EIA**：

DP-SGD 的隐私保证对模型最终参数 $\Theta$ 成立，意味着 $\Theta$ 和 $\Theta'$（在相邻训练集上训练）的分布接近。但 EIA 不关心两个相邻训练集的区别——EIA 只需要一组固定的 Embedding 做重建。因此 DP-SGD 放松后（$\varepsilon \to \infty$），EIA 依然存在。矛盾。

**假设 RotEmb 可以防御 MIA**：

RotEmb 保持模型输出完全不变（定理 1），因此 MIA 攻击者观察到的查询概率分布在旋转前后完全一致。若 RotEmb 能防御 MIA，则意味着原始 DeepFM 也能防御 MIA，显然不成立。

> 因此两层互补、不冗余。$\blacksquare$

---

# 第四篇：联合隐私保证分析

## 4.1 总隐私损失分解

双层框架下，总隐私损失可分解为两个独立分量：

$$
\mathcal{L}_{\text{privacy}}^{\text{total}} = \mathcal{L}_{\text{DP-SGD}} + \mathcal{L}_{\text{RotEmb}}
$$

| 分量 | 性质 | 量化方式 |
|------|------|----------|
| $\mathcal{L}_{\text{DP-SGD}}$ | 可通过 Rényi DP 会计精确追踪 | 可量化的 $(\varepsilon, \delta)$ 值 |
| $\mathcal{L}_{\text{RotEmb}}$ | 非差分隐私形式的安保度量 | 由 $R$ 的密钥强度和 Embedding 重建误差下界联合给出 |

---

## 4.2 DP-SGD 侧的量化隐私保证

给定训练超参数 $(q, T, \sigma)$ 和目标 $\delta$：

$$
\varepsilon_{\text{MIA}} = \min_{\alpha > 1} \left( T \cdot \varepsilon_{\text{step}}(\alpha; q, \sigma) + \frac{\log(1/\delta)}{\alpha - 1} \right)
$$

**典型 CCF-C 级别实验设置**（在 Criteo 上）：

- $\varepsilon \in \{1, 2, 4, 8\}$，$\delta = 10^{-5}$
- 对照：$\varepsilon = \infty$（无 DP-SGD）

---

## 4.3 RotEmb 侧的安保度量

### 度量 1：方向信息损失

定义方向估计的相对误差下界：

$$
\text{DirectionLoss} = \frac{\min_g \mathbb{E}_R\left[\left\|\mathbf{v} - g(R\mathbf{v})\right\|^2\right]}{\left\|\mathbf{v}\right\|^2} = 1 \quad \text{（由定理 3）}
$$

> 方向信息损失 = 100%，对方看旋转后的 Embedding 与看纯噪声无异。

### 度量 2：维度可解释性损失

原始 Embedding $\mathbf{v}$ 的某一维度 $j$ 编码语义 $s$，旋转后这一编码被散布到全部维度：

$$
\tilde{\mathbf{v}} = \sum_{\ell=1}^{k} \mathbf{r}_\ell \cdot v_\ell
$$

单个维度 $\tilde{v}_p$ 中来自原始语义 $s$（编码在 $v_j$）的信号比例为：

$$
\text{semantic}(s \to \tilde{v}_p) = \frac{|r_{pj} \cdot v_j|}{\sum_{\ell=1}^{k} |r_{p\ell} \cdot v_\ell|}
$$

对 $r_{pj} \sim \mathcal{N}(0, 1/k)$（大 $k$ 近似），该比例的期望值为 $\approx 1/k$。当 $k = 64$（典型 DeepFM 配置），每个旋转后维度仅含 $\sim$1.5% 的源语义信号，任何单个维度的可解释性被稀释 64 倍。

### 度量 3：配对相似度的剩余风险

旋转保持了内积 $\langle\tilde{\mathbf{v}}_i, \tilde{\mathbf{v}}_j\rangle = \langle\mathbf{v}_i, \mathbf{v}_j\rangle$，因此聚类和 KNN 攻击依然有效。这是已知的限制，论文中应诚实承认。

**缓解措施**（论文 Discussion 部分）：

1. 旋转后叠加少量高斯噪声（用极小的 $\varepsilon$）→ 形式化 DP + 方向保护
2. 不同 Embedding 组使用不同旋转矩阵 → 打破跨组内积一致性（代价是 FM 层失效）

---

# 第五篇：端到端算法

## 算法：DualGuard — 双层隐私保护的 DeepFM 训练与部署

### 阶段一：训练阶段（DP-SGD 保护）

```
输入: 训练集 D_train = {(x_i, y_i)}_{i=1}^{N}
      隐私预算 (ε_target, δ), 裁剪阈值 C
      batch size B, 学习率 η, 训练轮数 E

输出: Θ = {V, w, w_0, W^(0), b^(0), ..., W^(L), b^(L)}

1. 计算 noise_multiplier σ = f(ε_target, δ, B/N, E)
2. for epoch = 1 to E:
     for each batch B_t ⊂ D_train:
       a. 对每个样本 (x_i, y_i) ∈ B_t:
          - 计算 per-sample gradient g_i
          - 裁剪: ḡ_i = g_i · min(1, C/‖g_i‖₂)
       b. 聚合加噪:
          g̃ = (1/|B_t|)(Σ ḡ_i + N(0, σ²C²I))
       c. 更新: Θ ← Θ - η · g̃
       d. 更新 RDP accountant
     if ε_spent ≥ ε_target: break
3. return Θ
```

### 阶段二：RotEmb 后处理

```
输入: 训练好的 Θ, Embedding 维度 k

输出: Θ̃（旋转后模型）+ R（密钥）

1. 采样 R ~ Haar(O(k))
2. 对每个特征 f:
     ṽ_f ← R · v_f
3. 补偿 W^(0):
     W̃^(0) ← W^(0) · (I_F ⊗ R^⊤)
4. 其余参数原样拷贝
5. 安全存储 R（不部署）
6. return Θ̃, R
```

### 阶段三：部署与推理（无隐私泄露）

```
- 部署 Θ̃（不包含 R）
- 推理时: 使用变换后的 Embedding ṽ_f 和权重 W̃^(0)
- 预测结果与原始模型完全一致（Theorem 1）
```

---

## 实验设计建议

### 核心实验组

| 实验 | 对比维度 | 变量 |
|------|----------|------|
| Exp 1 | MIA 防御 | (No DP, ε=8, ε=4, ε=2, ε=1) × (With/Without RotEmb) |
| Exp 2 | EIA 防御 | (With/Without RotEmb) → Embedding reconstruction MSE |
| Exp 3 | Utility 损失 | AUC vs ε 曲线，对比 Pure DP-SGD vs DualGuard |
| Exp 4 | 参数敏感性 | k ∈ {8, 16, 32, 64, 128} 对 EIA 防御强度的影响 |

### 关键指标

| 类别 | 指标 | 说明 |
|------|------|------|
| 效用 | AUC（Criteo/Avazu）、LogLoss | 模型预测性能 |
| 隐私—MIA 侧 | MIA AUC | 越低越好，≤ 0.55 视为有效防御 |
| 隐私—EIA 侧 | Reconstruction MSE / ‖v‖² | 越高越好，→ 1 表示随机猜测 |
| 开销 | 训练时间增加 | RotEmb 后处理是 O(F·k³ + h₁·Fk²)，一次性 |

---

# 第六篇：关键技术决策深度分析

## 6.1 硬裁剪 vs 软裁剪：DualGuard 中的选择

### 硬裁剪（Hard Clipping）

对 per-sample gradient $\mathbf{g}$，硬裁剪定义为：

$$
\bar{\mathbf{g}}_{\text{hard}} = \mathbf{g} \cdot \min\left(1, \frac{C}{\left\|\mathbf{g}\right\|_2}\right)
$$

**性质**：梯度向量 $\mathbf{g}$ 被限制在半径为 $C$ 的 $\ell_2$ 球内。任何范数超过 $C$ 的梯度仅保留方向，幅度信息被丢弃。

### 软裁剪（Soft Clipping）

软裁剪的一种常见形式为：

$$
\bar{\mathbf{g}}_{\text{soft}} = C \cdot \frac{\mathbf{g}}{\left\|\mathbf{g}\right\|_2 + \gamma}
$$

其中 $\gamma > 0$ 为软化参数，控制过渡区的平滑程度。

**性质**：
- 当 $\left\|\mathbf{g}\right\|_2 \gg \gamma$：$\bar{\mathbf{g}}_{\text{soft}} \approx C \cdot \mathbf{g} / \left\|\mathbf{g}\right\|_2$，与硬裁剪一致
- 当 $\left\|\mathbf{g}\right\|_2 \ll \gamma$：$\bar{\mathbf{g}}_{\text{soft}} \approx (C/\gamma) \cdot \mathbf{g}$，线性缩放，保留相对大小关系
- 当 $\left\|\mathbf{g}\right\|_2 \approx \gamma$：平滑过渡

**优势**：保留了梯度范数之间的**相对排序关系**。两个梯度分别有 $\left\|\mathbf{g}_A\right\| = 0.5\gamma$ 和 $\left\|\mathbf{g}_B\right\| = 0.1\gamma$，软裁剪后仍然保持 5:1 的比例关系。硬裁剪对两者都不截断，信息保留相同；但当 $\left\|\mathbf{g}_A\right\| = 2C$ 而 $\left\|\mathbf{g}_B\right\| = 0.5C$ 时，硬裁剪后 $\left\|\bar{\mathbf{g}}_A\right\| = \left\|\bar{\mathbf{g}}_B\right\| = C$——两个完全不相关量级的梯度被压缩为同等大。

### DeepFM 视角下的关键差异

DeepFM 的梯度结构存在天然的异质性：

| 梯度来源 | 涉及参数量 | 典型梯度范数 | 稀疏性 |
|----------|-----------|-------------|--------|
| **Embedding**（活跃特征） | $F \times k$ | 偏大（FM + DNN 双通路贡献） | 极稀疏（仅 $F$ 个特征） |
| **Embedding**（非活跃特征） | $(m-F) \times k$ | 严格为零 | 零 |
| **DNN 第一层** | $h_1 \times Fk$ | 中等 | 稠密 |
| **DNN 深层** | 各层 | 偏小（反向传播递减） | 稠密 |
| **FM 一阶权重** | $m$ | 偏小（仅线性项贡献） | 稀疏 |

硬裁剪一刀切带来的问题：
1. **Embedding 梯度经常超过 $C$**：FM 通路的梯度 $\frac{\partial y_{\text{fm}}}{\partial \mathbf{v}_f} = \sum_{f' \neq f} \mathbf{v}_{f'}$ 是 $F-1$ 个向量的和，$F$ 大时范数自然大。这种情况下硬裁剪只保留方向、丢弃了"这个样本对 embedding 影响有多强"的信息。
2. **尾部特征的梯度更可能被裁剪**：低频特征在训练中出现次数少，梯度幅度天然偏大（模型对此特征拟合不足）。硬裁剪在它们最需要学习的时候截断了信号。
3. **DNN 深层梯度偏小**：通常不会被硬裁剪截断，裁剪对它们形同虚设。这意味着 $\ell_2$ 裁剪的"保护"在模型的不同层上不均匀。

### 推荐方案：硬裁剪 + 分层 $C$（Layer-wise Clipping）

**不建议用软裁剪**——尽管理论上更优，但 DP 文献中软裁剪的 privacy accounting 分析远不如硬裁剪成熟。对一个 CCF-C 论文，走一条没有充分理论支持的路风险太大。

**建议用硬裁剪，但不同层使用不同的裁剪阈值 $C$**：

$$
C_{\ell} = C_{\text{base}} \cdot \sqrt{d_{\ell}}
$$

其中 $d_{\ell}$ 为第 $\ell$ 层的参数维度。理由：对 $d$ 维向量，$\ell_2$ 范数的期望量级为 $\sqrt{d}$。按维度缩放能保证各层的有效噪声/信号比大致均匀。

具体分配：

| 参数组 | 维度 | 推荐 $C$ |
|--------|------|----------|
| Embedding（per feature） | $k = 16$ | $C_{\text{base}} \times 4$ |
| DNN 第一层 | $h_1 \times Fk \approx 400 \times 624$ | $C_{\text{base}} \times \sqrt{400 \times 624}$ |
| DNN 深层 | $\sim 400 \times 400$ | $C_{\text{base}} \times 400$ |

**这个分层裁剪策略本身就是一个小的 novelty**——可以作为论文的实验分析点。在实验中对比 uniform $C$ vs layer-wise $C$ 在相同 $\varepsilon$ 下的 AUC。

### 论文中如何呈现

在实验部分先对比 uniform clipping 和 layer-wise clipping 的 baseline 差异，然后统一用 layer-wise 跑后续所有实验。Discussion 中指出硬裁剪对稀疏 embedding 梯度的非均匀 bias，引向未来工作方向（per-feature-group adaptive clipping + fairness）。

---

## 6.2 FM 和 DNN 的统一保护：为什么一次旋转就足够

### 核心机制：共享 Embedding 层的群作用

DeepFM 的 FM 通路和 DNN 通路共享同一个 Embedding 矩阵 $\mathbf{V} \in \mathbb{R}^{m \times k}$。对 $\mathbf{V}$ 做一次正交变换 $R$，两个通路同时被影响——但影响方式不同，而这个差异恰好体现了旋转方法的优雅性。

### FM 通路：群作用下的不变量

FM 通路中，所有计算仅依赖 embedding 之间的**内积**：

$$
y_{\text{fm}} = \frac{1}{2} \sum_{f=1}^{F} \sum_{f' \neq f} \langle \mathbf{v}_f, \mathbf{v}_{f'} \rangle
$$

正交群 $O(k)$ 在 $\mathbb{R}^k$ 上的标准作用是**等距的**：

$$
\langle R\mathbf{v}_f, R\mathbf{v}_{f'} \rangle = \langle \mathbf{v}_f, \mathbf{v}_{f'} \rangle, \quad \forall R \in O(k)
$$

因此 FM 通路是 $O(k)$ 群作用下的**不变量**。不需要任何补偿——FM 生来就对旋转免疫。

### DNN 通路：群作用下的等变补偿

DNN 通路的第一个计算步骤是线性变换：

$$
\mathbf{W}^{(0)} \mathbf{a}^{(0)}, \quad \mathbf{a}^{(0)} = [\mathbf{v}_1; \dots; \mathbf{v}_F]
$$

这个操作不是群作用下的不变量——它依赖 embedding 的**绝对坐标值**（$\mathbf{W}^{(0)}$ 的每一列是某个 embedding 维度的权重）。

旋转改变坐标系后，$\mathbf{a}^{(0)} \to (\mathbf{I}_F \otimes R)\mathbf{a}^{(0)}$。为了让输出不变，$\mathbf{W}^{(0)}$ 必须反向旋转来吸收坐标系的变化：

$$
\mathbf{W}^{(0)} \to \mathbf{W}^{(0)} (\mathbf{I}_F \otimes R^\top)
$$

这使得 $\mathbf{W}^{(0)}$ 是 $O(k)$ 作用下的**逆变的（contravariant）**——它随坐标系反向旋转，使得两者的组合（线性变换的输出）保持不变。

### 统一性公式

用一个公式总结两层的关系：

$$
\begin{aligned}
\text{FM}: &\quad \langle R\mathbf{v}_f, R\mathbf{v}_{f'} \rangle = \langle \mathbf{v}_f, \mathbf{v}_{f'} \rangle \quad \text{（不变量—自动保持）} \\
\text{DNN}: &\quad \left[\mathbf{W}^{(0)}(\mathbf{I}_F \otimes R^\top)\right] \cdot \left[(\mathbf{I}_F \otimes R)\mathbf{a}^{(0)}\right] = \mathbf{W}^{(0)} \mathbf{a}^{(0)} \quad \text{（等变补偿—手动保持）}
\end{aligned}
$$

### 为什么这是有价值的

对于**纯 DNN 推荐模型**（如 DCN、xDeepFM 的 DNN 部分）：旋转后必须补偿第一层权重。能工作，但需要额外操作。

对于**纯 FM 模型**：旋转后什么都不用做。FM 天然就是旋转不变的。

对于 **DeepFM**：两者的混合——一次旋转同时影响两个通路，FM 侧自动消化，DNN 侧手动补偿。DeepFM 恰好展示了从"自动保持"到"需要补偿"的完整谱系。

**论文中的表述**：

> Rotation in embedding space is the natural action of $O(k)$ on the shared parameter manifold of DeepFM. The FM component is invariant under this action; the DNN component is equivariant with a one-step weight compensation. This unified treatment of both components is a structural consequence of DeepFM's architecture, not an ad hoc design choice.

---

## 6.3 创新的意义与价值：三层论证

### 第一层：现有方法的盲区

所有现有 DP 方法在推荐模型上的根本问题是**精度与隐私的二律背反**：

| 方法 | 论文 | 精度代价 | 保护范围 |
|------|------|----------|----------|
| DP-SGD | Abadi et al. 2016 | AUC ↓ 2-5% (ε=4) | 训练数据（MIA） |
| PATE | Papernot et al. 2017 | 需教师集成 | 训练数据（MIA） |
| LDP (输入扰动) | Duchi et al. 2013 | AUC ↓ 5-15% | 训练数据（MIA） |
| Model Encryption | — | 推理时解密开销 | 模型权重 |

注意：没有任何一行覆盖了"EIA 防御"。这个列在所有方法中都是空白的。

**DP-SGD 保护了训练数据，但模型训练完后，Embedding 以明文浮点数存在于 .pth 文件中。白盒权重读取是一个真实的部署场景威胁，在现有文献中没有被形式化地讨论过。**

### 第二层：DualGuard 填补的空白

DualGuard 的核心贡献不依赖于在单个维度上超越 SOTA：

$$
\text{贡献} = \text{覆盖了 SOTA 没有覆盖的攻击面} + \text{零额外精度代价}
$$

具体而言：
1. **EIA 是真实威胁**：模型文件部署到边缘设备、CDN 节点或第三方云环境时，权重被物理读取的风险是真实存在的。
2. **RotEmb 防御 EIA 的数学基础坚实**：定理 3 给出 $\left\|\mathbf{v}\right\|^2$ 的 MSE 下界——不是启发式防御，是有下界保证的。
3. **与 DP-SGD 的组合不是简单拼接**：定理 4 严格证明了互补性，两层不存在冗余。

### 第三层：用一张说服力表收束

| 攻击场景 | 对手能力 | 信息源 | DP-SGD only | RotEmb only | **DualGuard** |
|----------|----------|--------|-------------|-------------|---------------|
| 黑盒 MIA | 调 API | 输出概率值 | **防** | 不防 | **防** |
| 白盒 EIA | 读 .pth | 模型权重 | 不防 | **防** | **防** |
| 联合攻击 | API + 权重 | 输出 + 权重 | 各防一半 | 各防一半 | **全覆盖** |

**这个表是论文最有说服力的部分。** 审稿人看到它就会理解：之前的论文都在填第一行，没有人在填第二行。而你的工作同时填了两行——且填第二行的成本（RotEmb）是零精度代价。

### 和 SOTA 对比时怎么说

论文中不需要说"我们的 DP-SGD 比 XXX 的 DP-SGD 更好"——那本来就不是你的创新。正确的表述是：

> We adopt the standard DP-SGD (Abadi et al., 2016) with Rényi DP accounting (Mironov, 2017) as the training-time protection layer. Our contribution is not in improving DP-SGD per se, but in identifying a threat model—white-box embedding inversion—that DP-SGD does not address, and proposing a complementary post-training rotation mechanism (RotEmb) with formal reconstruction lower bounds to fill this gap. Used together, the two layers cover both query-based and weight-access-based privacy attacks on deployed DeepFM models.

### 这个创新够 CCF-C 吗？

**够。** CCF-C 会议的录用标准是：
- 问题定义清楚 ✓（EIA 威胁在推荐模型部署中真实存在）
- 方法有理论支撑 ✓（三条定理）
- 实验充分 ✓（Criteo + Avazu，MIA + EIA 双评估）
- 有 novelty ✓（EIA 防御这个空白+双层互补框架，之前没人做过）

不被录用的风险不在于"创新不够"，而在于"实验做少了"或"论文写得不清楚"。
