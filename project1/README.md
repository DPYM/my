# RiverMind — 多模型电影推荐系统

基于 MovieLens-32M 数据集，实现了三种推荐模型（MIND、DeepFM、Multihead Attention）的训练、评估与推理，并构建了完整的「召回→粗排→精排→重排」推荐链路。

## 项目结构

```
project1/
├── config.py                  # 全局配置（路径、模型参数、训练参数）
├── test.py                    # 模型离线测试入口（共用 evaluate 模块）
├── model/
│   ├── mind.py                # MIND 多兴趣胶囊网络（动态路由 + 标签感知注意力）
│   ├── deepfm.py              # DeepFM 因子分解机 + 深度网络（FM + DNN）
│   ├── multihead_attention.py # 多头注意力兴趣提取模型
│   └── embedding.py           # 基础嵌入层
├── train/
│   ├── train_mind.py          # MIND 训练脚本
│   ├── train_deepfm.py        # DeepFM 训练脚本
│   └── train_multihead.py     # Multihead 训练脚本
├── data_loader/
│   ├── dataset.py             # Dataset 定义（Mind_Dataset / DeepFM_Dataset / Multihead_Dataset）
│   └── loader.py              # 数据预处理与划分
├── metrics/
│   ├── evaluate.py            # 共享评估函数（三模型共用，消除训练脚本与 test.py 的重复）
│   ├── Focal_Loss.py          # Focal Loss（BCE 变种）
│   ├── InfoNCE.py             # InfoNCE 对比学习损失（支持 In-batch Negatives）
│   ├── BPRloss.py             # BPR Pairwise 损失
│   ├── earlystop.py           # 早停机制
│   └── reducelr.py            # 学习率自适应衰减
├── inference/
│   ├── pipeline.py            # 推荐链路编排（召回→粗排→精排→重排 + 多种融合策略 + 缓存）
│   ├── recall.py              # 召回模块（MIND + Faiss 近邻检索）
│   ├── CoarseRank.py          # 粗排模块（Multihead Attention 打分）
│   ├── Rank.py                # 精排模块（DeepFM 打分）
│   ├── ab_test.py             # A/B 测试框架（分桶、埋点、指标对比）
│   └── translator.py          # 中英文电影名映射
├── trained_model/             # 训练好的模型权重
└── data/                      # 数据目录（已 gitignore）
```

## 模型介绍

### MIND — Multi-Interest Network with Dynamic Routing

多兴趣提取模型，通过动态路由算法从用户行为序列中提取多个兴趣胶囊，再通过标签感知注意力（Label-Aware Attention）与候选物品计算匹配分数。

- **位置编码**：正弦位置编码，保留用户行为序列的时序信息
- **动态路由**：将历史序列映射为 K 个兴趣向量，迭代聚类（softmax → 加权求和 → squash → 相似度回传）
- **兴趣精炼**：每个兴趣向量通过独立的权重矩阵进行线性变换
- **门控机制**：用用户嵌入调制兴趣向量（`(1 + gate) × interest`），注入个性化信号
- **标签感知注意力**：根据候选物品动态选择最相关的兴趣计算分数
- **多样性约束**：diversity loss 惩罚兴趣向量间的相似度，强制正交
- **训练**：InfoNCE Loss + In-batch Negatives + 64 个随机负样本
- **评估指标**：Recall@K、NDCG@K、Hit@K

### DeepFM — Factorization Machine + Deep Neural Network

结合 FM 一阶/二阶特征交叉与 DNN 高阶特征交互的 CTR 预估模型。

- **FM 一阶项**：用户、电影、时间（小时/星期/月份）、电影类型、标签各自独立的线性权重
- **FM 二阶项**：所有特征嵌入两两内积交叉（和平方 − 平方和）
- **DNN 层**：4 层全连接（64→128→64→32），ReLU + Dropout(0.3)
- **训练**：Focal Loss（alpha=0.2, gamma=2）+ 梯度裁剪 + AMP 混合精度
- **评估指标**：AUC、F1、准确率、精确率、召回率

### Multihead Attention — 多头注意力兴趣模型

用多头自注意力机制从用户行为序列中提取多兴趣表示，通过兴趣门控与融合层生成用户表征。

- **可学习兴趣原型**：K 个可学习的兴趣向量作为 query，从历史序列中聚合兴趣信息
- **多头注意力**：标准 Transformer 注意力块（QKV 投影 + 残差 + LayerNorm + FFN）
- **兴趣门控**：Sigmoid 网络学习每个兴趣维度的激活权重
- **兴趣融合**：用户嵌入与门控后的注意力输出拼接 → GELU → LayerNorm → 投影
- **评分函数**：兴趣矩阵与候选电影嵌入拼接 → 双层 MLP → 取 max over 兴趣
- **训练**：BCEWithLogitsLoss + 梯度裁剪 + AMP 混合精度
- **评估指标**：Recall@200、NDCG@200、Hit@200、AUC、F1

## 推理链路

```
用户 ID → 召回（MIND + Faiss，500~1000） → 粗排（Multihead Attention，200） → 精排（DeepFM，20） → 重排（多样性+热度惩罚） → 最终推荐列表
```

| 阶段 | 模型 | 候选量 | 说明 |
|------|------|--------|------|
| 召回 | MIND + Faiss IndexFlatIP | 500~1000 | K 个兴趣向量各检索 Top-N，合并去重 |
| 粗排 | Multihead Attention | 200 | 对召回结果批量打分，保留 Top-200 |
| 精排 | DeepFM | 20 | 对候选集逐个精准打分排序 |
| 重排 | 规则 | — | 类型多样性过滤（连续 3+ 同类型跳过）+ 热度惩罚 + 已看过滤 |

### 多种推荐策略

Pipeline 支持多种推荐策略，可通过不同方法切换：

| 方法 | 说明 |
|------|------|
| `recommend()` | 标准链路：召回 → 粗排 → 精排 → 重排 |
| `recommend_blended()` | 召回内积分与 DeepFM 分数 min-max 归一化后加权融合 |
| `recommend_zscore_blended()` | 同上，但用 Z-score 标准化替代 min-max |
| `recommend_rrf()` | Reciprocal Rank Fusion 融合召回和精排的排名 |
| `recommend_mmr()` | MMR（最大边际相关性）平衡相关性与多样性 |
| `recommend_pop_penalty()` | 热门惩罚：精排分数 × `(1 − β × popularity)` |
| `recommend_mmr_pop()` | MMR + 热门惩罚组合 |
| `recommend_no_coarse()` | 跳过粗排，召回直送精排 |
| `recommend_deepfm_only()` | 仅用热门做召回 + DeepFM 精排 |
| `recommend_by_movie()` | 基于电影嵌入的 Faiss 相似度推荐 |

### A/B 测试框架

`inference/ab_test.py` 提供 `ABTestManager`：
- MD5 用户 ID 分桶，按 ratio 分配实验组/对照组
- 实时记录各组延迟、覆盖度、类型多样性
- 支持组间指标对比和日志回溯

## 训练优化

| 技术 | 说明 |
|------|------|
| AMP 混合精度 | 自动 FP16/FP32 混合训练，速度提升 30-50%，显存减少约 40% |
| In-batch Negatives | 利用 batch 内其他用户的正样本作为额外负样本 |
| InfoNCE Loss | 对比学习损失，1正 vs N负同时对比 |
| Focal Loss | 缓解正负样本不均衡，关注难分样本 |
| 梯度裁剪 | max_norm=5.0，防止梯度爆炸 |
| 早停 + 学习率衰减 | 连续 10 轮指标不提升自动降低学习率（折半），最低 1e-5 |

## 代码优化

### 共享评估函数

三段验证逻辑（validate_MIND / validate_DeepFM / validate_Multihead）原先在训练脚本和 test.py 中各有一份，共约 300 行重复代码。现已聚合到 `metrics/evaluate.py` 中，训练脚本和 test.py 统一导入调用，消除不一致风险。

### 冷启动预热

`pipeline.warm_up()` 不再是空壳，实际触发：
1. Faiss 召回索引加载/构建
2. 用户的真实历史跑一遍完整推理链路（MIND + Faiss + Multihead + DeepFM）
3. 触发各模型的 CUDA kernel 编译，后续首次 recommend() 无冷启动延迟

## 环境依赖

- Python 3.8+
- PyTorch 2.0+
- pandas, numpy, scikit-learn
- faiss-cpu（召回模块）

## 数据准备

1. 下载 [MovieLens-32M](https://grouplens.org/datasets/movielens/32m/) 数据集
2. 将 `ratings.csv`、`movies.csv`、`links.csv`、`tags.csv` 放入 `data/origin_data/`
3. 运行数据预处理生成训练/验证/测试集及编码器文件

```bash
python -m data_loader.loader
```

> 注意：loader.py 默认截断至前 1000 万条评分数据以控制训练时间。如需全量数据可修改 `data = data.head(10000000)` 一行。

## 使用方法

### 训练

```bash
# 训练 MIND
python -u -m train.train_mind

# 训练 DeepFM
python -u -m train.train_deepfm

# 训练 Multihead Attention
python -u -m train.train_multihead
```

> `-u` 参数禁用输出缓冲，确保实时看到训练日志。

### 测试

```bash
python test.py
# 输入模型名称：mind / deepfm / multihead
```

### 推理

```python
from inference import Pipeline

pipe = Pipeline(recall_top_k=500, coarse_rank_top=200, fine_rank_top=20)
pipe.warm_up()
result = pipe.recommend(user_id=0, top_n=20)
```

## 配置参数

在 `config.py` 中修改（所有路径自动基于项目根目录计算，无需手动修改）：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `dim` | 128 | 嵌入维度 |
| `n_interest` | 4 | 兴趣向量数量（MIND/Multihead） |
| `route` | 3 | 动态路由迭代次数（MIND） |
| `n_heads` | 8 | 注意力头数（Multihead） |
| `mind_dropout` | 0.3 | MIND Dropout 率 |
| `multihead_dropout` | 0.3 | Multihead Dropout 率 |
| `batch_size` | 2048 | 批大小 |
| `lr` | 0.0001 | 学习率 |
| `epoch` | 50 | 最大训练轮数 |
| `early_stop_p` | 10 | 早停耐心值 |
| `stop_delta` | 0.0005 | 早停 / 学习率衰减的阈值 |
| `reduce_rate` | 0.5 | 学习率衰减倍数 |
| `max_grad_norm` | 5.0 | 梯度裁剪阈值 |
| `n_neg_mind` | 64 | MIND 随机负采样数量 |
| `weight_decay` | 1e-4 | L2 正则化 |
| `num_workers` | 10 | DataLoader 工作进程数 |
| `seed` | 42 | 随机种子 |
