import os
import torch

# ── 项目根目录：config.py 所在目录 ──
_project_root = os.path.dirname(os.path.abspath(__file__))


def _p(relative: str) -> str:
    return os.path.join(_project_root, relative)


# ── 原始数据路径 ──
ratings_path = _p("data/origin_data/ratings.csv")
movies_path = _p("data/origin_data/movies.csv")
tags_path = _p("data/origin_data/tags.csv")
links_path = _p("data/origin_data/links.csv")

# ── 预处理后数据路径 ──
train_path = _p("data/splited_data/train.csv")
val_path = _p("data/splited_data/val.csv")
test_path = _p("data/splited_data/test.csv")
neg_path = _p("data/splited_data/neg.csv")

processed_data_path = _p("data/processed_data")
user_history_path = _p("data/processed_data/user_history.pkl")
user_encoder_path = _p("data/processed_data/user_encoder.pkl")
movie_encoder_path = _p("data/processed_data/movie_encoder.pkl")
movie_faiss_path = _p("data/processed_data/movie_faiss.faiss")
movie_meta_path = _p("data/processed_data/movie_meta.pkl")
n_movies_path = _p("data/processed_data/n_movies.pkl")
n_types_path = _p("data/processed_data/n_types.pkl")

# ── 训练好的模型权重 ──
MIND_path = _p("trained_model/Mind.pth")
DeepFM_path = _p("trained_model/Deepfm.pth")
Multiheadattention_path = _p("trained_model/Multihead_interest.pth")

# ── 正负样本比例 ──
neg_ratio_for_deepfm = 4
n_neg_mind = 64

# ── 模型参数 ──
dim = 128
n_interest = 4
route = 3
max_history_length = 50
n_heads = 8
mind_dropout = 0.3
multihead_dropout = 0.3

# ── 训练参数 ──
batch_size = 2048
lr = 0.0001
epoch = 50
early_stop_p = 10
stop_delta = 0.0005
reduce_rate = 0.5
weight_decay = 1e-4
max_grad_norm = 5.0
num_workers = 10
pin_memory = True

# ── 设备 ──
device = "cuda"
seed = 42
