"""共享评估函数：供训练脚本的验证和 test.py 的离线测试共用。"""

import random

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)


def evaluate_mind(model, val_df, user_history, n_movies, device, K=50):
    """MIND 模型验证：对全量电影做暴力检索，计算 Recall / NDCG / Hit@K。"""
    model.eval()

    with torch.no_grad():
        all_movie_emb = model.movie_embedding.weight.data[1:]
    all_movie_emb = all_movie_emb.to(device)

    user_data = []
    for uid, group in val_df.groupby("userId"):
        pos_movies = set(group[group["rating"] >= 4]["movieId"].values)
        if len(pos_movies) == 0:
            continue

        labels = (group["rating"].values >= 4).astype(int)
        movie_ids = list(group["movieId"].values)

        hist = user_history.get(uid, [])
        if len(hist) > 50:
            hist = hist[:50]
        elif len(hist) < 50:
            hist = hist + [0] * (50 - len(hist))

        user_data.append((uid, movie_ids, labels, hist, len(pos_movies), set(hist)))

    all_recall = []
    all_ndcg = []
    all_hit = []

    with torch.no_grad():
        for uid, movie_ids, labels, hist, pos_count, hist_set in user_data:
            uid_t = torch.tensor([uid], device=device).long()
            hist_t = torch.tensor([hist], device=device).long()

            interest_matrix = model(uid_t, hist_t)

            scores = torch.matmul(interest_matrix, all_movie_emb.t())
            scores = scores.max(dim=1)[0].squeeze(0).cpu().numpy()

            exclude_set = hist_set - {0}
            for mid in exclude_set:
                if 1 <= mid < n_movies:
                    scores[mid - 1] = -1e9

            candidate_labels = np.zeros(n_movies - 1, dtype=int)
            for mid, lbl in zip(movie_ids, labels):
                if 1 <= mid < n_movies:
                    candidate_labels[mid - 1] = lbl

            sorted_indices = np.argsort(-scores)
            sorted_labels = candidate_labels[sorted_indices]

            top_k_labels = sorted_labels[:K]
            hit_count = top_k_labels.sum()
            recall = hit_count / pos_count
            all_recall.append(recall)
            all_hit.append(1 if hit_count > 0 else 0)

            positions = np.log2(np.arange(2, K + 2))
            dcg = np.sum(top_k_labels / positions)
            ideal_labels = np.sort(candidate_labels)[::-1][:K]
            idcg = np.sum(ideal_labels / positions)
            ndcg = dcg / (idcg + 1e-7)
            all_ndcg.append(ndcg)

    return {
        f"Recall@{K}": np.mean(all_recall),
        f"NDCG@{K}": np.mean(all_ndcg),
        f"Hit@{K}": np.mean(all_hit),
    }


def evaluate_deepfm(model, val_loader, device):
    """DeepFM 验证：在 val_loader 上全量跑一遍，计算 AUC / F1 / 准确率等。"""
    model.eval()
    all_scores = []
    all_labels = []

    with torch.no_grad():
        for batch in val_loader:
            user_ids = batch["userid"].to(device).long()
            movie_ids = batch["movieid"].to(device).long()
            hour = batch["hour"].to(device).long()
            day = batch["day"].to(device).long()
            month = batch["month"].to(device).long()
            types = batch["types"].to(device).float()
            tags = batch["tags"].to(device).float()
            labels = batch["label"].to(device).float()

            logits = model(user_ids, movie_ids, hour, day, month, types, tags)
            probs = torch.sigmoid(logits)

            all_scores.extend(probs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    label = np.array(all_labels).astype(int)
    scores = np.array(all_scores)

    precisions, recalls, thresholds = precision_recall_curve(label, scores)
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-7)
    best_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_idx] if best_idx < len(thresholds) else 0.5
    pred_label = (scores >= best_threshold).astype(int)

    return {
        "准确率": accuracy_score(label, pred_label),
        "精确率": precision_score(label, pred_label),
        "召回率": recall_score(label, pred_label),
        "f1分数": f1_score(label, pred_label),
        "auc分数": roc_auc_score(label, scores),
        "最佳阈值": best_threshold,
    }


def evaluate_multihead(
    model, val_df, user_history, n_movies, n_tags, device, K=200, n_candidates=1000
):
    """Multihead 验证：对每个用户采样候选集后打分，计算 Recall / NDCG / Hit / AUC / F1。"""
    model.eval()

    all_movie_ids = np.arange(1, n_movies)

    tag_cols = [col for col in val_df.columns if col.startswith("tag_")]
    if tag_cols and n_tags > 0:
        movie_tag_df = val_df.groupby("movieId")[tag_cols].first()
        max_mid = int(val_df["movieId"].max()) + 1
        movie_tag_matrix = np.zeros((max_mid, n_tags), dtype=np.float32)
        for mid in movie_tag_df.index:
            movie_tag_matrix[int(mid)] = movie_tag_df.loc[mid].values
    else:
        movie_tag_matrix = None

    user_data = []
    for uid, group in val_df.groupby("userId"):
        pos_movies = set(group[group["rating"] >= 4]["movieId"].values)
        if len(pos_movies) == 0:
            continue

        hist = user_history.get(uid, [])
        hist_set = set(hist)

        neg_pool = list(set(all_movie_ids) - pos_movies - hist_set)
        if len(neg_pool) == 0:
            continue

        n_neg = min(n_candidates - len(pos_movies), len(neg_pool))
        neg_sample = random.sample(neg_pool, n_neg)

        candidate_movies = list(pos_movies) + neg_sample
        labels = [1] * len(pos_movies) + [0] * len(neg_sample)

        if len(hist) > 50:
            hist_padded = hist[:50]
        elif len(hist) < 50:
            hist_padded = hist + [0] * (50 - len(hist))
        else:
            hist_padded = hist

        user_data.append((uid, candidate_movies, labels, hist_padded, len(pos_movies)))

    all_recall = []
    all_ndcg = []
    all_hit = []
    all_auc_scores = []
    all_f1_scores = []

    batch_size = 64
    with torch.no_grad():
        for i in range(0, len(user_data), batch_size):
            batch = user_data[i : i + batch_size]
            uids = [d[0] for d in batch]
            candidate_movies_list = [d[1] for d in batch]
            labels_list = [d[2] for d in batch]
            hists = [d[3] for d in batch]
            pos_counts = [d[4] for d in batch]

            max_cand = max(len(c) for c in candidate_movies_list)
            uid_t = torch.tensor(uids, device=device).long()
            hist_t = torch.tensor(hists, device=device).long()
            movie_ids_t = torch.zeros(
                len(batch), max_cand, dtype=torch.long, device=device
            )

            if movie_tag_matrix is not None and n_tags > 0:
                hist_np = np.array(hists)
                hist_tags_np = np.zeros(
                    (len(hists), len(hists[0]), n_tags), dtype=np.float32
                )
                for bi in range(len(hists)):
                    for si in range(len(hists[bi])):
                        mid = int(hists[bi][si])
                        if mid > 0 and mid < len(movie_tag_matrix):
                            hist_tags_np[bi, si] = movie_tag_matrix[mid]
                hist_tags_t = torch.tensor(hist_tags_np, device=device).float()
            else:
                hist_tags_t = torch.zeros(
                    len(batch), len(hists[0]), n_tags, dtype=torch.float, device=device
                )

            mask = torch.zeros(len(batch), max_cand, dtype=torch.bool, device=device)
            for j, (cand, lab) in enumerate(
                zip(candidate_movies_list, labels_list)
            ):
                n = len(cand)
                movie_ids_t[j, :n] = torch.tensor(cand, device=device).long()
                mask[j, :n] = True

            logits = model(uid_t, hist_t, hist_tags_t, movie_ids_t)
            probs = torch.sigmoid(logits).cpu().numpy()

            for j in range(len(batch)):
                n = len(candidate_movies_list[j])
                p = probs[j, :n]
                lab = labels_list[j]

                sorted_indices = np.argsort(-p)
                sorted_labels = np.array(lab)[sorted_indices]

                top_k_labels = sorted_labels[: min(K, len(sorted_labels))]
                hit_count = top_k_labels.sum()
                recall = hit_count / pos_counts[j]
                all_recall.append(recall)
                all_hit.append(1 if hit_count > 0 else 0)

                k_actual = len(top_k_labels)
                positions = np.log2(np.arange(2, k_actual + 2))
                dcg = np.sum(top_k_labels / positions)
                ideal_labels = np.sort(lab)[::-1][:k_actual]
                idcg = np.sum(ideal_labels / positions)
                ndcg = dcg / (idcg + 1e-7)
                all_ndcg.append(ndcg)

                label_arr = np.array(lab).astype(int)
                if len(np.unique(label_arr)) > 1:
                    all_auc_scores.append(roc_auc_score(label_arr, p))
                pred_label = (p >= 0.5).astype(int)
                all_f1_scores.append(f1_score(label_arr, pred_label))

    return {
        f"Recall@{K}": np.mean(all_recall),
        f"NDCG@{K}": np.mean(all_ndcg),
        f"Hit@{K}": np.mean(all_hit),
        "auc分数": np.mean(all_auc_scores) if all_auc_scores else 0.0,
        "f1分数": np.mean(all_f1_scores),
    }
