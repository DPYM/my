import torch
import torch.nn as nn
import config
torch.manual_seed(config.seed)

#定义词嵌入矩阵
class Embedding(nn.Module):
    def __init__(self,n_users,n_movies,dim):
        super(Embedding,self).__init__()
        self.user_embedding=nn.Embedding(n_users,dim)
        self.movie_embedding=nn.Embedding(n_movies,dim)

    def forward(self,user_ids,movie_ids):#前向传播
        user_vecs=self.user_embedding(user_ids)
        movie_vecs=self.movie_embedding(movie_ids)
        return user_vecs,movie_vecs

if __name__=='__main__':
    import pandas as pd
    import pickle
    train_data=pd.read_csv(config.train_path)

    with open(config.user_history_path,'rb') as f:
        user_history=pickle.load(f)

    for i in list(user_history.keys())[:5]:
        print(f'用户{i}的长度{len(user_history[i])}')

    all_movie_ids = []
    for movie_list in user_history.values():
        all_movie_ids.extend(movie_list)

    n_movies=max(all_movie_ids)+1
    n_users=len(user_history)
    print(f"电影总数: {n_movies}")