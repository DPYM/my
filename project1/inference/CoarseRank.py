import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import pickle
import pandas as pd
import numpy as np
import config
from model import Multihead_interest

class CoarseRank:
    def __init__(self):
        with open(config.user_history_path,'rb') as f:
            self.user_history=pickle.load(f)
        with open(config.n_movies_path,'rb') as f:
            n_movies=pickle.load(f)
        with open(config.user_encoder_path,'rb') as f:
            user_encoder=pickle.load(f)
        n_users=len(user_encoder.classes_)
        
        # 只读表头获取标签列名，避免加载全量 10M 行
        header = pd.read_csv(config.train_path, nrows=0)
        tag_cols = [col for col in header.columns if col.startswith('tag_')]
        n_tags = len(tag_cols)

        if n_tags > 0:
            # 只加载 movieId + 标签列，groupby 取首条后构建矩阵
            cols_to_load = ['movieId'] + tag_cols
            tag_data = pd.read_csv(config.train_path, usecols=cols_to_load)
            movie_tag_df = tag_data.groupby('movieId', sort=False).first()
            del tag_data
            max_mid = int(movie_tag_df.index.max()) + 1
            self.movie_tag_matrix = np.zeros((max_mid, n_tags), dtype=np.float32)
            for mid, row in movie_tag_df.iterrows():
                self.movie_tag_matrix[int(mid)] = row.values
        else:
            self.movie_tag_matrix = None
        self.n_tags = n_tags

        state=torch.load(config.Multiheadattention_path,map_location=config.device,weights_only=True)
        hp=state.get('hyper_params',{}) if isinstance(state,dict) and 'state_dict' in state else {}
        state_dict=state['state_dict'] if isinstance(state,dict) and 'state_dict' in state else state
        
        ue=state_dict.get('user_embedding.weight',None)
        me=state_dict.get('movie_embedding.weight',None)
        n_users=ue.shape[0] if ue is not None else n_users
        n_movies=me.shape[0] if me is not None else n_movies
        
        if hp and 'dim' in hp:
            dim=hp['dim']
            n_interests=hp.get('n_interests',config.n_interest)
            n_heads=hp.get('n_heads',config.n_heads)
            n_tags=hp.get('n_tags',self.n_tags)
        else:
            dim=ue.shape[1] if ue is not None else config.dim
            iv=state_dict.get('interest_vecs',None)
            n_interests=iv.shape[0] if iv is not None else config.n_interest
            n_heads=config.n_heads
            n_tags=self.n_tags
        self.model=Multihead_interest(
            n_users,n_movies,
            dim=dim,
            n_interests=n_interests,
            n_heads=n_heads,
            n_tags=n_tags
        )
        self.model.load_state_dict(state_dict)
        self.model.to(config.device)
        self.model.eval()

    def rank(self,user_id,movie_ids,top_n=200):
        if not movie_ids:
            return []

        hist=self.user_history.get(user_id,[])
        if len(hist)>config.max_history_length:
            hist=hist[:config.max_history_length]
        elif len(hist)<config.max_history_length:
            hist=hist+[0]*(config.max_history_length-len(hist))

        with torch.no_grad():
            user_id_t=torch.tensor([user_id],device=config.device).long()
            hist_t=torch.tensor([hist],device=config.device).long()
            movie_ids_t=torch.tensor(movie_ids,device=config.device).long().unsqueeze(0)
            
            if self.movie_tag_matrix is not None and self.n_tags>0:
                hist_np=np.array(hist)
                hist_tags_np=np.zeros((1,len(hist),self.n_tags),dtype=np.float32)
                for si in range(len(hist)):
                    mid=int(hist[si])
                    if mid>0 and mid<len(self.movie_tag_matrix):
                        hist_tags_np[0,si]=self.movie_tag_matrix[mid]
                hist_tags_t=torch.tensor(hist_tags_np,device=config.device).float()
            else:
                hist_tags_t=torch.zeros(1,len(hist),self.n_tags,dtype=torch.float,device=config.device)

            scores=self.model(user_id_t,hist_t,hist_tags_t,movie_ids_t)
            scores=scores.squeeze(0).cpu().numpy()

        ranked=sorted(zip(movie_ids,scores),key=lambda x:x[1],reverse=True)
        return ranked[:top_n]
