import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import pickle
import torch
import config
import faiss
import numpy as np
from model import Mind

class Recall:
    def __init__(self):
        with open(config.user_history_path,'rb') as f:
            user_history=pickle.load(f)
        with open(config.n_movies_path,'rb') as f:
            n_movies=pickle.load(f)
        n_users=len(user_history)
        state=torch.load(config.MIND_path,map_location=config.device,weights_only=True)
        hp=state.get('hyper_params',{}) if isinstance(state,dict) and 'state_dict' in state else {}
        state_dict=state['state_dict'] if isinstance(state,dict) and 'state_dict' in state else state
        
        if hp and 'dim' in hp:
            dim=hp['dim']
            n_interest=hp.get('n_interest',config.n_interest)
            route=hp.get('route',config.route)
            dropout=hp.get('dropout',config.mind_dropout)
        else:
            emb=state_dict.get('user_embedding.weight',None)
            if emb is not None:
                n_users=emb.shape[0]
                dim=emb.shape[1]
            else:
                dim=config.dim
            me=state_dict.get('movie_embedding.weight',None)
            if me is not None:
                n_movies=me.shape[0]
            ir=state_dict.get('interest_refine_weight',None)
            if ir is not None:
                n_interest=ir.shape[0]
            else:
                n_interest=config.n_interest
            route=config.route
            dropout=config.mind_dropout
        self.model=Mind(n_users,n_movies,dim=dim,n_interest=n_interest,route=route,dropout=dropout)
        self.model.load_state_dict(state_dict)
        self.model.to(config.device)
        self.model.eval()

        movie_emb_layer=self.model.movie_embedding.weight.data
        self.movie_emb=movie_emb_layer.cpu().numpy().astype(float)
        self.movie_id_map={i:i for i in range(self.movie_emb.shape[0])}

        self.nprobe=32
        self.index=None
    
    def ensure_index(self):
        """确保索引可用：优先从磁盘加载，加载失败则重新构建。"""
        self.load_index()
        if self.index is None:
            self.build_index()

    def build_index(self):
        """强制重建 Faiss 索引并持久化到磁盘。"""
        self.build_faiss_index()
        self.save_index()
        print('Faiss 索引已构建并保存')

    def recall(self, user_id, user_history, top_k=50, return_scores=False):
        self.ensure_index()
        if len(user_history)>config.max_history_length:
            user_history=user_history[:config.max_history_length]

        with torch.no_grad():
            user_id_tensor=torch.tensor([user_id]).to(config.device).long()
            user_history_tensor=torch.tensor(np.array([user_history])).to(config.device).long()
            interest_tensor=self.model(user_id_tensor,user_history_tensor).squeeze(0)
            interest_np=interest_tensor.cpu().numpy().astype(float)

        history_set=set(user_history)
        all_interest=[]
        history_movie=set()
        for i in range(interest_np.shape[0]):
            vec=interest_np[i:i+1]
            sims,idxs=self.search_index(vec,top_k)
            for sim,idx in zip(sims,idxs):
                movie_real_id=self.movie_id_map.get(idx,idx)
                if movie_real_id not in history_movie and movie_real_id not in history_set:
                    history_movie.add(movie_real_id)
                    all_interest.append((sim,movie_real_id))
        all_interest.sort(key=lambda x:x[0],reverse=True)
        if return_scores:
            return [movie_id for _,movie_id in all_interest],[sim for sim,_ in all_interest]
        return [movie_id for _,movie_id in all_interest]
    
    def batch_recall(self,user_id,user_histories,top_k_interest=50):
        return [self.recall(uid,h,top_k_interest) for uid,h in zip(user_id,user_histories)]

    def build_faiss_index(self):
        n_movies,dim=self.movie_emb.shape

        self.index=faiss.IndexFlatIP(dim)
        self.index.add(self.movie_emb)

        if hasattr(self.index,'nprobe'):
            self.index.nprobe=self.nprobe

    def search_index(self,vec,k):
        sims,idxs=self.index.search(vec,k)
        return sims[0],idxs[0]
    
    def save_index(self):
        index_cpu=self.index
        faiss.write_index(index_cpu,config.movie_faiss_path)
        
        meta={
            'movie_embedding':self.movie_emb,
            'movie_id_map':self.movie_id_map
        }
        with open(config.movie_meta_path,'wb') as f:
            pickle.dump(meta,f)

    def load_index(self):
        try:
            #加载索引
            self.index=faiss.read_index(config.movie_faiss_path)
        except:
            return
        if hasattr(self.index,'nprobe'):
            self.index.nprobe=self.nprobe

        with open(config.movie_meta_path,'rb') as f:
            meta=pickle.load(f)
        self.movie_emb=meta['movie_embedding']
        self.movie_id_map=meta['movie_id_map']