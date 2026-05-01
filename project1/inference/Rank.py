import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import pickle
import torch
import config
from datetime import datetime
from model import DeepFM

class Rank:
    def __init__(self):
        with open(config.user_history_path,'rb') as f:
            self.user_history=pickle.load(f)
        with open(config.n_movies_path,'rb') as f:
            self.n_movies=pickle.load(f)
        with open(config.n_types_path,'rb') as f:
            self.n_types=pickle.load(f)
        with open(config.movie_encoder_path,'rb') as f:
            self.movie_encoder=pickle.load(f)
        self.movie_data=pd.read_csv(config.movies_path)
        self.n_users=len(self.user_history)

        self.original_to_encoded={}
        for i,orig_id in enumerate(self.movie_encoder.classes_):
            self.original_to_encoded[int(orig_id)]=i+1
        self.encoded_to_original={}
        for orig_id,enc_id in self.original_to_encoded.items():
            self.encoded_to_original[enc_id]=orig_id

        # 只读表头获取标签列名，_build_tags_matrix 中按需加载
        header = pd.read_csv(config.train_path, nrows=0)
        self.tag_cols = [col for col in header.columns if col.startswith('tag_')]
        self.n_tags = len(self.tag_cols)

        state=torch.load(config.DeepFM_path,map_location=config.device,weights_only=True)
        hp=state.get('hyper_params',{}) if isinstance(state,dict) and 'state_dict' in state else {}
        state_dict=state['state_dict'] if isinstance(state,dict) and 'state_dict' in state else state

        ue=state_dict.get('user_embedding.weight',None)
        me=state_dict.get('movie_embedding.weight',None)
        if ue is not None:
            self.n_users=ue.shape[0]
            dim=ue.shape[1]
        elif hp and 'dim' in hp:
            dim=hp['dim']
        else:
            dim=config.dim
        if me is not None:
            self.n_movies=me.shape[0]

        has_tags='tag_embedding.weight' in state_dict
        if has_tags:
            tag_shape=state_dict['tag_embedding.weight'].shape
            n_tags=tag_shape[1] if len(tag_shape)==2 else 0
        else:
            n_tags=0

        dnn0_w=state_dict.get('dnn.0.weight',None)
        if dnn0_w is not None:
            actual_input_dim=dnn0_w.shape[1]
            if actual_input_dim!=dim*(7 if has_tags else 6):
                n_types_from_input=actual_input_dim-dim*5-dim*(1 if has_tags else 0)
                if n_types_from_input>0:
                    self.n_types=n_types_from_input
        else:
            n_tags=hp.get('n_tags',self.n_tags) if hp else self.n_tags

        dnn_layers=[]
        i=0
        while f'dnn.{i}.weight' in state_dict:
            out_dim=state_dict[f'dnn.{i}.weight'].shape[0]
            dnn_layers.append(out_dim)
            i+=3
        if dnn_layers:
            hidden_dim=dnn_layers
        else:
            hidden_dim=[64,128,64,32]

        self.model=DeepFM(self.n_users,self.n_movies,dim=dim,hidden_dim=hidden_dim,n_types=self.n_types,n_tags=n_tags)
        self.model.load_state_dict(state_dict,strict=False)
        self.model.to(config.device)
        self.model.eval()
        self.types_matrix=None
        self.tags_matrix=None

    def _build_types_matrix(self):
        """从训练数据的列名中提取类型列表，与训练时 MultiLabelBinarizer 的顺序一致。"""
        type_cols = [
            col for col in pd.read_csv(config.train_path, nrows=1).columns
            if col not in ('userId', 'movieId', 'rating', 'hour', 'day_of_week', 'month')
            and not col.startswith('tag_')
        ]
        typeidx = {t: i for i, t in enumerate(type_cols)}

        matrix = np.zeros((self.n_movies, self.n_types), dtype=np.float32)
        for _, row in self.movie_data.iterrows():
            orig_mid = int(row['movieId'])
            enc_mid = self.original_to_encoded.get(orig_mid)
            if enc_mid is None:
                continue
            for t in str(row.get('genres', '')).split('|'):
                idx = typeidx.get(t)
                if idx is not None and idx < self.n_types:
                    matrix[enc_mid, idx] = 1.0
        self.types_matrix = matrix

    def _build_tags_matrix(self):
        if self.n_tags == 0:
            return

        # 只加载需要的列，避免读全量 10M 行
        cols_to_load = ['movieId'] + self.tag_cols
        tag_data = pd.read_csv(config.train_path, usecols=cols_to_load)
        movie_tag_df = tag_data.groupby('movieId', sort=False).first()
        del tag_data

        matrix = np.zeros((self.n_movies, self.n_tags), dtype=np.float32)
        for enc_mid, row in movie_tag_df.iterrows():
            if 0 <= enc_mid < self.n_movies:
                matrix[enc_mid] = row.values
        self.tags_matrix = matrix

    def rank(self,user_id,movie_interest_id,top_n=10):
        if not movie_interest_id:
            return []

        movie_interest_id=[mid for mid in movie_interest_id if 0<=mid<self.n_movies]
        if not movie_interest_id:
            return []

        features=self.build_features(user_id,movie_interest_id)
        scores=self.predict(features)

        rank_movie=sorted(zip(movie_interest_id,scores),key=lambda x:x[1],reverse=True)
        return rank_movie[:top_n]

    def batch_rank(self,requests,top_n=10):
        result=[]
        for req in requests:
            userid=req['user_id']
            cans=req['candidates']
            ranked=self.rank(userid,cans,top_n)
            result.append(ranked)
        return result

    def build_features(self,user_id,movie_interest_id):
        n=len(movie_interest_id)
        now=datetime.now()
        hour=now.hour
        day=now.weekday()
        month=now.month

        user_ids=torch.tensor([user_id]*n).to(config.device).long()
        movie_ids=torch.tensor(movie_interest_id).to(config.device).long()
        hour_ids=torch.tensor([hour]*n).to(config.device).long()
        day_ids=torch.tensor([day]*n).to(config.device).long()
        month_ids=torch.tensor([month]*n).to(config.device).long()

        if self.types_matrix is None:
            self._build_types_matrix()

        select_type=self.types_matrix[movie_interest_id]
        types=torch.tensor(select_type).to(config.device).float()

        model_n_tags=self.model.n_tags
        if self.tags_matrix is None and model_n_tags>0:
            self._build_tags_matrix()

        if self.tags_matrix is not None and model_n_tags>0:
            select_tags=self.tags_matrix[movie_interest_id]
            tags=torch.tensor(select_tags).to(config.device).float()
        else:
            tags=torch.zeros(n,model_n_tags).to(config.device).float()

        return {
            'user_ids':user_ids,
            'movie_ids':movie_ids,
            'hours':hour_ids,
            'days':day_ids,
            'months':month_ids,
            'types':types,
            'tags':tags
        }

    def get_recommendation(self,user_id,recall_list,top_n=10):
        rank_movie=self.rank(user_id,recall_list,top_n)

        details=[]
        for movie_id,score in rank_movie:
            orig_id=self.encoded_to_original.get(movie_id,movie_id)
            row=self.movie_data[self.movie_data['movieId']==orig_id]
            if not row.empty:
                details.append({
                    'movieid':int(orig_id),
                    'title':row.iloc[0].get('title','未知'),
                    'type':row.iloc[0].get('genres','未知'),
                    'score':round(float(score),4)
                })
            else:
                details.append({
                    'movieid':int(orig_id),
                    'title':'未知',
                    'type':'未知',
                    'score':round(float(score),4)
                })

        return {
            'userid':user_id,
            'recommendations':details,
            'total_interests':len(recall_list),
            'returned':len(details)
        }

    def predict(self,features):
        with torch.no_grad():
            out=self.model(
                features['user_ids'],
                features['movie_ids'],
                features['hours'],
                features['days'],
                features['months'],
                features['types'],
                features['tags']
            )
            scores=torch.sigmoid(out).cpu().numpy()
        return scores
