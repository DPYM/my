import torch
import pandas as pd
import pickle
import config
import random
from torch.utils.data import Dataset

#加载数据
class Mind_Dataset(Dataset):
    def __init__(self,data_path,user_history_path,max_length=50,n_movies=None,n_neg=4):
        origin_data=pd.read_csv(data_path)
        self.train_data=origin_data[origin_data['rating']>=4].reset_index(drop=True)
        low_data=origin_data[origin_data['rating']<4]
        self.neg_data={}
        for uid,group in low_data.groupby('userId'):
            self.neg_data[uid]=set(group['movieId'].values)
        with open(user_history_path,'rb') as f:
            user_history=pickle.load(f)

        self.user_ids=torch.tensor(self.train_data['userId'].astype(int).values).long()
        self.movie_ids=torch.tensor(self.train_data['movieId'].astype(int).values).long()
        self.user_history=user_history
        self.max_length=max_length
        self.n_neg=n_neg
        if n_movies is not None:
            self.all_movies=set(range(1,n_movies))
        else:
            self.all_movies=set(range(1,self.movie_ids.max().item()+1))
        self.user_history_set={}
        for uid,hist in user_history.items():
            self.user_history_set[uid]=set(hist)

    def __len__(self):
        return len(self.train_data)
    
    def __getitem__(self,idx):
        idx=int(idx)
        userid=self.user_ids[idx]
        pos_movie=self.movie_ids[idx]
        uid=userid.item()
        pos_item=pos_movie.item()

        user_history_interacted=self.user_history_set.get(uid,set())
        low_rating=self.neg_data.get(uid,set())
        uninteract_pools=self.all_movies-user_history_interacted-{pos_item}
        neg_pools=list(uninteract_pools.union(low_rating)-{pos_item})
        if len(neg_pools)==0:
            neg_pools=list(self.all_movies-{pos_item})
        if len(neg_pools)<self.n_neg:
            neg_movie=random.choices(neg_pools,k=self.n_neg)
        else:
            neg_movie=random.sample(neg_pools,k=self.n_neg)

        user_history_interest=self.user_history.get(uid,[])
        filtered_history=[m for m in user_history_interest if m!=pos_item]

        if len(filtered_history)>self.max_length:
            filtered_history=filtered_history[:self.max_length]
        elif len(filtered_history)<self.max_length:
            pad_len=self.max_length-len(filtered_history)
            filtered_history=filtered_history+[0]*pad_len

        return {
            'userid':userid,
            'pos_movie':pos_movie,
            'neg_movie':torch.tensor(neg_movie).long(),
            'user_history':torch.tensor(filtered_history).long()
        }

class DeepFM_Dataset(Dataset):
    def __init__(self,data_path):
        train_data=pd.read_csv(data_path)
        self.train_data=train_data
        self.user_ids=torch.tensor(train_data['userId'].astype(int).values).long()
        self.movie_ids=torch.tensor(train_data['movieId'].astype(int).values).long()
        self.hour_ids=torch.tensor(train_data['hour'].astype(int).values).long()
        self.day_ids=torch.tensor(train_data['day_of_week'].astype(int).values).long()
        self.month_ids=torch.tensor(train_data['month'].astype(int).values-1).long()
        # 从 CSV 列名动态发现类型列，与 loader.py 的 MultiLabelBinarizer 保持一致
        self.type_cols = [
            col for col in train_data.columns
            if col not in ('userId', 'movieId', 'rating', 'hour', 'day_of_week', 'month')
            and not col.startswith('tag_')
        ]
        self.types_ids = torch.tensor(train_data[self.type_cols].astype(int).values).float()
        
        tag_cols=[col for col in train_data.columns if col.startswith('tag_')]
        if tag_cols:
            self.tag_ids=torch.tensor(train_data[tag_cols].astype(int).values).float()
        else:
            self.tag_ids=torch.zeros(len(train_data),1).float()
        
        self.labels=torch.tensor((train_data['rating'].astype(int)>=4).astype(int).values).float()

    def __len__(self):
        return len(self.train_data)
    
    def __getitem__(self,idx):
        idx=int(idx)
        return {
            'userid':self.user_ids[idx],
            'movieid':self.movie_ids[idx],
            'label':self.labels[idx],
            'types':self.types_ids[idx],
            'tags':self.tag_ids[idx],
            'hour':self.hour_ids[idx],
            'day':self.day_ids[idx],
            'month':self.month_ids[idx],
        }

class Multihead_Dataset(Dataset):
    def __init__(self,data_path,user_history_path,max_length=50):
        train_data=pd.read_csv(data_path)
        self.train_data=train_data
        self.user_ids=torch.tensor(train_data['userId'].astype(int).values).long()
        self.movie_ids=torch.tensor(train_data['movieId'].astype(int).values).long()
        self.labels=torch.tensor((train_data['rating'].astype(int)>=4).astype(float).values)
        with open(user_history_path,'rb') as f:
            self.user_history=pickle.load(f)
        self.max_length=max_length
        
        tag_cols=[col for col in train_data.columns if col.startswith('tag_')]
        if tag_cols:
            self.tag_cols=tag_cols
            self.n_tags=len(tag_cols)
            movie_tag_df=train_data.groupby('movieId')[tag_cols].first()
            max_mid=int(train_data['movieId'].max())+1
            self.movie_tag_matrix=torch.zeros((max_mid,self.n_tags),dtype=torch.float32)
            for mid in movie_tag_df.index:
                self.movie_tag_matrix[int(mid)]=torch.tensor(movie_tag_df.loc[mid].values,dtype=torch.float32)
        else:
            self.tag_cols=[]
            self.n_tags=0
            self.movie_tag_matrix=torch.zeros((1,1),dtype=torch.float32)
    
    def __len__(self):
        return len(self.user_ids)
    
    def __getitem__(self, idx):
        uid=self.user_ids[idx].item()
        hist=self.user_history.get(uid,[])
        if len(hist)>self.max_length:
            hist=hist[:self.max_length]
        elif len(hist)<self.max_length:
            hist=hist+[0]*(self.max_length-len(hist))
        
        hist_tensor=torch.tensor(hist).long()
        hist_tags=self.movie_tag_matrix[hist_tensor]
        
        return {
            'userid': self.user_ids[idx],
            'movieid': self.movie_ids[idx],
            'label': self.labels[idx],
            'user_history': hist_tensor,
            'history_tags': hist_tags
        }