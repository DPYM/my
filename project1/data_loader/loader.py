import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder,MultiLabelBinarizer
import random
import sys
import os
import numpy as np
parent_dir=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)
import config
import gc

#读取数据
ratings_df=pd.read_csv(
    config.ratings_path,
    dtype={'userId':'int32','movieId':'int32','rating':'float32','timestamp':'int64'},
)
print(ratings_df.head())

movies_df=pd.read_csv(
    config.movies_path,
    dtype={'movieId':'int32','genres':'object'}
)
print(movies_df.head())

tags_df=pd.read_csv(
    config.tags_path,
    dtype={'userId':'int32','movieId':'int32','timestamp':'int64'}
)
tags_df['tag']=tags_df['tag'].fillna('').astype(str)
print(tags_df.head())

#合并ratings和movies
print('开始合并基础数据')
data=ratings_df.merge(movies_df,on='movieId',how='left')
print(f'合并后数据量: {len(data)}')

#截断数据（保留前1000万条，避免训练时间过长）
data=data.head(10000000).reset_index(drop=True)

#处理用户标签特征
print('开始处理用户标签')
tags_df=tags_df[tags_df['tag'].notna()&(tags_df['tag'].str.strip()!='')]
tags_df['tag']=tags_df['tag'].astype(str).str.lower().str.strip()

tag_counts=tags_df['tag'].value_counts()
top_n_tags=50
all_tags=tag_counts.head(top_n_tags).index.tolist()
print(f'标签总数: {len(tag_counts)}, 保留Top-{top_n_tags}: {len(all_tags)}')

movie_tags=tags_df[tags_df['tag'].isin(all_tags)].groupby('movieId')['tag'].apply(lambda x:'|'.join(x)).reset_index()
movie_tags.columns=['movieId','movie_tags']

mlb_tags=MultiLabelBinarizer(classes=all_tags)
movie_tags['tag_list']=movie_tags['movie_tags'].apply(lambda x:[t for t in x.split('|') if t] if x else [])
tags_encode=mlb_tags.fit_transform(movie_tags['tag_list'])
tags_feature_df=pd.DataFrame(tags_encode,columns=[f'tag_{t}' for t in all_tags],dtype=np.float32)
movie_tags_df=pd.concat([movie_tags[['movieId']],tags_feature_df],axis=1)

#释放内存
del tags_df,mlb_tags,tags_encode
gc.collect()

#合并标签特征到主数据（使用原始movieId合并后再编码）
print('开始合并标签特征')
data=data.merge(movie_tags_df,on='movieId',how='left')
tag_cols=[f'tag_{t}' for t in all_tags]
data[tag_cols]=data[tag_cols].fillna(0).astype(np.float32)

#释放内存
del movie_tags_df
gc.collect()

#用户id编码，标签编码
print('开始用户编码')
user_encoder=LabelEncoder()
data['userId']=user_encoder.fit_transform(data['userId'])#电影id编码，标签编码
print('开始电影编码')
movie_encoder=LabelEncoder()
data['movieId']=movie_encoder.fit_transform(data['movieId'])+1
n_movies=int(data['movieId'].max())+1
with open(config.n_movies_path,'wb') as f:
    pickle.dump(n_movies,f)

#对于电影类别采用多热编码
print('开始电影类型编码')
data['genres']=data['genres'].fillna('').astype(str)
data['movietype_list']=data['genres'].apply(lambda x:x.split('|'))
all_types=set()
for movie_type in data['movietype_list']:
    all_types.update(movie_type)
all_types=sorted(list(all_types))
mlb=MultiLabelBinarizer(classes=all_types)
types_encode=mlb.fit_transform(data['movietype_list'])
types_df=pd.DataFrame(types_encode,columns=[f'{t}' for t in all_types],dtype=np.float32)
with open(config.n_types_path,'wb') as f:
    pickle.dump(len(all_types),f)

#日期转换
print('开始日期编码')
data['timestamp']=pd.to_datetime(data['timestamp'],unit='s')
date_features=pd.DataFrame({
    'hour':data['timestamp'].dt.hour.astype(np.int8),
    'day_of_week':data['timestamp'].dt.dayofweek.astype(np.int8),
    'month':data['timestamp'].dt.month.astype(np.int8)
},index=data.index)
data=pd.concat([data,date_features],axis=1)

#优化数据类型减少内存
print('优化数据类型')
data['userId']=data['userId'].astype(np.int32)
data['movieId']=data['movieId'].astype(np.int32)
data['rating']=data['rating'].astype(np.float32)
data[tag_cols]=data[tag_cols].astype(np.float32)

#合并特征
print('开始合并特征')
features_data=pd.concat([
    data[['userId','movieId','rating','timestamp','hour','day_of_week','month']],types_df,data[tag_cols]],
    axis=1
)

#释放内存
del data,types_df
gc.collect()

#划分训练集，验证集，测试集，比例为7:1:2
train_idx=[]
val_idx=[]
test_idx=[]

#创建用户历史行为列表
user_history={}
print('开始分类')
for user_id,user_val in features_data.groupby('userId'):
    #按时间排序
    user_val=user_val.sort_values('timestamp')
    n=len(user_val)
    #把行为少于三次的直接放训练集里面
    if n<3:
        train_idx.extend(user_val.index.tolist())
        continue
    train_end=int(n*0.7)
    val_end=int(n*0.8)

    movie_idx=user_val['movieId'].tolist()
    user_history[user_id]=movie_idx[:train_end]

    train_user=user_val.index[:train_end]
    val_user=user_val.index[train_end:val_end]
    test_user=user_val.index[val_end:]

    train_idx.extend(train_user)
    val_idx.extend(val_user)
    test_idx.extend(test_user)
    
        
print('合并数据')
#合并数据
train_df=features_data.loc[train_idx].copy()
val_df=features_data.loc[val_idx].copy()
test_df=features_data.loc[test_idx].copy()

#优化数据类型
for df in [train_df,val_df,test_df]:
    df['userId']=df['userId'].astype(np.int32)
    df['movieId']=df['movieId'].astype(np.int32)
    df['rating']=df['rating'].astype(np.float32)
    df['hour']=df['hour'].astype(np.int8)
    df['day_of_week']=df['day_of_week'].astype(np.int8)
    df['month']=df['month'].astype(np.int8)

train_df=train_df.drop(columns='timestamp')
val_df=val_df.drop(columns='timestamp')
test_df=test_df.drop(columns='timestamp')

print('采集负样本')
#采集负样本（用户没有交互过的样本+评分小于4的样本）
movie_features_cols=['movieId']+[f'{t}' for t in all_types]
movie_info_df=features_data[movie_features_cols].drop_duplicates('movieId')
user_all_movie=set(range(1,n_movies))
user_watched_movies=features_data.groupby('userId')['movieId'].apply(set).to_dict()

#释放内存
del features_data
gc.collect()

print('开始保存文件')
#保存文件
train_df.to_csv(config.train_path,index=False)
val_df.to_csv(config.val_path,index=False)
test_df.to_csv(config.test_path,index=False)

#释放内存
del val_df,test_df
gc.collect()

neg_ratio=config.neg_ratio_for_deepfm
train_pos_df=train_df[train_df['rating']>=4]

user_ids,movie_ids,hour,day,month=[],[],[],[],[]
for user_id,user_pos in train_pos_df.groupby('userId'):
    user_pos_movie=user_watched_movies.get(user_id,set())
    neg_pools=list(user_all_movie-user_pos_movie)
    if not neg_pools:
        continue

    n_neg=int(len(user_pos)*neg_ratio)
    #假如负样本池的数量小于负样本数，就采取重复采样
    if len(neg_pools)<n_neg:
        n_neg=len(neg_pools)
    
    neg_sample=random.sample(neg_pools,int(n_neg))
    random_row=user_pos.sample(n=1).iloc[0]

    user_ids.extend([user_id]*len(neg_sample))
    movie_ids.extend(neg_sample)
    hour.extend([random_row['hour']]*len(neg_sample))
    day.extend([random_row['day_of_week']]*len(neg_sample))
    month.extend([random_row['month']]*len(neg_sample))

print('开始保存负样本')
neg_df=pd.DataFrame({
    'userId':np.array(user_ids,dtype=np.int32),
    'movieId':np.array(movie_ids,dtype=np.int32),
    'rating':0.0,
    'hour':np.array(hour,dtype=np.int8),
    'day_of_week':np.array(day,dtype=np.int8),
    'month':np.array(month,dtype=np.int8)
})
if not neg_df.empty:
    neg_df=neg_df.merge(movie_info_df,on='movieId',how='left')

train_neg_df=pd.concat([train_df,neg_df],axis=0,ignore_index=True)
train_neg_df.to_csv(config.neg_path,index=False)

print('保存历史序列')
with open(config.user_history_path,'wb') as f:
    pickle.dump(user_history,f)
with open(config.user_encoder_path,'wb') as f:
    pickle.dump(user_encoder,f)
with open(config.movie_encoder_path,'wb') as f:
    pickle.dump(movie_encoder,f)
print('已保存')