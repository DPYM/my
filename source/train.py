import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
import pandas as pd
import pickle
from model.embedding import Embedding
from data.dataset import User_Movie_Dataset
device=torch.device('cuda')

train_data=pd.read_csv(r'C:/github/project1/data/splited_data/train.csv')
#print(train_data.head())
#统计个数
with open(r'C:/github/project1/data/processed_data/user_history.pkl','rb') as f:
    user_history=pickle.load(f)

all_movie_ids = []
for movie_list in user_history.values():
    all_movie_ids.extend(movie_list)

n_movies=max(all_movie_ids)+1
n_users=len(user_history)

#数据加载
data_train=User_Movie_Dataset(
    train_data=train_data,
    user_history=user_history
)
#因为在dataset.py中，处理的是整个数据集而不是单个用户，导致.item()报错，所以这里把一个batch中的数据转化为tensor张量

train_loader=DataLoader(data_train,batch_size=256,shuffle=True)

model=Embedding(n_users,n_movies,dim=64)#定义模型
model=model.to(device)
opt=optim.Adam(model.parameters(),lr=0.0001)#采用自适应优化算法
critersion=nn.BCELoss()#采用交叉熵作为损失函数

epoch=100#循环100次
for i in range(5):
    total_loss=0
    for batch in train_loader:#到这一步发现因为batch中样本长度不同报错，所以定义了一个长度阈值，见(dataset.py)
        user_ids=batch['userid'].to(device)
        movie_ids=batch['movieid'].to(device)
        labels=batch['label'].to(device)

        #计算用户和电影的向量矩阵
        user_vecs,movie_vecs=model(user_ids,movie_ids)

        #矩阵分解，根据用户和电影的特征矩阵计算用户对电影的兴趣
        scores=torch.sigmoid((user_vecs*movie_vecs).sum(dim=1))

        loss=critersion(scores,labels)#计算损失
        
        opt.zero_grad()#清空梯度
        loss.backward()#反向传播
        opt.step()#更新权重
        total_loss+=loss.item()

    average_loss=total_loss/len(train_loader)
    print(f'第{i+1}次的平均损失为{average_loss:.5f}')