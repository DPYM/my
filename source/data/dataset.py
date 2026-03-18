import torch
import pandas as pd
import pickle
from torch.utils.data import Dataset,DataLoader
dtype=torch.float32

train_data=pd.read_csv(r'C:/github/project1/data/splited_data/train.csv')
#print(train_df.head())

with open(r'C:/github/project1/data/processed_data/user_history.pkl','rb') as f:
    user_history=pickle.load(f)

#加载数据
class User_Movie_Dataset(Dataset):
    def __init__(self,train_data,user_history,max_seqlength=50):#注意这里传入的是已经读取的数据而非文件路径
        self.train_df=train_data
        self.user_ids=torch.tensor(train_data['userid'].values)#把用户id转化为张量，便于计算
        self.momvie_ids=torch.tensor(train_data['movieid'].values)
        self.labels=torch.tensor((train_data['rating']>=4).astype(int).values,dtype=dtype)#这里规定评分大于等于4作为用户感兴趣的评判标准
        self.user_history=user_history
        self.max_seqlength=max_seqlength#因为每个用户的电影交互数量不同，导致无法组成batch，所以定义了一个最大序列长度

    def __len__(self):
        return len(self.train_df)
    
    def __getitem__(self,idx):
        seq=self.user_history.get(self.user_ids[idx].item(),[])
        #假如电影序列长度超过阈值，就截断，反之电影长度不足的话就用0填充
        if len(seq)>self.max_seqlength:
            seq=seq[:self.max_seqlength]
        else:
            seq=seq+[0]*(self.max_seqlength-len(seq))
        seqq=torch.tensor(seq)
        return {
            'userid':self.user_ids[idx],
            'movieid':self.momvie_ids[idx],
            'label':self.labels[idx],
            'user_history_length':seq
}
    
train_dataset=User_Movie_Dataset(train_data,user_history)
train_loader=DataLoader(train_dataset,batch_size=256,shuffle=True)

#print(f'样本数：{len(train_dataset)}')
#sample=train_dataset[0]
#print(sample)